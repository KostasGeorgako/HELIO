"""
Real Estate Beyond RGB — web backend.

Endpoints implement the handoff spec exactly, scoped to Path B (no follow-up
endpoint, no live training chart endpoint — those are deliberately omitted).

Run:
    uvicorn app:app --reload --port 5050
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from llm import chat_json, chat_text
from pipeline import (
    DATA_DIR, ENMAP_DIR, cache_info, get_results, get_status,
    stage_uploaded_sites, start_run,
)
from prompts import (
    prompt_narrate, prompt_parse_selection, prompt_parse_usecase,
    prompt_selection_confirm, prompt_usecase_confirm, prompt_welcome,
)

# ─── App ──────────────────────────────────────────────────────────────────

app = FastAPI(title="HELIO — Real Estate Beyond RGB", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

UPLOADS_DIR = Path(os.environ.get("UPLOADS_DIR", "uploads")).resolve()
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Serve plot images
if DATA_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DATA_DIR)), name="static")
else:
    print(f"[app] WARNING: DATA_DIR not found at {DATA_DIR} — plots won't serve")

# ─── Session state (in-memory; one process only — good enough for demo) ──

SESSIONS: dict[str, dict[str, Any]] = {}

DISPLAY_NAMES = {
    "arkadia": "Arkadia",
    "arkadia2": "Arkadia 2",
    "magnisia": "Magnisia",
    "veroia": "Veroia",
}


def _session(session_id: str) -> dict[str, Any]:
    if session_id not in SESSIONS:
        raise HTTPException(404, f"unknown session_id {session_id!r}")
    return SESSIONS[session_id]


# ─── Models ───────────────────────────────────────────────────────────────

class ChatIn(BaseModel):
    session_id: str
    message: str
    stage: str = "selection"  # "selection" | "usecase" — defaulted for safety


class RunIn(BaseModel):
    session_id: str
    sites: list[str]
    prices: dict[str, float]
    weights: dict[str, float]
    anomaly_sign: int = -1
    date_discounts: dict[str, float]
    use_case: str
    fast: bool = False  # reuse the cached run + pretrained model, re-score only


class SessionSaveIn(BaseModel):
    session_id: str
    messages: list[dict[str, Any]]
    results: dict[str, Any]


# ─── Health ───────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "HELIO API",
        "data_dir_exists": DATA_DIR.exists(),
        "enmap_dir_exists": ENMAP_DIR.exists(),
        "github_token_set": bool(os.environ.get("GITHUB_TOKEN")),
        "github_model": os.environ.get("GITHUB_MODEL", "gpt-4o-mini"),
        "endpoints": [
            "POST /api/upload",
            "POST /api/chat",
            "POST /api/run",
            "GET  /api/status/{job_id}",
            "GET  /api/results/{job_id}",
        ],
    }


# ─── /api/upload ──────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """Unzip, scan structure, generate welcome message."""
    session_id = uuid.uuid4().hex[:6]
    session_dir = UPLOADS_DIR / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    # Write the ZIP to disk
    zip_path = session_dir / "upload.zip"
    with open(zip_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Unzip
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(session_dir)
    except zipfile.BadZipFile:
        raise HTTPException(400, "not a valid ZIP file")

    # Scan for sites
    sites_found = _scan_sites(session_dir)
    if not sites_found:
        raise HTTPException(
            400,
            "no recognised sites found in upload. Expected enmap/<site>/<acquisition>/SPECTRAL_IMAGE.TIF",
        )

    # Build summary text for the welcome prompt
    summary_lines = []
    for key, info in sites_found.items():
        n = len(info["images"])
        summary_lines.append(
            f"  - {DISPLAY_NAMES.get(key, key)}: {n} acquisition{'s' if n != 1 else ''}"
        )
    sites_summary = "\n".join(summary_lines)

    welcome = chat_text(prompt_welcome(sites_summary))

    SESSIONS[session_id] = {
        "session_id": session_id,
        "sites_found": sites_found,
        "selection": None,
        "use_case_config": None,
        "job_id": None,
    }

    return {
        "session_id": session_id,
        "sites_found": sites_found,
        "total_sites": len(sites_found),
        "first_ai_message": welcome,
    }


def _scan_sites(root: Path) -> dict[str, dict[str, Any]]:
    """Find enmap/<site>/<mosaic_folder> structure in the unzipped upload."""
    result: dict[str, dict[str, Any]] = {}
    enmap = root / "enmap"
    if not enmap.exists():
        # Maybe the user zipped the enmap folder directly
        nested = next((p for p in root.iterdir() if (p / "enmap").exists()), None)
        if nested:
            enmap = nested / "enmap"
    if not enmap.exists():
        return result
    for site_dir in sorted(enmap.iterdir()):
        if not site_dir.is_dir() or site_dir.name not in DISPLAY_NAMES:
            continue
        images = sorted(p.name for p in site_dir.iterdir() if p.is_dir())
        if not images:
            continue
        result[site_dir.name] = {
            "images": images,
            "challenge_image": images[0],
            "extra_images": images[1:],
        }
    return result


# ─── /api/chat ────────────────────────────────────────────────────────────

@app.post("/api/chat")
def chat(req: ChatIn):
    """Branch on stage. Returns the AI reply and any extracted action."""
    session = _session(req.session_id)

    if req.stage == "selection":
        return _chat_selection(session, req.message)
    if req.stage == "usecase":
        return _chat_usecase(session, req.message)
    raise HTTPException(400, f"unknown stage {req.stage!r}")


def _chat_selection(session: dict, message: str) -> dict:
    parsed = chat_json(prompt_parse_selection(message))
    available = set(session["sites_found"].keys())

    # Defensive recovery: LLMs sometimes collapse "arkadia 2" → "arkadia" or
    # forget it entirely. If the user's raw message clearly names a site, force
    # it into the selection.
    msg_lower = message.lower()
    forced = []
    if "arkadia 2" in msg_lower or "arkadia2" in msg_lower:
        forced.append("arkadia2")
    if re.search(r"\barkadia\b(?!\s*2)", msg_lower):
        forced.append("arkadia")
    if "magnisia" in msg_lower:
        forced.append("magnisia")
    if "veroia" in msg_lower:
        forced.append("veroia")

    selected = list(parsed.get("sites_selected") or [])
    for k in forced:
        if k in available and k not in selected:
            selected.append(k)
    parsed["sites_selected"] = [s for s in selected if s in available]

    # Backfill prices: if the user said "all are 1M each" the LLM sometimes
    # only puts the price on the first site. If exactly ONE price was given
    # and the user clearly wanted it to apply to everyone, propagate it.
    prices = dict(parsed.get("prices") or {})
    if len(prices) == 1 and len(parsed["sites_selected"]) > 1:
        sole = next(iter(prices.values()))
        if sole and re.search(r"\b(each|all|every|same)\b", msg_lower):
            for s in parsed["sites_selected"]:
                prices.setdefault(s, sole)

    # Coerce price values to numbers and drop None/0 so downstream sees clean data.
    def _num(v):
        if isinstance(v, (int, float)):
            return float(v) if v else None
        if isinstance(v, str):
            # tolerate "1.5M", "1,500,000", "€1000000"
            s = v.lower().replace("€", "").replace(",", "").strip()
            mult = 1.0
            if s.endswith("m"):
                mult, s = 1_000_000.0, s[:-1].strip()
            elif s.endswith("k"):
                mult, s = 1_000.0, s[:-1].strip()
            try:
                n = float(s) * mult
                return n if n else None
            except ValueError:
                return None
        return None

    prices = {k: _num(v) for k, v in prices.items()}
    prices = {k: v for k, v in prices.items() if v}

    # General backfill: any selected site STILL missing a price gets the most
    # common provided price. This keeps the UI legend complete even when the
    # user gave mixed prices (e.g. "Arkadia 1M, Veroia 1.5M") and the model
    # only captured some of them.
    if prices:
        vals = list(prices.values())
        modal = max(set(vals), key=vals.count)
        for s in parsed["sites_selected"]:
            prices.setdefault(s, modal)
    parsed["prices"] = prices

    print(f"[chat] selection parsed: sites={parsed['sites_selected']} "
          f"prices={parsed['prices']} "
          f"dates={parsed.get('preferred_dates', {})}")

    if len(parsed["sites_selected"]) < 2:
        return {
            "reply": ("I need at least two sites to compare. Could you tell me "
                      "which ones you'd like to look at, and what each costs?"),
            "parsed_action": None,
            "next_stage": "selection",
        }
    reply = chat_text(prompt_selection_confirm(json.dumps(parsed, indent=2)))
    session["selection"] = parsed
    return {
        "reply": reply,
        "parsed_action": {
            "type": "selection_confirmed",
            "sites": parsed["sites_selected"],
            "prices": parsed.get("prices", {}),
            "preferred_dates": parsed.get("preferred_dates", {}),
        },
        "next_stage": "usecase",
    }


def _chat_usecase(session: dict, message: str) -> dict:
    parsed = chat_json(prompt_parse_usecase(message))
    reply = chat_text(prompt_usecase_confirm(
        use_case=parsed.get("use_case", ""),
        reasoning=parsed.get("reasoning", ""),
    ))
    session["use_case_config"] = parsed
    return {
        "reply": reply,
        "parsed_action": {
            "type": "usecase_confirmed",
            "use_case": parsed["use_case"],
            "weights": parsed["weights"],
            "anomaly_sign": parsed.get("anomaly_sign", -1),
            "date_discounts": parsed.get("date_discounts", {}),
        },
        "next_stage": "run",
    }


# ─── /api/run ─────────────────────────────────────────────────────────────

@app.post("/api/run")
def run(req: RunIn):
    """Stage uploaded sites, then start the pipeline (full run or fast re-score)."""
    session = _session(req.session_id)
    # Fast mode runs no scripts, so there is nothing to stage.
    if not req.fast:
        try:
            stage_uploaded_sites(req.session_id, req.sites, UPLOADS_DIR)
        except Exception as e:
            print(f"[app] stage_uploaded_sites failed (non-fatal): {e}")

    payload = req.model_dump()
    job_id = start_run(req.session_id, payload)
    session["job_id"] = job_id
    return {"job_id": job_id, "status": "started"}


# ─── /api/cache-info ──────────────────────────────────────────────────────

@app.get("/api/cache-info")
def get_cache_info():
    """Tell the frontend whether 'fast mode' can be offered, and for which cohort."""
    return cache_info()


# ─── /api/status ──────────────────────────────────────────────────────────

@app.get("/api/status/{job_id}")
def status(job_id: str):
    job = get_status(job_id)
    if not job:
        raise HTTPException(404, "unknown job_id")
    return {
        "job_id": job_id,
        "status": job["status"],
        "stage_label": job["stage_label"],
        "error": job.get("error"),
    }


# ─── /api/results ─────────────────────────────────────────────────────────

@app.get("/api/results/{job_id}")
def results(job_id: str):
    job = get_status(job_id)
    if not job:
        raise HTTPException(404, "unknown job_id")
    if job["status"] == "error":
        raise HTTPException(500, job.get("error") or "pipeline error")
    if job["status"] != "complete":
        raise HTTPException(409, f"job not complete (status: {job['status']})")

    raw = get_results(job_id) or {}
    return _enrich_results(job, raw)


def _enrich_results(job: dict, raw: dict) -> dict:
    """Add plot URLs (relative to /static), AI narrative, and a flat per_site dict."""
    payload = job.get("payload", {})

    anomaly = (raw.get("steps", {}) or {}).get("anomaly", {}) or {}
    ranking_keys = anomaly.get("ranking", payload.get("sites", []))
    per_site_raw = anomaly.get("per_site", {})
    sensitivity = anomaly.get("sensitivity", {})

    # Flatten per_site into the shape the frontend expects
    per_site: dict[str, dict] = {}
    for key in ranking_keys:
        site = per_site_raw.get(key, {}) or {}
        per_site[key] = {
            "final_score":  site.get("final_score", 0),
            "price_eur":    payload.get("prices", {}).get(key),
            "components":   site.get("components", {}),
            "anomaly":      site.get("anomaly", {}),
            "raw_indices":  site.get("raw_indices", {}),
        }

    # Plot URLs
    def to_urls(stage_block):
        urls = []
        for p in (stage_block or {}).get("plots", []) or []:
            path = p.get("path") or ""
            urls.append({
                "id": p.get("id"),
                "title": p.get("title"),
                "url": f"/static/{path}" if path else None,
            })
        return urls

    steps = raw.get("steps", {}) or {}
    plots = {
        "explore":    to_urls(steps.get("explore")),
        "indices":    to_urls(steps.get("indices")),
        "anomaly":    to_urls(steps.get("anomaly")),
        "evaluation": to_urls(steps.get("evaluate")),
    }

    # Build narrative
    use_case = payload.get("use_case", raw.get("use_case", ""))
    ranking_display = "  ".join(
        f"{i+1}. {DISPLAY_NAMES.get(k, k)}" for i, k in enumerate(ranking_keys)
    )
    # Backfill: if any selected site has no price, copy the most common one
    raw_prices = payload.get("prices", {}) or {}
    nonzero = [p for p in raw_prices.values() if p]
    fallback_price = max(set(nonzero), key=nonzero.count) if nonzero else 0
    def _price_for(key):
        p = raw_prices.get(key)
        return p if p else fallback_price

    summary_lines = []
    for k in ranking_keys:
        s = per_site.get(k, {})
        comp = s.get("components", {}) or {}
        anom = s.get("anomaly", {}) or {}
        # Also patch the per_site dict so the frontend sees the corrected price
        if not s.get("price_eur"):
            s["price_eur"] = _price_for(k)
            per_site[k] = s
        summary_lines.append(
            f"{DISPLAY_NAMES.get(k, k)}: score={s.get('final_score', 0):.3f}, "
            f"soil={comp.get('soil_quality', 0):.2f}, veg={comp.get('veg_quality', 0):.2f}, "
            f"consistency={comp.get('spatial_consist', 0):.2f}, "
            f"anomaly burden={anom.get('burden_raw', 0)*100:.1f}%, "
            f"price=€{int(s.get('price_eur') or 0):,}"
        )
    per_site_summary = "\n".join(summary_lines)

    winner = ranking_keys[0] if ranking_keys else "—"
    winner_display = DISPLAY_NAMES.get(winner, winner)
    sens_pct = (sensitivity.get(winner, {}) or {}).get("rank1_pct", 0)

    # Build the "active factors" description for the narrative prompt so the
    # LLM can't invent justifications based on unweighted axes.
    factor_labels = {
        "W_SOIL":     "soil quality (low clay / low bare-soil index / low salinity)",
        "W_CLAY":     "clay richness (favoured for excavation)",
        "W_MINERAL":  "mineral interest (iron oxides + carbonates)",
        "W_CONSIST":  "spatial uniformity of the terrain",
        "W_VEG":      "vegetation vigour (seasonally adjusted)",
        "W_MOISTURE": "canopy and soil moisture signal",
        "W_ANOMALY":  "subsurface spectral anomaly burden",
    }
    weights_used = payload.get("weights", {}) or {}
    active = [(k, v) for k, v in weights_used.items() if v >= 0.10]
    active.sort(key=lambda kv: -kv[1])
    active_factors_text = ", ".join(
        f"{factor_labels.get(k, k)} (weight {v:.2f})" for k, v in active
    ) or "no factor was dominant"
    anomaly_dir = ("anomalies treated as POSITIVE signals (indicates mineral interest)"
                   if payload.get("anomaly_sign", -1) > 0
                   else "anomalies treated as NEGATIVE signals (contamination risk)")

    narrative = chat_text(prompt_narrate(
        use_case=use_case,
        ranking_display=ranking_display,
        per_site_summary=per_site_summary,
        winner=winner_display,
        sensitivity_pct=sens_pct,
        active_factors=active_factors_text,
        anomaly_direction=anomaly_dir,
    ))

    # Surface training/evaluation insights so the frontend can show them.
    # Primary source: data/plots/03_anomaly/training_progress.json
    #   (full per-epoch arrays — written by 03_anomaly_ml.py during training)
    # Secondary: steps.evaluate.ae_metrics (written by 04_evaluate_model.py)
    training_progress = {}
    progress_path = DATA_DIR / "plots" / "03_anomaly" / "training_progress.json"
    if progress_path.exists():
        try:
            with open(progress_path, encoding="utf-8") as f:
                training_progress = json.load(f)
        except Exception as e:
            print(f"[app] couldn't read training_progress.json: {e}")

    evaluate_block = (raw.get("steps", {}) or {}).get("evaluate", {}) or {}
    ae_metrics = evaluate_block.get("ae_metrics", {}) or {}
    synthetic = evaluate_block.get("synthetic_anomaly", {}) or {}

    # If evaluate block didn't run / isn't there, derive scalars from training_progress
    train_hist = training_progress.get("train_history") or []
    val_hist   = training_progress.get("val_history") or []
    final_train = ae_metrics.get("final_train_loss") if ae_metrics else (train_hist[-1] if train_hist else None)
    final_val   = ae_metrics.get("final_val_loss")   if ae_metrics else (val_hist[-1]   if val_hist   else None)
    vt_ratio    = ae_metrics.get("val_train_ratio")
    if vt_ratio is None and final_train and final_val:
        vt_ratio = round(final_val / final_train, 3)

    # Architecture is fixed in 03_anomaly_ml.py: N_BANDS → 96 → 48 → 24 → 48 → 96 → N_BANDS
    anomaly_block_raw = raw.get("steps", {}).get("anomaly", {}) or {}
    anomaly_cfg = anomaly_block_raw.get("anomaly_config", {}) or {}

    return {
        "job_id": job["job_id"],
        "use_case": use_case,
        "ranking": ranking_keys,
        "display_names": DISPLAY_NAMES,
        "weights_used": payload.get("weights", {}),
        "anomaly_sign": payload.get("anomaly_sign", -1),
        "per_site": per_site,
        "plots": plots,
        "ai_narrative": narrative,
        "sensitivity": sensitivity,
        "training_insights": {
            "final_train_loss": final_train,
            "final_val_loss":   final_val,
            "val_train_ratio":  vt_ratio,
            "synthetic_anomaly": synthetic,
            "train_history":    train_hist,
            "val_history":      val_hist,
            "total_epochs":     training_progress.get("total_epochs", len(train_hist) or None),
            "architecture":     [None, 96, 48, 24, 48, 96, None],  # N_BANDS at ends
            "rx_threshold":     anomaly_cfg.get("rx_threshold"),
            "ae_threshold":     anomaly_cfg.get("ae_threshold"),
        },
    }


# ─── Session history ──────────────────────────────────────────────────────
# Past sessions (chat transcript + analysis) are archived under
# DATA_DIR/archive/<session_id>/ so they survive future runs. Plot files are
# snapshotted there too, since every pipeline run overwrites data/plots/.

ARCHIVE_DIR = DATA_DIR / "archive"


def _rewrite_plot_urls(results: dict, session_id: str) -> dict:
    """Point a saved session's plot URLs at its own snapshot folder."""
    for section in (results.get("plots") or {}).values():
        for p in section or []:
            u = p.get("url")
            if u and u.startswith("/static/plots/"):
                p["url"] = u.replace(
                    "/static/plots/", f"/static/archive/{session_id}/plots/", 1)
    return results


@app.post("/api/session-save")
def session_save(req: SessionSaveIn):
    """Archive a finished session: chat transcript + results + a plot snapshot."""
    archive = ARCHIVE_DIR / req.session_id
    archive.mkdir(parents=True, exist_ok=True)

    results = req.results
    # Snapshot the plots so the gallery stays faithful after later runs.
    plots_src = DATA_DIR / "plots"
    if plots_src.exists():
        plots_dst = archive / "plots"
        try:
            if plots_dst.exists():
                shutil.rmtree(plots_dst)
            shutil.copytree(plots_src, plots_dst)
            results = _rewrite_plot_urls(results, req.session_id)
        except Exception as e:
            print(f"[app] plot snapshot failed (non-fatal): {e}")

    record = {
        "session_id": req.session_id,
        "saved_at": time.time(),
        "messages": req.messages,
        "results": results,
    }
    with open(archive / "session.json", "w", encoding="utf-8") as f:
        json.dump(record, f)
    return {"ok": True, "session_id": req.session_id}


@app.get("/api/sessions")
def list_sessions():
    """List archived sessions, newest first."""
    out: list[dict[str, Any]] = []
    if ARCHIVE_DIR.exists():
        for d in ARCHIVE_DIR.iterdir():
            sj = d / "session.json"
            if not sj.exists():
                continue
            try:
                with open(sj, encoding="utf-8") as f:
                    rec = json.load(f)
                res = rec.get("results", {}) or {}
                ranking = res.get("ranking", []) or []
                out.append({
                    "session_id":    rec.get("session_id"),
                    "saved_at":      rec.get("saved_at"),
                    "use_case":      res.get("use_case"),
                    "winner":        ranking[0] if ranking else None,
                    "ranking":       ranking,
                    "message_count": len(rec.get("messages", []) or []),
                })
            except Exception as e:
                print(f"[app] skipping unreadable session {d.name}: {e}")
    out.sort(key=lambda r: r.get("saved_at") or 0, reverse=True)
    return {"sessions": out}


@app.get("/api/session/{session_id}")
def get_session(session_id: str):
    """Return one archived session (chat transcript + results)."""
    sj = ARCHIVE_DIR / session_id / "session.json"
    if not sj.exists():
        raise HTTPException(404, f"no archived session {session_id!r}")
    with open(sj, encoding="utf-8") as f:
        return json.load(f)
