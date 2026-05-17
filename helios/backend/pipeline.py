"""
Pipeline orchestration.

Runs the 4 data-science scripts as subprocesses, tracks progress in memory,
and exposes a simple status getter for the /api/status endpoint.

Designed to fail gracefully — if the analysis directory is missing, or any
script blows up, we record the error and the frontend shows a sensible
message. The web app keeps running.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

# ── Paths — overridable via env var for portability ───────────────────────

ANALYSIS_ROOT = Path(os.environ.get(
    "ANALYSIS_ROOT",
    str(Path(__file__).resolve().parent.parent.parent / "analysis"),
)).resolve()
SCRIPTS_DIR = ANALYSIS_ROOT / "scripts"
DATA_DIR = ANALYSIS_ROOT / "data"
ENMAP_DIR = DATA_DIR / "images_makeathlon" / "enmap"
RESULTS_PATH = DATA_DIR / "pipeline_results.json"

# ── In-memory job registry ────────────────────────────────────────────────

_JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()

STAGES = [
    ("exploring",  "Exploring spectral data"),
    ("indexing",   "Computing spectral indices"),
    ("training",   "Training anomaly model"),
    ("evaluating", "Evaluating model"),
]


# ── Public API ────────────────────────────────────────────────────────────

def start_run(session_id: str, payload: dict[str, Any]) -> str:
    """Kick off a pipeline run. Returns job_id. Non-blocking."""
    job_id = uuid.uuid4().hex[:6]
    with _LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "session_id": session_id,
            "status": "queued",
            "stage_label": "Queued",
            "started_at": time.time(),
            "payload": payload,
            "error": None,
        }
    thread = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
    thread.start()
    return job_id


def get_status(job_id: str) -> dict[str, Any] | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def get_results(job_id: str) -> dict[str, Any] | None:
    """Loads pipeline_results.json after the job is complete."""
    job = get_status(job_id)
    if not job or job["status"] != "complete":
        return None
    # Fast-mode jobs carry their re-scored result in-memory rather than on disk.
    override = job.get("result_override")
    if override is not None:
        return override
    if not RESULTS_PATH.exists():
        return None
    try:
        with open(RESULTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[pipeline] failed to load results: {e}")
        return None


def cache_info() -> dict[str, Any]:
    """Report whether a previous full run is cached and which cohort it covers.

    Used by the frontend to decide if 'fast mode' can be offered.
    """
    if not RESULTS_PATH.exists():
        return {"available": False, "sites": [], "use_case": None}
    try:
        with open(RESULTS_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        anomaly = (raw.get("steps", {}) or {}).get("anomaly", {}) or {}
        sites = anomaly.get("ranking") or list((anomaly.get("per_site") or {}).keys())
        return {
            "available": bool(sites),
            "sites": sites,
            "use_case": raw.get("use_case"),
        }
    except Exception as e:
        print(f"[pipeline] cache_info failed: {e}")
        return {"available": False, "sites": [], "use_case": None}


# ── Internal ──────────────────────────────────────────────────────────────

def _set(job_id: str, **fields: Any) -> None:
    with _LOCK:
        _JOBS[job_id].update(fields)


def _run_job(job_id: str) -> None:
    """Body of the worker thread. Wraps everything in try/except."""
    try:
        payload = _JOBS[job_id]["payload"]

        # Fast mode: skip the four analysis scripts entirely and re-score the
        # cached run against the freshly-chosen weights.
        if payload.get("fast"):
            _run_fast(job_id, payload)
            return

        if not SCRIPTS_DIR.exists():
            raise RuntimeError(
                f"analysis directory not found at {SCRIPTS_DIR}. "
                f"Set ANALYSIS_ROOT to point at the analysis folder."
            )

        # Script 01
        _set(job_id, status="exploring", stage_label="Exploring spectral data")
        _run_script("01_explore_data.py")

        # Script 02
        _set(job_id, status="indexing", stage_label="Computing spectral indices")
        _run_script("02_indices_maps.py")

        # Script 03 — the one with dynamic args
        _set(job_id, status="training", stage_label="Training anomaly model")
        _run_script("03_anomaly_ml.py", [
            "--weights",        json.dumps(payload.get("weights", {})),
            "--date_discounts", json.dumps(payload.get("date_discounts", {})),
            "--anomaly_sign",   str(payload.get("anomaly_sign", -1)),
            "--use_case",       payload.get("use_case", ""),
        ])

        # Script 04
        _set(job_id, status="evaluating", stage_label="Evaluating model")
        _run_script("04_evaluate_model.py")

        _set(job_id, status="complete", stage_label="Analysis complete",
             finished_at=time.time())

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[pipeline] job {job_id} failed:\n{tb}")
        _set(job_id, status="error", stage_label="Pipeline error",
             error=str(e), finished_at=time.time())


def _run_fast(job_id: str, payload: dict[str, Any]) -> None:
    """Fast mode — reuse the cached analysis, re-score with new weights.

    The pretrained autoencoder, spectral indices and anomaly maps from the last
    full run are kept as-is. Only the final weighted score and ranking are
    recomputed, which is exact because the per-site components are already
    normalised across the cohort. Instant, no subprocesses.
    """
    if not RESULTS_PATH.exists():
        raise RuntimeError(
            "fast mode needs a previous full analysis — none is cached yet. "
            "Run a full analysis first."
        )
    with open(RESULTS_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    anomaly = (raw.get("steps", {}) or {}).get("anomaly", {}) or {}
    per_site = anomaly.get("per_site", {}) or {}
    if not per_site:
        raise RuntimeError("cached analysis has no per-site data — run a full analysis.")

    # Walk the stage labels quickly so the progress UI still animates.
    for status, label in STAGES:
        _set(job_id, status=status, stage_label=label)
        time.sleep(0.45)

    w    = payload.get("weights", {}) or {}
    sign = payload.get("anomaly_sign", -1)

    def _score(site: dict) -> float:
        c = site.get("components", {}) or {}
        anom = (site.get("anomaly", {}) or {}).get("burden_norm", 0) or 0
        return (
            w.get("W_SOIL",     0) * c.get("soil_quality",     0)
            + w.get("W_CLAY",     0) * c.get("clay_quality",     0)
            + w.get("W_MINERAL",  0) * c.get("mineral_quality",  0)
            + w.get("W_CONSIST",  0) * c.get("spatial_consist",  0)
            + w.get("W_VEG",      0) * c.get("veg_quality",      0)
            + w.get("W_MOISTURE", 0) * c.get("moisture_quality", 0)
            + sign * w.get("W_ANOMALY", 0) * anom
        )

    for key, site in per_site.items():
        site["final_score"] = round(float(_score(site)), 4)

    ranking = sorted(per_site.keys(), key=lambda k: -per_site[k]["final_score"])
    anomaly["ranking"]  = ranking
    anomaly["per_site"] = per_site
    raw.setdefault("steps", {})["anomaly"] = anomaly
    raw["use_case"] = payload.get("use_case", raw.get("use_case", ""))

    _JOBS[job_id]["result_override"] = raw
    _set(job_id, status="complete", stage_label="Analysis complete (fast)",
         finished_at=time.time())
    print(f"[pipeline] job {job_id} completed in FAST mode — ranking: {ranking}")


def _run_script(name: str, extra_args: list[str] | None = None) -> None:
    """Run one analysis script. Raises on non-zero exit."""
    cmd = [sys.executable, name] + (extra_args or [])
    print(f"[pipeline] >>> {' '.join(cmd)}")
    proc = subprocess.run(
        cmd, cwd=str(SCRIPTS_DIR), capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{name} failed (exit {proc.returncode})\n"
            f"--- stdout ---\n{proc.stdout[-1000:]}\n"
            f"--- stderr ---\n{proc.stderr[-1000:]}"
        )
    print(f"[pipeline] <<< {name} OK")


# ── Helper used by /api/upload to put uploaded data in the right place ────

def stage_uploaded_sites(session_id: str, selected: list[str], uploads_dir: Path) -> None:
    """Copy selected site folders from uploads/ into the pipeline's expected location."""
    import shutil
    src_root = uploads_dir / f"session_{session_id}" / "enmap"
    if not src_root.exists():
        print(f"[pipeline] uploads root missing: {src_root}")
        return
    ENMAP_DIR.mkdir(parents=True, exist_ok=True)
    for key in selected:
        src = src_root / key
        if not src.exists():
            continue
        dst = ENMAP_DIR / key
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"[pipeline] staged {key}")
