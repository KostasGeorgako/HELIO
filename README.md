# HELIO — Beyond RGB

### Hyperspectral land-investment intelligence · Makeathon 2026

HELIO turns raw **EnMAP hyperspectral satellite imagery** into a defensible
land-investment decision. A buyer describes, in plain language, what they intend
to do with a parcel; HELIO analyses the spectral fingerprint of every candidate
site — soil chemistry, clay mineralogy, moisture, vegetation, and hidden
subsurface anomalies invisible to ordinary photography — and returns a **ranked
recommendation backed by physical evidence**.

> A photograph shows you what land *looks like*. 190 spectral bands from
> 418–2445 nm show you what it *is*.

A full technical write-up is in **[HELIO_TECHNICAL_OVERVIEW.md](HELIO_TECHNICAL_OVERVIEW.md)**
and an architecture diagram in **[HELIO_architecture.svg](HELIO_architecture.svg)**.

---

## Repository layout

```
contest/
├── analysis/                  ← data-science pipeline (Python)
│   ├── scripts/
│   │   ├── 01_explore_data.py    exploration & quality masks
│   │   ├── 02_indices_maps.py    spectral indices
│   │   ├── 03_anomaly_ml.py      RX + autoencoder anomaly + ranking
│   │   ├── 04_evaluate_model.py  evaluation
│   │   └── spectral_common.py    shared loader / model / index library
│   ├── data/
│   │   ├── images_makeathlon/enmap/   EnMAP scenes
│   │   ├── plots/                     generated figures
│   │   ├── spectral_ae.pt             trained autoencoder checkpoint
│   │   └── pipeline_results.json      latest run results
│   └── requirements.txt
├── helios/
│   ├── backend/               ← FastAPI service
│   │   ├── app.py                endpoints + session state
│   │   ├── pipeline.py           runs the 4 analysis scripts / fast re-score
│   │   ├── llm.py                GitHub Models client (+ mock fallback)
│   │   ├── prompts.py            HELIO prompt library
│   │   └── requirements.txt
│   └── frontend/
│       └── index.html            single-file React app
├── HELIO_TECHNICAL_OVERVIEW.md
└── HELIO_architecture.svg
```

---

## How it works

```
 Intro ─▶ Upload ZIP ─▶ Conversational chat ─▶ Full / Fast ─▶ Pipeline ─▶ Results
                         (sites · prices ·       choice                   ranking +
                          intended use)                                   AI narrative
                                                                          + plot gallery
```

1. **Upload** — drop a ZIP of EnMAP scenes (`enmap/<site>/<acquisition>/SPECTRAL_IMAGE.TIF`). 
    Important: for demonstration purposes, use ```images_makeathlon.zip```, insice ```analysis/data```.
2. **Chat** — HELIO asks which sites to compare and the asking price of each,
   then what you intend to use the land for. It interprets that intent into a
   7-axis scoring objective.
3. **Full or Fast** — choose a full analysis (recomputes everything) or fast
   mode (reuses the pretrained model + cached maps — see below).
4. **Pipeline** — exploration → spectral indices → RX + autoencoder anomaly
   detection → cohort scoring.
5. **Results** — a ranked recommendation, an AI narrative, score breakdown,
   autoencoder training curves, and a guided plot walkthrough.

### Full vs Fast analysis

| Mode | What it does | When available |
|------|--------------|----------------|
| **Full** | Re-runs all four analysis scripts from the raw cubes — re-explores, re-indexes, retrains the autoencoder, re-evaluates. ~60–90 s. | Always. |
| **Fast** | Skips the heavy compute: reuses the cached `pipeline_results.json` and pretrained `spectral_ae.pt`, and only re-scores the cohort against the new use case. Instant. | Only when the selected sites **exactly match** the cohort of the last full run (the cached scores are normalised across that whole cohort). |

---

## Running it

### 1 · Backend (Terminal 1)

```bash
cd helios/backend
python -m venv .venv && source .venv/bin/activate     # or use an existing venv
pip install -r requirements.txt

# GitHub Models token — https://github.com/settings/tokens
# (classic PAT, no scopes needed). Without it, deterministic mocks are used.
export GITHUB_TOKEN="ghp_..."

uvicorn app:app --host 0.0.0.0 --port 5050
```

Then open `http://localhost:5050/` — you should see service-status JSON.

> **Do not use `--reload`.** The reloader watches `helios/backend/`, and an
> upload extracting files there would restart the server mid-request. Run
> without it.
>
> **On WSL**, bind `--host 0.0.0.0` (as above) so the Windows browser can reach
> the backend. If `localhost:5050` still hangs, run `wsl --shutdown` from
> Windows and restart.

### 2 · Frontend (Terminal 2)

```bash
cd helios/frontend
python -m http.server 8080
```

Open **`http://localhost:8080`**.

### Running a Full analysis

A full run shells out to the `analysis/scripts/` scripts using the **backend's
Python**, so for Full mode the backend environment also needs the analysis
dependencies (`pip install -r ../../analysis/requirements.txt` — PyTorch,
rasterio, scikit-learn, …). Fast mode needs none of this — it only reads the
cached results.

---

## Environment variables

| Var | Default | Purpose |
|-----|---------|---------|
| `GITHUB_TOKEN` | *(none → mocks)* | GitHub PAT for GitHub Models. |
| `GITHUB_MODEL` | `gpt-4o-mini` | Any GitHub Models catalogue name. |
| `GITHUB_MODELS_ENDPOINT` | `https://models.inference.ai.azure.com` | Rarely changed. |
| `ANALYSIS_ROOT` | `../../analysis` (auto-resolved) | Path to the analysis folder. The default is correct for this layout — only set it if you move things. |
| `UPLOADS_DIR` | `./uploads` | Where uploaded ZIPs are unpacked. |

---

## API reference

| Method | Path | Returns |
|--------|------|---------|
| `GET`  | `/` | Service status |
| `POST` | `/api/upload` | `{session_id, sites_found, first_ai_message}` |
| `POST` | `/api/chat` | `{reply, parsed_action, next_stage}` |
| `POST` | `/api/run` | `{job_id, status}` |
| `GET`  | `/api/status/{job_id}` | `{status, stage_label, error}` |
| `GET`  | `/api/results/{job_id}` | Full results + AI narrative |
| `GET`  | `/api/cache-info` | Fast-mode availability |
| `GET`  | `/api/sessions`, `/api/session/{id}` | Archived sessions |
| `GET`  | `/static/...` | Pipeline plot images |

---

## The analysis pipeline (short version)

- **Preprocessing** — EnMAP L2A, 190 analysis bands (418–2445 nm); atmospheric
  and detector-noise bands masked; Savitzky–Golay spectral smoothing.
- **Spectral indices** — NDVI, NDRE, NDMI, EVI, BSI, Clay, Iron-oxide,
  Carbonate, Salinity.
- **Anomaly detection** — a two-detector consensus: the classical **RX**
  (Reed–Xiaoli, Mahalanobis distance in PCA space) and a **spectral
  autoencoder** (`N→96→48→24→48→96→N`); a pixel counts as anomalous only when
  *both* flag it.
- **Scoring & ranking** — seven cohort-normalised component axes combined with
  the weights inferred from the user's stated use case, plus a sensitivity
  sweep for ranking-stability confidence.

See **[HELIO_TECHNICAL_OVERVIEW.md](HELIO_TECHNICAL_OVERVIEW.md)** for the full
detail.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Upload hangs on "scanning structure" | You're running uvicorn with `--reload` — remove it. |
| `localhost:5050` hangs (WSL) | Bind `--host 0.0.0.0`; if still stuck, `wsl --shutdown` from Windows and restart. |
| `analysis directory not found` | `ANALYSIS_ROOT` is set to a wrong/relative path. Unset it (the default is correct) or set an **absolute** path. |
| Fast mode greyed out | The selected sites differ from the last full run's cohort — select the same set, or run a full analysis first. |
| GitHub Models 401 / 429 | Token invalid, or free-tier rate limit. Unset `GITHUB_TOKEN` to fall back to mocks and keep the demo moving. |
| Plots don't appear | `ANALYSIS_ROOT`/`DATA_DIR` not pointing at the real `analysis/data`. |

---

*HELIO · Beyond RGB · Makeathon 2026*
