# HELIO — Technical Overview
### Real Estate Beyond RGB · Hyperspectral Land Intelligence
*Makeathon 2026 · ISENSE / ICCS*

---

## 1. The concept

A photograph tells you what a parcel of land **looks like**. HELIO tells you what it
**is**.

We ingest EnMAP hyperspectral satellite imagery — 190 analysis bands spanning
**418–2445 nm** (visible, near-infrared and short-wave infrared) — and turn it into
a defensible land-investment decision. A buyer describes, in plain language, what
they intend to do with the land; HELIO analyses the spectral fingerprint of every
candidate parcel and returns a **ranked recommendation backed by physical
evidence**: soil chemistry, clay mineralogy, moisture, vegetation vigour, and
hidden subsurface anomalies that are invisible to ordinary aerial photography.

The product is three layers that hand off cleanly:

```
 Natural language ──▶  Conversational layer (LLM)  ──▶  weight vector W, anomaly sign, date discounts
                                                            │
 EnMAP scenes  ──────▶  Analysis pipeline (4 scripts)  ◀─────┘
                                                            │
                        Ranked recommendation + plots  ◀────┘
```

---

## 2. The conversational layer — a translator, not a wrapper

This is the part most likely to be mistaken for "just a chatbot". It is not. The
language model (GitHub Models, `gpt-4o-mini`) is given **exactly two cognitive
jobs**, both genuine NLP problems, and is deliberately walled off from everything
quantitative.

### 2a. Context extraction
From a free-text message ("compare Arkadia and Veroia, both at a million, prefer
the October scene") the model produces structured JSON: which sites, the asking
price per site, and any preferred acquisition date. It runs at `temperature = 0`
with JSON-mode pinning. The backend then hardens that output — forced site
detection from the raw text, numeric price coercion (`"1.5M"` → `1500000`), and
modal-price backfill — so a sloppy parse can never corrupt the analysis.

### 2b. Objective synthesis — the important one
The user says *"I want to plant olive trees and run a small agritourism site."*
There is no "olive trees" button. The model's job is to **map an open-ended human
intention onto a 7-dimensional optimisation objective**:

| Weight        | Meaning                                              |
|---------------|------------------------------------------------------|
| `W_SOIL`      | soil quality — low clay, low bare-soil, low salinity |
| `W_CLAY`      | clay richness (only for excavation use cases)        |
| `W_MINERAL`   | iron-oxide + carbonate mineral interest              |
| `W_CONSIST`   | spatial uniformity of the terrain                    |
| `W_VEG`       | vegetation / biomass vigour                          |
| `W_MOISTURE`  | canopy & soil moisture                               |
| `W_ANOMALY`   | weight on the subsurface anomaly burden              |

The model also returns an **`anomaly_sign`** (`-1` = anomalies are a contamination
risk; `+1` = anomalies are *desirable*, e.g. for mineral prospecting) and
**`date_discounts`** to neutralise seasonal bias.

The crucial design rule: **the LLM sets the objective; it never computes the
answer.** It never sees a pixel, never produces a score, never invents a number.
It outputs `W`; the deterministic engine computes `S(W)`. The final written
recommendation is likewise constrained — the narration prompt is fed only the
*active* factors (weights ≥ 0.10), so the model physically cannot justify a pick
using a metric that was not actually scored. This is what separates HELIO from a
GPT wrapper: the language model is a **semantic compiler** from intent to a
numeric objective function, and nothing more.

---

## 3. Data & preprocessing

Implemented in `spectral_common.py`.

- **Source.** EnMAP L2A surface-reflectance mosaics. Raw values are reflectance
  × 10; `NODATA = -9999` pixels become `NaN`.
- **Band masking** (`good_band_mask`). Three spectral regions are dropped before
  any analysis because they carry no reliable signal:
  - **850–1000 nm** — VNIR/SWIR detector-stitch artefacts
  - **1340–1460 nm** — atmospheric water-vapour absorption
  - **1790–1960 nm** — atmospheric water-vapour absorption
- **Quality masking.** EnMAP's own cloud, cloud-shadow and haze quality layers are
  combined; a pixel is additionally rejected unless **≥ 70 %** of its good bands
  are finite.
- **Spectral smoothing** (`smooth_cube`). A **Savitzky–Golay filter**
  (window = 9, polynomial order = 2) is applied along the *spectral* axis of every
  pixel. This suppresses sensor noise while preserving absorption-feature shape —
  important because every downstream index reads specific narrow bands.

Each site also has **extra temporal acquisitions** (different dates). These are
never scored — they are pooled only as **training augmentation** for the
autoencoder, widening its notion of "normal" Greek agricultural spectra.

---

## 4. Spectral indices (Script 02)

Nine physically-grounded band-ratio indices convert 190 raw bands into
interpretable quantities. Each is a 2-D map; per-site we keep **mean**, **std**,
and **coefficient of variation** (`CV = std / |mean|`).

| Index           | Bands (nm)             | Reads                                   |
|-----------------|------------------------|-----------------------------------------|
| NDVI            | 860 / 660              | vegetation vigour                       |
| NDRE            | 780 / 720              | chlorophyll / canopy stress             |
| NDMI            | 860 / 1640             | canopy & soil moisture                  |
| EVI             | 860 / 660 / 480        | biomass (saturation-resistant)          |
| BSI             | 1600 / 660 / 830 / 480 | bare-soil exposure                      |
| Clay Index      | 2100 / 2200            | clay-mineral absorption (SWIR)          |
| Iron Oxide      | 700 / 500              | hematite / goethite content             |
| Carbonate Index | 2340 / 2300            | limestone / carbonate content           |
| Salinity Index  | 1600 / 820             | salt-affected soils                     |

The **coefficient of variation** is used as a *spatial-uniformity* metric: a low
CV means the mineral signal is evenly distributed — predictable terrain, valued by
solar farms, logistics and large-scale agriculture.

---

## 5. Anomaly detection — a two-detector consensus (Script 03)

HELIO flags pixels whose spectra do not belong, using **two independent detectors
with different failure modes**, then trusting only where they agree.

### 5a. RX detector (classical)
The **Reed–Xiaoli** detector. The pooled pixel population is reduced with **PCA
(30 components, ~retaining most variance)**; each pixel's **Mahalanobis distance**
from the population mean is its anomaly score:

```
d²(x) = (x − μ)ᵀ Σ⁻¹ (x − μ)
```

The covariance is inverted with a **pseudo-inverse** (`np.linalg.pinv`) for
numerical stability. Fast, statistically principled, no training.

### 5b. Spectral autoencoder (machine learning)
A pixel-wise autoencoder, `SpectralAE`:

```
N_BANDS → 96 → 48 → 24 → 48 → 96 → N_BANDS
          └── encoder ──┘ │ └── decoder ──┘
                    bottleneck (24-D)
```

- **Layers:** `Linear → BatchNorm → LeakyReLU(0.1)`, with **Dropout 0.15 in the
  encoder only** (the decoder must reconstruct faithfully).
- **No skip connections** — a skip would let information bypass the 24-D
  bottleneck and defeat the whole anomaly mechanism.
- **Why a bottleneck?** Compressing 190 bands → 24 dimensions forces the network
  to learn the *manifold of normal* Greek agricultural spectra. A pixel that
  cannot be reconstructed from 24 numbers is, by definition, spectrally unusual.
- **Training.** Pooled pixels from challenge + extra temporal scenes (~3500+),
  standardised; Adam (`lr = 1e-3`, `weight_decay = 1e-4`), **cosine-annealing**
  LR schedule, 150 epochs, batch 64, **MSE** loss. An **80/20 split stratified by
  site** (seed 42) gives an honest val curve and a val/train overfit ratio.
- **Anomaly score** = per-pixel reconstruction MSE.

### 5c. Consensus
Both detectors threshold at the **90th percentile** of their pooled scores. The
**anomaly burden** of a site is the fraction of pixels flagged by **BOTH** RX
**and** the autoencoder (`RX ∩ AE`). Requiring agreement suppresses each
detector's individual false positives and yields high-confidence anomalies — the
agreement map visualises all four cases (normal / RX-only / AE-only / both).

---

## 6. Scoring & ranking (Script 03)

For every site, seven **component axes** are derived from the indices and the
anomaly burden, then **min–max normalised across the cohort** (relative scoring —
the question is "which of *these* parcels is best", not an absolute grade):

| Axis           | Definition                                                        |
|----------------|-------------------------------------------------------------------|
| `soil_quality` | `0.40·(1−Clay) + 0.35·(1−BSI) + 0.25·(1−Salinity)`                |
| `clay_quality` | normalised Clay Index (high = good, excavation only)              |
| `mineral_q`    | `0.50·IronOxide + 0.50·Carbonate`                                 |
| `consistency`  | `1 − CV` of Clay & BSI (terrain uniformity)                       |
| `veg_quality`  | NDVI × date discount                                              |
| `moisture`     | NDMI                                                              |
| `anomaly`      | `RX ∩ AE` burden                                                  |

The **final investment score**:

```
S = W_SOIL·soil + W_CLAY·clay + W_MINERAL·mineral
  + W_CONSIST·consistency + W_VEG·veg + W_MOISTURE·moisture
  + anomaly_sign · W_ANOMALY · anomaly
```

Sites are ranked by `S`. The weights come straight from the conversational
layer — so the *same* spectral evidence produces a different ranking for an
olive-grove buyer than for a solar-farm developer, which is exactly the point.

### Date robustness
A May acquisition looks artificially lush because it is peak growing season.
`date_discounts` (≈0.70 for spring scenes, 1.0 otherwise) deflate vegetation-
derived axes so two parcels imaged in different seasons can still be compared
fairly.

### Sensitivity / stability analysis
Ranking from one weight vector is fragile. HELIO sweeps the weights across a grid
(0→1 in 0.1 steps, constrained to sum ≤ 1) — **hundreds of weight combinations** —
and records how often each site lands at rank 1. A site that is #1 in 85 % of
combinations is a robust recommendation; one that wins only at a knife-edge
weighting is flagged as marginal. This is a small Monte-Carlo-style robustness
check that turns a single number into a confidence statement.

---

## 7. Niche techniques & algorithms — quick reference

For the judges, the non-obvious methods worth calling out:

- **Savitzky–Golay spectral smoothing** — denoise without blurring absorption
  features.
- **Atmospheric / detector band masking** — domain knowledge baked into
  preprocessing.
- **Reed–Xiaoli (RX) anomaly detection** — textbook hyperspectral target
  detection, Mahalanobis distance in PCA space.
- **Pseudo-inverse covariance** — numerically stable RX on a near-singular
  covariance.
- **Autoencoder manifold learning** — reconstruction error as an unsupervised
  anomaly score; bottleneck deliberately starves the network.
- **Two-detector consensus (`RX ∩ AE`)** — ensemble agreement to kill false
  positives.
- **Cohort-relative min–max normalisation** — comparative, not absolute, scoring.
- **Coefficient of variation as a uniformity metric.**
- **Temporal augmentation** — extra dated scenes expand "normal" without
  polluting the scored set.
- **Stratified 80/20 split** — honest validation on a small dataset.
- **Cosine-annealing LR schedule.**
- **Phenological date discounting** — season-fair cross-site comparison.
- **Weight-grid sensitivity sweep** — ranking-stability confidence.
- **LLM as a semantic compiler** — natural language → numeric objective vector,
  with the model fenced out of all quantitative work.

---

## 8. Web application

- **Backend** — FastAPI. Endpoints: `/api/upload` (unzip + scan scenes),
  `/api/chat` (the two LLM stages), `/api/run` (spawns the 4-script pipeline as a
  background job), `/api/status`, `/api/results`. Plots are served from `/static`.
- **Pipeline orchestration** — `pipeline.py` runs scripts 01→04 as subprocesses,
  tracks stage progress, and reads back `pipeline_results.json`.
- **Frontend** — a single-file React app: an animated intro, a guided
  conversational flow, an animated results page (recommendation, cohort
  scorecard, component radar, weight breakdown, autoencoder training curves) and
  a guided "Demonstrate Pipeline" walkthrough of every figure.

---

## Appendix — Presentation diagram prompt

A prompt for an image / diagram generation model, designed to produce an accurate
architecture figure for the slide deck:

> A clean, modern technical architecture diagram for a hyperspectral
> satellite land-analysis system called "HELIO", dark background (#1a1919),
> warm amber-and-orange accent palette, thin connector lines, minimal flat
> infographic style, monospace labels. Left-to-right flow in three labelled
> stages.
> **Stage 1 — "Conversational Layer":** a chat-bubble icon feeding a language-
> model block; two arrows out of it labelled "context: sites + prices" and
> "objective: 7-D weight vector W". Caption: "LLM translates intent into a
> numeric objective — it never computes scores".
> **Stage 2 — "Hyperspectral Pipeline":** a stacked datacube icon labelled
> "EnMAP · 190 bands · 418–2445 nm" flowing through four boxes in sequence:
> "Preprocess (band masking, Savitzky-Golay smoothing, quality masks)" →
> "Spectral Indices (NDVI, NDMI, Clay, Iron, Carbonate, BSI…)" → "Anomaly
> Detection" → "Scoring". Inside the Anomaly box show two parallel detectors,
> "RX — Mahalanobis in PCA space" and "Autoencoder 190→96→48→24→48→96→190",
> merging into a box labelled "RX ∩ AE consensus".
> **Stage 3 — "Decision":** a weighted-sum formula
> "S = Σ Wᵢ·axisᵢ" leading to a podium / ranked bar chart of four land
> parcels, plus a small "sensitivity sweep" gauge.
> Polished, presentation-quality, no photographic realism, no clutter,
> high contrast, legible at slide size.
