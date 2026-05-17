"""
03_anomaly_ml.py — Anomaly Detection (RX + Autoencoder) + Final Ranking
========================================================================
RX DETECTOR (classical, fast)
  Mahalanobis distance of each pixel from pooled population mean in PCA space.
  Math: d²(x) = (x−μ)ᵀ Σ⁻¹ (x−μ)

PIXEL AUTOENCODER (ML)
  Architecture: N_BANDS → 96 → 48 → 24 → 48 → 96 → N_BANDS
  Training: pooled pixels from all scenes including temporal extras (~3500+ px)
  Val split: 80/20 stratified by site, same seed as evaluate_model.py
  Loss: MSE. High reconstruction error = spectrally anomalous pixel.

FINAL SCORING (date-robust axes only)
  S = W_SOIL × soil_quality + W_CONSIST × spatial_consistency − W_ANOMALY × anomaly_burden
  soil_quality     = f(Clay Index, BSI) — SWIR mineral bands, permanent
  spatial_consist  = 1 − CV(Clay, BSI) — uniformity of mineral distribution
  anomaly_burden   = fraction of pixels flagged by BOTH RX and AE
  NDVI/NDRE/NDMI   = contextual only, not scored

Outputs: plots to PLOTS_DIR, model to MODEL_PATH, manifest via pipeline_utils.
"""

# 03_anomaly_ml.py — top of file
import sys, os, json, warnings
warnings.filterwarnings("ignore")
from itertools import product as iproduct

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from spectral_common import (
    DATA_ROOT, MODEL_PATH, SCENES, SCENES_EXTRA, COLORS,
    RANDOM_SEED, VAL_FRAC,
    good_band_mask, load_scene, smooth_cube,
    extract_valid_spectra, scene_path, load_all_scenes,
    compute_all_indices, summarise_indices,
    SpectralAE, band_at,
)


PLOTS_DIR        = "../data/plots/03_anomaly"   # was wrongly set to 01_explore
PROGRESS_PATH    = os.path.join(PLOTS_DIR, "training_progress.json")
os.makedirs(PLOTS_DIR, exist_ok=True)

AE_EPOCHS        = 150
AE_LR            = 1e-3
AE_BATCH         = 64
PCA_DIMS         = 30
ANOMALY_PERCENTILE = 90

W_SOIL    = 0.30   # was 0.35 — slightly reduced to make room for veg
W_CONSIST = 0.25   # was 0.35 — reduced; consistency less dominant now
W_VEG     = 0.25   # was 0.00 — now scored
W_ANOMALY = 0.20   # was 0.30 — reduced; anomaly still penalized but less harshly

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PROGRESS_PATH = os.path.join(PLOTS_DIR, "training_progress.json")



import argparse, json

parser = argparse.ArgumentParser()
parser.add_argument("--weights",        type=str, default=None)
parser.add_argument("--date_discounts", type=str, default=None)
parser.add_argument("--anomaly_sign",   type=float, default=-1.0,
                    help="+1 = anomalies are good (excavation), -1 = bad (default)")
parser.add_argument("--use_case",       type=str, default="general")
args = parser.parse_args()

ANOMALY_SIGN = args.anomaly_sign  # +1 or -1

if args.weights:
    w = json.loads(args.weights)
    W_SOIL     = w.get("W_SOIL",    0.20)
    W_CLAY     = w.get("W_CLAY",    0.15)
    W_MINERAL  = w.get("W_MINERAL", 0.00)
    W_CONSIST  = w.get("W_CONSIST", 0.25)
    W_VEG      = w.get("W_VEG",     0.20)
    W_MOISTURE = w.get("W_MOISTURE",0.10)
    W_ANOMALY  = w.get("W_ANOMALY", 0.10)
else:
    W_SOIL=0.20; W_CLAY=0.15; W_MINERAL=0.00
    W_CONSIST=0.25; W_VEG=0.20; W_MOISTURE=0.10; W_ANOMALY=0.10

DATE_DISCOUNTS = json.loads(args.date_discounts) if args.date_discounts else {
    "arkadia": 1.0, "arkadia2": 0.70, "magnisia": 1.0, "veroia": 1.0
}


def train_ae(pixels_norm, n_bands, labels_for_stratify):
    """
    Train with 80/20 stratified val split.
    Returns: model, train_losses, val_losses, train_idx, val_idx
    """
    train_idx, val_idx = train_test_split(
        np.arange(len(pixels_norm)),
        test_size=VAL_FRAC,
        random_state=RANDOM_SEED,
        stratify=labels_for_stratify,
    )

    X_train = torch.tensor(pixels_norm[train_idx], dtype=torch.float32).to(DEVICE)
    X_val   = torch.tensor(pixels_norm[val_idx],   dtype=torch.float32).to(DEVICE)

    dl = DataLoader(TensorDataset(X_train), batch_size=AE_BATCH, shuffle=True, drop_last=False)

    model     = SpectralAE(n_bands).to(DEVICE)
    opt       = torch.optim.Adam(model.parameters(), lr=AE_LR, weight_decay=1e-4)
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=AE_EPOCHS)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(AE_EPOCHS):
        model.train()
        ep_loss = 0.0
        for (batch,) in dl:
            opt.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(batch)
        sched.step()
        train_losses.append(ep_loss / len(X_train))

        model.eval()
        with torch.no_grad():
            val_losses.append(criterion(model(X_val), X_val).item())

        if (epoch + 1) % 10 == 0:
            with open(PROGRESS_PATH, "w") as f:
                json.dump({
                    "epoch": epoch + 1, "total_epochs": AE_EPOCHS,
                    "train_loss": round(train_losses[-1], 6),
                    "val_loss":   round(val_losses[-1],   6),
                    "train_history": [round(l, 6) for l in train_losses],
                    "val_history":   [round(l, 6) for l in val_losses],
                    "done": False,
                }, f, indent=2)
            print(f"    Epoch {epoch+1:3d}/{AE_EPOCHS}  "
                  f"train={train_losses[-1]:.6f}  val={val_losses[-1]:.6f}")

    with open(PROGRESS_PATH, "w") as f:
        json.dump({
            "epoch": AE_EPOCHS, "total_epochs": AE_EPOCHS,
            "train_loss": round(train_losses[-1], 6),
            "val_loss":   round(val_losses[-1],   6),
            "train_history": [round(l, 6) for l in train_losses],
            "val_history":   [round(l, 6) for l in val_losses],
            "done": True,
        }, f, indent=2)

    return model, train_losses, val_losses, train_idx, val_idx


# ─── LOAD CHALLENGE SCENES ────────────────────────────────────────────────────

print("=" * 65)
print(f"Loading challenge scenes  |  device: {DEVICE}")
print("=" * 65)
all_data, wl_ref, good_ref, N_BANDS = load_all_scenes(smooth=True, verbose=True)

if not all_data:
    sys.exit("No challenge scenes loaded. Check DATA_ROOT.")

sites   = list(all_data.keys())


# ─── BUILD TRAINING POOL ─────────────────────────────────────────────────────
# Pool is a list until all sources are collected, then converted once to numpy.

print("\nBuilding training pool...")

pool_pixels_list = []
pool_labels_list = []

# Step 1: challenge scenes
for key, d in all_data.items():
    spectra, _ = extract_valid_spectra(d["cube"], d["wl"], d["valid"], d["good"])
    pool_pixels_list.append(spectra)
    pool_labels_list.extend([key] * len(spectra))
    print(f"  [challenge] {key}: {len(spectra)} px")

# Step 2: extra temporal images (training augmentation only)
print("\nLoading extra temporal images...")
for extra_key, (fname, site_key) in SCENES_EXTRA.items():
    fpath = scene_path(site_key, fname)
    if not os.path.isdir(fpath):
        print(f"  [{extra_key}] SKIPPED — not found")
        continue
    cube_e, wl_e, valid_e, _ = load_scene(fpath)
    good_e = good_band_mask(wl_e)
    cube_e = smooth_cube(cube_e, good_e)
    spectra_e, _ = extract_valid_spectra(cube_e, wl_e, valid_e, good_e)
    pool_pixels_list.append(spectra_e)
    pool_labels_list.extend([site_key] * len(spectra_e))
    print(f"  [{extra_key}] → {len(spectra_e)} px added (labelled as '{site_key}')")


# Single conversion to numpy
pool_pixels = np.vstack(pool_pixels_list).astype(np.float32)
pool_labels = np.array(pool_labels_list)
print(f"\n  Total training pool: {len(pool_pixels)} pixels × {N_BANDS} bands")

# Standardise
scaler    = StandardScaler()
pool_norm = scaler.fit_transform(pool_pixels)

# ─── RX DETECTOR ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("RX Anomaly Detector (Reed-Xiaoli Mahalanobis)")
print("=" * 65)

pca = PCA(n_components=PCA_DIMS, random_state=RANDOM_SEED)
pool_pca = pca.fit_transform(pool_norm)
print(f"  PCA variance retained ({PCA_DIMS} components): "
      f"{pca.explained_variance_ratio_.sum()*100:.1f}%")

mu_pca  = pool_pca.mean(axis=0)
cov_inv = np.linalg.pinv(np.cov(pool_pca.T))

def rx_score(spectra_norm):
    proj = pca.transform(spectra_norm)
    diff = proj - mu_pca
    return np.array([d @ cov_inv @ d for d in diff])

rx_pool_scores = rx_score(pool_norm)
rx_threshold   = np.percentile(rx_pool_scores, ANOMALY_PERCENTILE)
print(f"  RX threshold (p{ANOMALY_PERCENTILE}): {rx_threshold:.2f}")

# Compute RX maps for challenge scenes only
rx_maps = {}
for key, d in all_data.items():
    spectra, coords = extract_valid_spectra(d["cube"], d["wl"], d["valid"], d["good"])
    rx_map = np.full(d["cube"].shape[1:], np.nan)
    if len(spectra):
        scores = rx_score(scaler.transform(spectra))
        for (r, c), s in zip(coords, scores):
            rx_map[r, c] = s
    rx_maps[key] = rx_map
    print(f"  [{key}]  RX anomaly burden: {np.nanmean(rx_map > rx_threshold)*100:.1f}%")

# ─── AUTOENCODER TRAINING ────────────────────────────────────────────────────

print("\n" + "=" * 65)
print(f"Training Spectral Autoencoder  "
      f"({N_BANDS}→96→48→24→48→96→{N_BANDS})")
print(f"  Pool: {len(pool_pixels)} px  |  Epochs: {AE_EPOCHS}  "
      f"|  Batch: {AE_BATCH}  |  LR: {AE_LR}")
print("=" * 65)

model, train_losses, val_losses, train_idx, val_idx = train_ae(
    pool_norm, N_BANDS, pool_labels
)
model.eval()

ae_pool_errors = model.reconstruction_error(
    torch.tensor(pool_norm, dtype=torch.float32).to(DEVICE)
)
ae_threshold = np.percentile(ae_pool_errors, ANOMALY_PERCENTILE)
print(f"\n  AE threshold (p{ANOMALY_PERCENTILE}): {ae_threshold:.6f}")
print(f"  Final train loss : {train_losses[-1]:.6f}")
print(f"  Final val loss   : {val_losses[-1]:.6f}")
print(f"  Val/train ratio  : {val_losses[-1]/train_losses[-1]:.2f}")

# Compute AE maps for challenge scenes only
ae_maps = {}
for key, d in all_data.items():
    spectra, coords = extract_valid_spectra(d["cube"], d["wl"], d["valid"], d["good"])
    ae_map = np.full(d["cube"].shape[1:], np.nan)
    if len(spectra):
        norm_t = torch.tensor(scaler.transform(spectra), dtype=torch.float32).to(DEVICE)
        errs   = model.reconstruction_error(norm_t)
        for (r, c), e in zip(coords, errs):
            ae_map[r, c] = e
    ae_maps[key] = ae_map
    print(f"  [{key}]  AE anomaly burden: {np.nanmean(ae_map > ae_threshold)*100:.1f}%")

# ─── SAVE CHECKPOINT ─────────────────────────────────────────────────────────

torch.save({
    "model_state":   model.state_dict(),
    "n_bands":       N_BANDS,
    "bottleneck":    24,
    "scaler_mean":   scaler.mean_,
    "scaler_std":    scaler.scale_,
    "train_losses":  train_losses,
    "val_losses":    val_losses,
    "train_idx":     train_idx,
    "val_idx":       val_idx,
    "rx_threshold":  float(rx_threshold),
    "ae_threshold":  float(ae_threshold),
    "wl_good":       wl_ref[good_ref],
    "pool_size":     len(pool_pixels),
}, MODEL_PATH)
print(f"\n  Model saved → {MODEL_PATH}")

# ─── PLOTS ───────────────────────────────────────────────────────────────────

print("\nGenerating plots...")

# Training loss (train + val)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(train_losses, color='steelblue', lw=1.5, label='Train')
axes[0].plot(val_losses,   color='tomato',    lw=1.5, ls='--', label='Val')
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE Loss")
axes[0].set_title("AE Training vs Validation Loss")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

ratio = np.array(val_losses) / (np.array(train_losses) + 1e-10)
axes[1].plot(ratio, color='darkorange', lw=1.5)
axes[1].axhline(1.0, color='green',  lw=0.8, ls='--', label='Perfect (ratio=1)')
axes[1].axhline(1.5, color='orange', lw=0.8, ls='--', label='Mild overfit')
axes[1].axhline(2.5, color='red',    lw=0.8, ls='--', label='Severe overfit')
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val / Train ratio")
axes[1].set_title("Overfitting Ratio")
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, max(4.0, ratio.max() * 1.1))

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "ae_training_loss.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: ae_training_loss.png")

# Anomaly maps (RX and AE)
for detector, maps, threshold, title in [
    ("rx", rx_maps, rx_threshold, "RX Anomaly Score (Mahalanobis)"),
    ("ae", ae_maps, ae_threshold, "AE Reconstruction Error"),
]:
    fig, axes = plt.subplots(1, len(all_data), figsize=(4.5 * len(all_data), 4.5))
    if len(all_data) == 1: axes = [axes]

    cmap = plt.get_cmap("hot_r")
    cmap.set_bad(color="#888888")
    all_vals = np.concatenate([m[np.isfinite(m)] for m in maps.values()])
    vmin, vmax = 0, np.percentile(all_vals, 97)

    for ax, (key, m) in zip(axes, maps.items()):
        im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        if (m > threshold).any():
            ax.contour(m > threshold, levels=[0.5], colors='cyan', linewidths=1.5)
        ax.set_title(all_data[key]["label"], fontsize=9)
        pct = np.nanmean(m > threshold) * 100
        ax.text(0.03, 0.06, f"Anomaly: {pct:.0f}%", transform=ax.transAxes,
                fontsize=8, color='cyan',
                bbox=dict(boxstyle='round', fc='black', alpha=0.6))
        ax.axis('off')

    plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
    fig.suptitle(f"{title}\nCyan contour = above p{ANOMALY_PERCENTILE} threshold", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"anomaly_{detector}_maps.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: anomaly_{detector}_maps.png")

# Agreement map
fig, axes = plt.subplots(1, len(all_data), figsize=(4.5 * len(all_data), 4.5))
if len(all_data) == 1: axes = [axes]

for ax, (key, d) in zip(axes, all_data.items()):
    rx, ae = rx_maps[key], ae_maps[key]
    valid  = np.isfinite(rx) & np.isfinite(ae)
    agreement = np.full(rx.shape, np.nan)
    agreement[valid & ~(rx > rx_threshold) & ~(ae > ae_threshold)] = 0
    agreement[valid &  (rx > rx_threshold) & ~(ae > ae_threshold)] = 1
    agreement[valid & ~(rx > rx_threshold) &  (ae > ae_threshold)] = 2
    agreement[valid &  (rx > rx_threshold) &  (ae > ae_threshold)] = 3

    cmap_a = matplotlib.colors.ListedColormap(['#c8ebc8', '#f0a060', '#6090e0', '#cc2222'])
    ax.imshow(agreement, cmap=cmap_a, vmin=0, vmax=3, interpolation='nearest')
    ax.set_title(all_data[key]["label"], fontsize=9)
    both_pct = np.nanmean(agreement == 3) * 100
    ax.text(0.03, 0.06, f"Both: {both_pct:.0f}%", transform=ax.transAxes,
            fontsize=8, color='white',
            bbox=dict(boxstyle='round', fc='#cc2222', alpha=0.8))
    ax.axis('off')

legend_patches = [
    mpatches.Patch(color='#c8ebc8', label='Normal'),
    mpatches.Patch(color='#f0a060', label='RX only'),
    mpatches.Patch(color='#6090e0', label='AE only'),
    mpatches.Patch(color='#cc2222', label='Both (high confidence)'),
]
fig.legend(handles=legend_patches, loc='lower center', ncol=4,
           fontsize=8, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Anomaly Agreement: RX ∩ AE\nRed = both detectors agree", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "anomaly_agreement.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: anomaly_agreement.png")

# ─── SCORING ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("FINAL SCORING — date-robust axes only")
print("=" * 65)

details = {}

# In the scoring loop, replace the manual clay/bsi blocks:
for key, d in all_data.items():
    cube, wl, valid = d["cube"], d["wl"], d["valid"]

    idx   = compute_all_indices(cube, wl, valid)
    stats = summarise_indices(idx, valid)

    clay_mean  = stats["clay"]["mean"]
    clay_cv    = stats["clay"]["cv"]
    bsi_mean   = stats["bsi"]["mean"]
    bsi_cv     = stats["bsi"]["cv"]
    iron_mean  = stats["iron_oxide"]["mean"]
    carb_mean  = stats["carbonate"]["mean"]
    sal_mean   = stats["salinity"]["mean"]

    ndvi_adj   = stats["ndvi"]["mean"] * DATE_DISCOUNTS.get(key, 1.0)
    ndmi_mean  = stats["ndmi"]["mean"]
    consist_raw = (clay_cv + bsi_cv) / 2.0

    rx, ae = rx_maps[key], ae_maps[key]
    valid_both = np.isfinite(rx) & np.isfinite(ae)
    anomaly_burden = float(np.mean(
        (rx[valid_both] > rx_threshold) & (ae[valid_both] > ae_threshold)
    )) if valid_both.sum() > 0 else 0.0

    # anomaly burden already computed from RX/AE above
    details[key] = {
        "clay_mean":      clay_mean,
        "bsi_mean":       bsi_mean,
        "iron_mean":      iron_mean,
        "carbonate_mean": carb_mean,
        "salinity_mean":  sal_mean,
        "clay_cv":        clay_cv,
        "bsi_cv":         bsi_cv,
        "consist_raw":    consist_raw,
        "ndvi_raw":       stats["ndvi"]["mean"],
        "ndvi_adj":       ndvi_adj,
        "ndre_mean":      stats["ndre"]["mean"],   # ← needed for prints and manifest
        "ndmi_mean":      ndmi_mean,
        "evi_mean":       stats["evi"]["mean"],    # ← available, worth keeping
        "anomaly_burden": anomaly_burden,
    }

    print(f"  [{key}]  {d['label'].replace(chr(10), ' ')}")
    print(f"    Clay mean    : {clay_mean:.3f}   BSI mean: {bsi_mean:+.3f}")
    print(f"    CV(Clay)     : {clay_cv:.3f}   CV(BSI): {bsi_cv:.3f}")
    print(f"    Anomaly (RX∩AE): {anomaly_burden*100:.1f}%")
    print(f"    NDVI (adj): {details[key]['ndvi_adj']:.3f}  NDMI: {details[key]['ndmi_mean']:.3f}")
    print()

def norm01(vals):
    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-9: return [0.5] * len(vals)
    return [(v - mn) / (mx - mn) for v in vals]

def invert(normed): return [1 - v for v in normed]

# Normalise all axes
clay_n    = norm01([details[k]["clay_mean"]      for k in sites])
bsi_n     = norm01([details[k]["bsi_mean"]       for k in sites])
iron_n    = norm01([details[k]["iron_mean"]      for k in sites])
carb_n    = norm01([details[k]["carbonate_mean"] for k in sites])
sal_n     = norm01([details[k]["salinity_mean"]  for k in sites])
consist_n = norm01([details[k]["consist_raw"]    for k in sites])
veg_n     = norm01([details[k]["ndvi_adj"]       for k in sites])
moist_n   = norm01([details[k]["ndmi_mean"]      for k in sites])
anomaly_n = norm01([details[k]["anomaly_burden"] for k in sites])

scores = {}
for ki, key in enumerate(sites):
    # soil_quality: lower clay AND lower BSI AND lower salinity = better
    soil_q    = (invert(clay_n)[ki]*0.40 + invert(bsi_n)[ki]*0.35
                 + invert(sal_n)[ki]*0.25)
    # clay_quality: higher clay = better (for excavation use case)
    clay_q    = clay_n[ki]
    # mineral_quality: iron oxides + carbonates
    mineral_q = (iron_n[ki] * 0.50 + carb_n[ki] * 0.50)
    # consistency
    consist_q = invert(consist_n)[ki]
    # vegetation
    veg_q     = veg_n[ki]
    # moisture
    moist_q   = moist_n[ki]
    # anomaly — sign controlled externally
    anom_q    = anomaly_n[ki]

    S = (W_SOIL     * soil_q
       + W_CLAY     * clay_q
       + W_MINERAL  * mineral_q
       + W_CONSIST  * consist_q
       + W_VEG      * veg_q
       + W_MOISTURE * moist_q
       + ANOMALY_SIGN * W_ANOMALY * anom_q)

    scores[key] = S
    details[key].update({
        "soil_q": soil_q, "clay_q": clay_q, "mineral_q": mineral_q,
        "consist_q": consist_q, "veg_q": veg_q, "moist_q": moist_q,
        "anom_q": anom_q, "final_score": S,
    })

ranked = sorted(scores.items(), key=lambda x: -x[1])

print("─" * 50)
print(f"RANKING  (soil={W_SOIL} | consist={W_CONSIST} | anomaly={W_ANOMALY})")
print("─" * 50)
for rank, (key, S) in enumerate(ranked, 1):
    print(f"  #{rank}  {key:12s}  S={S:.4f}  "
          f"(soil={details[key]['soil_q']:.3f}  "
          f"consist={details[key]['consist_q']:.3f}  "
          f"anom={details[key]['anom_q']:.3f})")

# ─── SENSITIVITY ANALYSIS (runs once) ────────────────────────────────────────

print("\nRunning sensitivity sweep...")
w_vals = np.arange(0.0, 1.05, 0.1)
ranking_counts = {k: {r: 0 for r in range(1, len(sites) + 1)} for k in sites}
n_tested = 0

for w1, w2, w3, w4 in iproduct(w_vals, w_vals, w_vals, w_vals):
    if w1 + w2 + w3 + w4 > 1.0 + 1e-6: continue
    w5 = max(0.0, 1.0 - w1 - w2 - w3 - w4)
    trial = {k: (w1*details[k]["soil_q"] + w2*details[k]["consist_q"]
                 + w3*details[k]["veg_q"] + w4*details[k]["moist_q"]
                 - w5*details[k]["anom_q"])
             for k in sites}
    for rank, (k, _) in enumerate(sorted(trial.items(), key=lambda x: -x[1]), 1):
        ranking_counts[k][rank] += 1
    n_tested += 1

print(f"  Tested {n_tested} combinations")
for key in sites:
    print(f"  {key:12s}  rank-1 in {ranking_counts[key][1]/n_tested*100:.0f}% of combinations")

# Sensitivity stacked bar
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(sites))
bottom = np.zeros(len(sites))
rank_colors = ['#2e8b22', '#90c040', '#e0a030', '#cc3333']
for rank in range(1, len(sites) + 1):
    vals = [ranking_counts[k][rank] / n_tested * 100 for k in sites]
    ax.bar(x, vals, bottom=bottom, color=rank_colors[rank-1],
           label=f"Rank #{rank}", alpha=0.9, edgecolor='white', linewidth=0.5)
    bottom += np.array(vals)
ax.set_xticks(x)
ax.set_xticklabels([all_data[k]["label"].replace('\n', ' ') for k in sites], fontsize=9)
ax.set_ylabel("% of weight combinations")
ax.set_title(f"Ranking Stability — date-robust scoring (n={n_tested} combinations)", fontsize=10)
ax.legend(loc='upper right', fontsize=9)
ax.set_ylim(0, 100); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "sensitivity_analysis.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: sensitivity_analysis.png")

# Component breakdown
fig, axes = plt.subplots(1, 5, figsize=(18, 4))   # was (1, 3)
bar_colors_sites = [COLORS[k] for k in sites]
labels_short     = [all_data[k]["label"].split('\n')[0] for k in sites]
for ax, (title, vals) in zip(axes, [
    ("Soil quality\n(1−Clay,1−BSI,1−Sal)",  [details[k]["soil_q"]    for k in sites]),
    ("Spatial consistency\n(1−CV)",          [details[k]["consist_q"] for k in sites]),
    ("Vegetation\n(NDVI adj)",               [details[k]["veg_q"]     for k in sites]),
    ("Moisture\n(NDMI)",                     [details[k]["moist_q"]   for k in sites]),
    ("Anomaly burden\n(RX∩AE, inverted)",    [1-details[k]["anom_q"]  for k in sites]),
]):
    
    ax.bar(labels_short, vals, color=bar_colors_sites, alpha=0.85, edgecolor='black', linewidth=0.7)
    ax.set_title(title, fontsize=9); ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', labelsize=8); ax.grid(axis='y', alpha=0.3)
    ax.axhline(0.5, color='gray', lw=0.7, ls='--', alpha=0.5)
fig.suptitle("Score component breakdown — date-robust axes", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "component_breakdown.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: component_breakdown.png")

# Leaderboard
fig, ax = plt.subplots(figsize=(8, 5))
keys_ranked = [k for k, _ in ranked]
score_vals  = [scores[k] for k in keys_ranked]
lbls        = [all_data[k]["label"].replace('\n', ' ') for k in keys_ranked]
medals      = ["#1 🥇", "#2 🥈", "#3 🥉", "#4"]
bars = ax.barh(lbls[::-1], score_vals[::-1],
               color=[COLORS[k] for k in reversed(keys_ranked)],
               alpha=0.9, edgecolor='black', linewidth=0.8)
for i, (bar, key) in enumerate(zip(bars, reversed(keys_ranked))):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f"{medals[len(sites)-1-i]}  S={scores[key]:.3f}", va='center', fontsize=10)
ax.set_xlabel("Investment Score S  (date-robust)")
ax.set_title(
    f"Land Investment Ranking\n"
    f"soil={W_SOIL} clay={W_CLAY} consist={W_CONSIST} "
    f"veg={W_VEG} moist={W_MOISTURE} anom={W_ANOMALY}×{ANOMALY_SIGN:+.0f}",
    fontsize=10
)
ax.set_xlim(min(score_vals) - 0.08, max(score_vals) + 0.18)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "final_leaderboard.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: final_leaderboard.png")

# ─── MANIFEST ────────────────────────────────────────────────────────────────

try:
    from pipeline_utils import save_step
    save_step("anomaly", {
        "plots": [
            {"id": "ae_loss",      "path": "plots/03_anomaly/ae_training_loss.png",    "title": "AE Training Loss"},
            {"id": "rx_maps",      "path": "plots/03_anomaly/anomaly_rx_maps.png",     "title": "RX Anomaly Maps"},
            {"id": "ae_maps",      "path": "plots/03_anomaly/anomaly_ae_maps.png",     "title": "AE Anomaly Maps"},
            {"id": "agreement",    "path": "plots/03_anomaly/anomaly_agreement.png",   "title": "Anomaly Agreement"},
            {"id": "sensitivity",  "path": "plots/03_anomaly/sensitivity_analysis.png","title": "Ranking Stability"},
            {"id": "components",   "path": "plots/03_anomaly/component_breakdown.png", "title": "Score Components"},
            {"id": "leaderboard",  "path": "plots/03_anomaly/final_leaderboard.png",   "title": "Leaderboard"},
        ],
        "training_log": "plots/03_anomaly/training_progress.json",
        "ranking": [k for k, _ in ranked],
        "per_site": {
            key: {
                "final_score":      round(float(scores[key]), 4),
                "components": {
                    "soil_quality":      round(float(details[key]["soil_q"]),    4),
                    "clay_quality":      round(float(details[key]["clay_q"]),    4),
                    "mineral_quality":   round(float(details[key]["mineral_q"]), 4),
                    "spatial_consist":   round(float(details[key]["consist_q"]), 4),
                    "veg_quality":       round(float(details[key]["veg_q"]),     4),
                    "moisture_quality":  round(float(details[key]["moist_q"]),   4),
                },
                "anomaly": {
                    "burden_raw":   round(float(details[key]["anomaly_burden"]), 4),
                    "burden_norm":  round(float(details[key]["anom_q"]),         4),
                    "sign_used":    ANOMALY_SIGN,
                    "interpretation": "higher = more mineral interest" if ANOMALY_SIGN > 0
                                    else "higher = more subsurface risk",
                },
                "raw_indices": {
                    "ndvi":      round(float(details[key]["ndvi_raw"]),       4),
                    "ndvi_adj":  round(float(details[key]["ndvi_adj"]),       4),
                    "ndmi":      round(float(details[key]["ndmi_mean"]),      4),
                    "clay":      round(float(details[key]["clay_mean"]),      4),
                    "bsi":       round(float(details[key]["bsi_mean"]),       4),
                    "iron":      round(float(details[key]["iron_mean"]),      4),
                    "carbonate": round(float(details[key]["carbonate_mean"]), 4),
                    "salinity":  round(float(details[key]["salinity_mean"]),  4),
                },
            }
            for key in sites
        },
        "anomaly_config": {
            "sign": ANOMALY_SIGN,
            "threshold_percentile": ANOMALY_PERCENTILE,
            "rx_threshold": float(rx_threshold),
            "ae_threshold": float(ae_threshold),
        },
        "use_case": args.use_case,
        "weights_used": {
            "W_SOIL": W_SOIL, "W_CLAY": W_CLAY, "W_MINERAL": W_MINERAL,
            "W_CONSIST": W_CONSIST, "W_VEG": W_VEG,
            "W_MOISTURE": W_MOISTURE, "W_ANOMALY": W_ANOMALY,
            "anomaly_sign": ANOMALY_SIGN,
        },
        "sensitivity": {
            key: {f"rank{r}_pct": round(ranking_counts[key][r] / n_tested * 100, 1)
                  for r in range(1, 5)}
            for key in sites
        },
    })
except ImportError:
    print("  pipeline_utils not found — skipping manifest update")

# ─── DONE ────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("DONE — Script 03")
print("=" * 65)
for rank, (key, S) in enumerate(ranked, 1):
    print(f"  #{rank}  {all_data[key]['label'].replace(chr(10), ' '):35s}  S={S:.4f}")