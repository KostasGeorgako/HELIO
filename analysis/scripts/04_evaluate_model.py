"""
04_evaluate_model.py — Autoencoder Validation & Evaluation
===========================================================
This script:
  1. Re-trains the AE with a proper 80/20 train/val split
  2. Saves the model to disk
  3. Produces a full suite of evaluation plots:

     eval_01_training_curves.png   — train vs val loss per epoch
     eval_02_error_distribution.png— histogram of reconstruction errors
     eval_03_error_by_site.png     — boxplot per site (who is anomalous?)
     eval_04_latent_space.png      — PCA of 16D bottleneck (where do sites sit?)
     eval_05_per_band_error.png    — which spectral regions are hardest to reconstruct?
     eval_06_synthetic_anomaly.png — inject known anomalies, verify detection

WHY EACH PLOT EXISTS:
  Training curves    → confirms model learned, not memorised (val loss must track train)
  Error distribution → must be right-skewed with a clear tail (anomalies live in the tail)
  Error by site      → shows which site has the most spectrally unusual pixels
  Latent space       → confirms the 16D bottleneck separates site types meaningfully
  Per-band error     → tells you which wavelengths the AE struggles with most
  Synthetic anomaly  → ground-truth sanity check: if you inject a known weird spectrum,
                       does the AE catch it? If not, the detector is broken.

Run after scripts 01–03. Requires the same DATA_ROOT and PLOTS_DIR structure.
"""

import os, sys
import xml.etree.ElementTree as ET
import warnings
warnings.filterwarnings("ignore")

from spectral_common import (
    SCENES, COLORS, RANDOM_SEED,
    load_all_scenes, extract_valid_spectra,
    SpectralAE, load_model_checkpoint,
    good_band_mask, band_at,
)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


PLOTS_DIR   = "../data/plots/04_evaluation"
RANDOM_SEED = 42
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH  = "../data/spectral_ae.pt"
os.makedirs(PLOTS_DIR, exist_ok=True)


# Score everything
def score_spectra(x):
    norm = scaler.transform(x.reshape(-1, N_BANDS))
    return model.reconstruction_error(torch.tensor(norm, dtype=torch.float32).to(DEVICE))


# ─── LOAD + POOL ─────────────────────────────────────────────────────────────

print("=" * 65)
print(f"Loading scenes  |  device: {DEVICE}")
print("=" * 65)
all_data, wl_ref, good_ref, N_BANDS = load_all_scenes(smooth=True, verbose=True)
sites = list(all_data.keys())

pool_pixels_list, pool_labels_list = [], []
for key, d in all_data.items():
    spectra, _ = extract_valid_spectra(d["cube"], d["wl"], d["valid"], d["good"])
    pool_pixels_list.append(spectra)
    pool_labels_list.extend([key] * len(spectra))

pool_pixels = np.vstack(pool_pixels_list).astype(np.float32)
pool_labels = np.array(pool_labels_list)
print(f"\nPool: {len(pool_pixels)} pixels × {N_BANDS} bands")

# ─── LOAD TRAINED MODEL FROM CHECKPOINT ──────────────────────────────────────
print(f"\nLoading model from {MODEL_PATH} ...")
model, scaler, ckpt = load_model_checkpoint(DEVICE)
pool_norm    = scaler.transform(pool_pixels)
N_BANDS      = ckpt["n_bands"]
train_losses = ckpt["train_losses"]
val_losses   = ckpt["val_losses"]
train_idx    = ckpt["train_idx"]
val_idx      = ckpt["val_idx"]
final_ratio  = val_losses[-1] / (train_losses[-1] + 1e-10)

print(f"  {len(train_idx)} train / {len(val_idx)} val pixels")
print(f"  Final train={train_losses[-1]:.6f}  val={val_losses[-1]:.6f}  ratio={final_ratio:.2f}")

print(f"  Model loaded  |  {len(train_idx)} train / {len(val_idx)} val pixels")
print(f"  Final train loss : {train_losses[-1]:.6f}")
print(f"  Final val loss   : {val_losses[-1]:.6f}")
print(f"  Val/train ratio  : {val_losses[-1]/train_losses[-1]:.2f}")

# ─── COMPUTE RECONSTRUCTION ERRORS (full pool) ───────────────────────────────

model.eval()
all_norm_t   = torch.tensor(pool_norm, dtype=torch.float32).to(DEVICE)
with torch.no_grad():
    recon        = model(all_norm_t).cpu().numpy()
    latent       = model.encode(all_norm_t).cpu().numpy()
    errors       = ((recon - pool_norm) ** 2).mean(axis=1)          # per-pixel scalar
    per_band_err = ((recon - pool_norm) ** 2).mean(axis=0)          # per-band scalar

# ─── PLOT 1 — Training curves ─────────────────────────────────────────────────

print("\nGenerating evaluation plots...")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

ax = axes[0]
ax.plot(train_losses, label="Train loss", color="steelblue", lw=1.5)
ax.plot(val_losses,   label="Val loss",   color="tomato",    lw=1.5, ls="--")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_title("Training vs validation loss\n"
             "Val should track train — rising val = overfitting")
ax.legend()
ax.grid(True, alpha=0.3)

# Gap ratio over time
ax2 = axes[1]
ratio = np.array(val_losses) / (np.array(train_losses) + 1e-10)
ax2.plot(ratio, color="darkorange", lw=1.5)
ax2.axhline(1.0, color="green", lw=0.8, ls="--", label="Perfect match (ratio=1)")
ax2.axhline(1.5, color="orange", lw=0.8, ls="--", label="Mild overfit threshold")
ax2.axhline(2.5, color="red",    lw=0.8, ls="--", label="Severe overfit threshold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Val loss / Train loss")
ax2.set_title("Overfitting ratio over time\n"
              "Stable near 1.0 = healthy   |   Rising = overfitting")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, max(4.0, ratio.max() * 1.1))

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "eval_01_training_curves.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: eval_01_training_curves.png")

# ─── PLOT 2 — Reconstruction error distribution ───────────────────────────────

threshold_90 = np.percentile(errors, 90)

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(errors, bins=40, color="steelblue", alpha=0.7, edgecolor="white", linewidth=0.5)
ax.axvline(np.median(errors),  color="green",  lw=1.5, ls="--",
           label=f"Median = {np.median(errors):.5f}")
ax.axvline(threshold_90,       color="red",    lw=1.5, ls="--",
           label=f"90th pct = {threshold_90:.5f}  (anomaly threshold)")
ax.set_xlabel("Reconstruction MSE")
ax.set_ylabel("Number of pixels")
ax.set_title(
    "Reconstruction error distribution — all pooled pixels\n"
    "Healthy: right-skewed with a clear tail. Flat/symmetric = model memorised everything."
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "eval_02_error_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: eval_02_error_distribution.png")

# ─── PLOT 3 — Error by site (boxplot) ────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))

site_errors = [errors[pool_labels == key] for key in sites]
bp = ax.boxplot(site_errors, patch_artist=True, widths=0.5,
                medianprops=dict(color="black", lw=2))
for patch, key in zip(bp["boxes"], sites):
    patch.set_facecolor(COLORS[key])
    patch.set_alpha(0.8)

ax.axhline(threshold_90, color="red", lw=1.2, ls="--",
           label=f"90th pct anomaly threshold = {threshold_90:.5f}")
ax.set_xticks(range(1, len(sites) + 1))
ax.set_xticklabels([all_data[k]["label"] for k in sites], fontsize=9)
ax.set_ylabel("Reconstruction MSE")
ax.set_title(
    "Reconstruction error per site\n"
    "Higher median = site is spectrally unusual relative to pooled population\n"
    "Wider box = more internal heterogeneity within site"
)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "eval_03_error_by_site.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: eval_03_error_by_site.png")

# ─── PLOT 4 — Latent space (PCA of 16D bottleneck) ───────────────────────────
#
# The 16D latent representation is visualised in 2D via PCA.
# What to look for:
#   - October sites (arkadia, magnisia) should cluster together
#     because they share the same dry-soil spectral signature
#   - Arkadia2 (May) and Veroia (Aug) should sit apart — different phenology
#   - If ALL sites overlap completely, the bottleneck isn't encoding
#     meaningful structure (model failed to learn)
#   - Wide spread within a site = heterogeneous land (useful signal)

pca_latent = PCA(n_components=2, random_state=RANDOM_SEED)
latent_2d  = pca_latent.fit_transform(latent)
var_exp    = pca_latent.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(8, 7))
for key in sites:
    mask = pool_labels == key
    ax.scatter(latent_2d[mask, 0], latent_2d[mask, 1],
               c=COLORS[key], label=all_data[key]["label"],
               s=30, alpha=0.65, edgecolors="white", linewidths=0.3)

ax.set_xlabel(f"Latent PC1 ({var_exp[0]*100:.1f}% variance)")
ax.set_ylabel(f"Latent PC2 ({var_exp[1]*100:.1f}% variance)")
ax.set_title(
    "Latent space (PCA of 24D AE bottleneck)\n"
    "Oct sites should cluster together · May/Aug sites should separate\n"
    "Wide spread within a site = heterogeneous land"
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "eval_04_latent_space.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: eval_04_latent_space.png")

# ─── PLOT 5 — Per-band reconstruction error ───────────────────────────────────
#
# Shows which wavelength regions the AE finds hardest to reconstruct.
# Expected:
#   - Low error in VNIR (simple smooth curves, lots of training signal)
#   - Higher error at the red-edge (steep slope, harder to pin down)
#   - Higher error near the bad-band boundaries (noisy regions)
#   - SWIR mineral features (2200nm) may have higher error if the AE
#     hasn't fully captured the clay absorption shape

wl_good = wl_ref[good_ref]   # wavelengths of the usable bands only

fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(wl_good, per_band_err, color="steelblue", lw=1.2)
ax.fill_between(wl_good, 0, per_band_err, color="steelblue", alpha=0.2)

# Highlight key spectral features
for nm, lbl, col in [
    (720,  "Red-edge",  "green"),
    (1640, "NDMI",      "blue"),
    (2200, "Clay",      "orange"),
]:
    ax.axvline(nm, color=col, lw=0.8, ls="--", alpha=0.6)
    ax.text(nm + 10, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 0.001,
            lbl, fontsize=7, color=col)

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Mean squared reconstruction error")
ax.set_title(
    "Per-band reconstruction error\n"
    "High error = the AE struggles to reconstruct this wavelength region\n"
    "Key features (red-edge, clay at 2200nm) should show moderate spikes"
)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "eval_05_per_band_error.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: eval_05_per_band_error.png")

# ─── PLOT 6 — Synthetic anomaly injection test ────────────────────────────────
#
# This is the ground-truth sanity check.
# We create three types of known synthetic anomalies and verify the AE
# assigns them higher reconstruction error than normal pixels.
#
# If the AE FAILS to detect synthetic anomalies, the detector is broken.
# If it detects them cleanly, you have evidence it works as intended.

print("\nRunning synthetic anomaly injection test...")

# Take the mean spectrum as the "typical pixel" baseline
mean_spectrum = pool_pixels.mean(axis=0)   # in original (unscaled) space

# Anomaly type 1: Extreme clay — amplify the 2200nm absorption feature
# More realistic strong clay anomaly — depress the whole OH absorption feature
clay_region = (wl_good >= 2150) & (wl_good <= 2280)
synth_clay = mean_spectrum.copy()
synth_clay[clay_region] *= 0.35

# Anomaly type 2: Salt/mineral crust — uniformly high reflectance across all bands
synth_salt = mean_spectrum.copy()
synth_salt = synth_salt * 0.3 + 0.45   # flatten and lift to high reflectance

# Anomaly type 3: Random noise spectrum — completely unphysical
rng = np.random.default_rng(RANDOM_SEED)
synth_noise = rng.uniform(0.0, 0.5, size=mean_spectrum.shape).astype(np.float32)

# Normal pixel sample (100 random real pixels from val set for comparison)
# 1. Sample normal pixels from both train AND val, not just val[:100]
#    to get a representative spread including some above threshold
normal_sample_idx = np.random.default_rng(42).choice(
    len(pool_pixels), size=100, replace=False
)
normal_sample = pool_pixels[normal_sample_idx]

# 2. Report what fraction of normal pixels exceed threshold, 
#    not whether the mean does — the mean SHOULD be below threshold
normal_errors = score_spectra(normal_sample)
normal_pct_flagged = (normal_errors > threshold_90).mean() * 100
print(f"Normal pixels flagged: {normal_pct_flagged:.1f}%  (expect ~10% by construction)")


normal_errors = score_spectra(normal_sample)
clay_error    = score_spectra(synth_clay.reshape(1, -1))[0]
salt_error    = score_spectra(synth_salt.reshape(1, -1))[0]
noise_error   = score_spectra(synth_noise.reshape(1, -1))[0]

print(f"\n  Normal pixels:        mean={normal_errors.mean():.5f}  "
      f"std={normal_errors.std():.5f}")
print(f"  Synthetic clay:       error={clay_error:.5f}  "
      f"{'✓ DETECTED' if clay_error > threshold_90 else '✗ missed'}")
print(f"  Synthetic salt/crust: error={salt_error:.5f}  "
      f"{'✓ DETECTED' if salt_error > threshold_90 else '✗ missed'}")
print(f"  Random noise:         error={noise_error:.5f}  "
      f"{'✓ DETECTED' if noise_error > threshold_90 else '✗ missed'}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: spectra comparison
ax = axes[0]
ax.plot(wl_good, mean_spectrum,  label="Typical pixel (mean)",   color="steelblue", lw=1.5)
ax.plot(wl_good, synth_clay,     label="Synthetic: deep clay",   color="orange",    lw=1.5, ls="--")
ax.plot(wl_good, synth_salt,     label="Synthetic: salt/crust",  color="red",       lw=1.5, ls="-.")
ax.plot(wl_good, synth_noise,    label="Synthetic: random noise", color="gray",      lw=1.0, alpha=0.7)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Reflectance")
ax.set_title("Injected synthetic anomaly spectra vs typical pixel")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 0.65)

# Right: error comparison bar
ax = axes[1]
categories = ["Normal\n(val set\nmean)", "Synthetic\nclay", "Synthetic\nsalt", "Random\nnoise"]
err_vals   = [normal_errors.mean(), clay_error, salt_error, noise_error]
err_colors = ["steelblue", "orange", "red", "gray"]

bars = ax.bar(categories, err_vals, color=err_colors, alpha=0.85,
              edgecolor="black", linewidth=0.7)
ax.axhline(threshold_90, color="red", lw=1.5, ls="--",
           label=f"Anomaly threshold (p90) = {threshold_90:.5f}")
ax.set_ylabel("Reconstruction MSE")
ax.set_title("Reconstruction error: normal vs synthetic anomalies\n"
             "Bars above red line = correctly detected as anomalous")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

# Annotate each bar
for bar, val in zip(bars, err_vals):
    status = "✓" if val > threshold_90 else "✗"
    ax.text(bar.get_x() + bar.get_width() / 2, val + threshold_90 * 0.02,
            f"{status}\n{val:.5f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "eval_06_synthetic_anomaly.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: eval_06_synthetic_anomaly.png")

# ─── DONE ─────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("DONE — Script 04 (Evaluation)")
print("=" * 65)
print(f"  Model saved : {os.path.abspath(MODEL_PATH)}")
print(f"  Plots saved : {os.path.abspath(PLOTS_DIR)}")
print()
print("  Reading the plots:")
print("  eval_01_training_curves.png")
print("    Left:  train (blue) and val (red dashed) should decrease together.")
print("           If val rises while train falls → overfitting.")
print("    Right: val/train ratio should stay near 1.0 and be stable.")
print()
print("  eval_02_error_distribution.png")
print("    Should be right-skewed (long tail to the right).")
print("    Symmetric / narrow bell = model memorised everything = broken.")
print()
print("  eval_03_error_by_site.png")
print("    Boxplot per site. Higher box = more spectrally unusual site.")
print("    Wide box = heterogeneous land within that site.")
print()
print("  eval_04_latent_space.png")
print("    October sites should cluster. May/Aug should separate.")
print("    Complete overlap of all sites = bottleneck learned nothing.")
print()
print("  eval_05_per_band_error.png")
print("    Peaks at red-edge and 2200nm are expected and healthy.")
print("    Flat line near zero = model reconstructs everything trivially = overfit.")
print()
print("  eval_06_synthetic_anomaly.png")
print("    All synthetic anomalies must score above the red threshold line.")
print("    Any ✗ = the detector is not sensitive enough.")
print()
print(f"  Val/train ratio: {final_ratio:.2f}  "
      f"({'✓ healthy' if final_ratio < 1.5 else '⚠ check training curves'})")