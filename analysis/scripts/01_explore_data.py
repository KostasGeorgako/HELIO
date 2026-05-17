"""
01_explore_data.py — Makethlon Hyperspectral Exploration
=========================================================
What this script does (in order):
  1. Loads all 4 scenes and prints their structure (shape, valid pixels, date)
  2. Shows the quality masks (cloud/shadow/haze) so you know what gets removed
  3. Plots RGB composites for each site (true-color, 660/560/490 nm)
  4. Plots false-color composites (NIR/Red/Green — vegetation pops red)
  5. Overlays mean spectra of all 4 sites on one chart with absorption regions marked
  6. Shows per-band variance across each scene (which bands carry information)

Run from: anywhere. Set DATA_ROOT below to match your folder structure.
Output:   plots saved to PLOTS_DIR
"""

# 01_explore_data.py and 02_indices_maps.py — top of file
from spectral_common import (
    DATA_ROOT, SCENES, COLORS,
    good_band_mask, load_scene, smooth_cube,
    load_all_scenes, extract_valid_spectra,
    scene_path, band_at,
)

import os, sys, json
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio


def make_rgb(cube, wl, r_nm=660, g_nm=560, b_nm=490, percentile=2):
    r = cube[band_at(wl, r_nm)]
    g = cube[band_at(wl, g_nm)]
    b = cube[band_at(wl, b_nm)]
    rgb = np.stack([r, g, b], axis=-1)
    valid = np.all(np.isfinite(rgb), axis=-1)
    lo = np.nanpercentile(rgb[valid], percentile)
    hi = np.nanpercentile(rgb[valid], 100 - percentile)
    rgb_disp = np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1)
    rgb_disp[~valid] = 0.5
    return rgb_disp, valid


PLOTS_DIR = "../data/plots/01_explore"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── STEP 1: Load all scenes and print summary ───────────────────────────────


print("=" * 65)
print("Loading scenes...")
print("=" * 65)
all_data, wl_ref, good_ref, N_BANDS = load_all_scenes(smooth=False, verbose=True)
good_bands = good_ref

if not all_data:
    print("\nNo scenes loaded. Check DATA_ROOT path.")
    sys.exit(1)


# ─── STEP 2: Quality mask visualisation ──────────────────────────────────────

print("\n" + "=" * 65)
print("STEP 2 — Quality masks")
print("=" * 65)

fig, axes = plt.subplots(1, len(all_data), figsize=(4 * len(all_data), 4))
if len(all_data) == 1:
    axes = [axes]

for ax, (key, d) in zip(axes, all_data.items()):
    valid = d["valid"]
    # Show valid (white) vs masked (black) pixels
    display = np.zeros((*valid.shape, 3))
    display[valid]  = [1, 1, 1]   # valid → white
    display[~valid] = [0.2, 0.2, 0.8]  # masked → blue

    ax.imshow(display, interpolation='nearest')
    ax.set_title(d["label"].replace('\n', ' '), fontsize=9)
    pct = valid.sum() / valid.size * 100
    ax.set_xlabel(f"{valid.sum()} valid / {valid.size} total  ({pct:.0f}%)", fontsize=8)
    ax.axis('off')

# Legend
white_patch = mpatches.Patch(color='white', label='Valid pixel', linewidth=0.5,
                              edgecolor='gray')
blue_patch  = mpatches.Patch(color=[0.2, 0.2, 0.8], label='Masked (cloud/shadow/haze)')
fig.legend(handles=[white_patch, blue_patch], loc='lower center',
           ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Quality Masks — White = usable pixel,  Blue = masked out", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "01_quality_masks.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: 01_quality_masks.png")

# ─── STEP 3: RGB True-Colour Composites ──────────────────────────────────────

print("\n" + "=" * 65)
print("STEP 3 — RGB true-colour composites (660/560/490 nm)")
print("=" * 65)

fig, axes = plt.subplots(1, len(all_data), figsize=(4 * len(all_data), 4.5))
if len(all_data) == 1:
    axes = [axes]

for ax, (key, d) in zip(axes, all_data.items()):
    rgb, valid = make_rgb(d["cube"], d["wl"])
    ax.imshow(rgb, interpolation='nearest')
    ax.set_title(d["label"], fontsize=9)
    B, H, W = d["cube"].shape
    ax.text(0.03, 0.04, f"{H}×{W} px @ 30m GSD", transform=ax.transAxes,
            fontsize=7, color='white', bbox=dict(boxstyle='round', fc='black', alpha=0.6))
    ax.axis('off')

fig.suptitle("True-Colour RGB  (660 nm / 560 nm / 490 nm)\n"
             "Brown/orange = dry bare soil   |   Green = active vegetation   |"
             "   Grey = nodata", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "02_rgb_truecolor.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: 02_rgb_truecolor.png")

# ─── STEP 4: False-Colour Composites (NIR/Red/Green) ─────────────────────────

print("\n" + "=" * 65)
print("STEP 4 — False-colour composites (NIR/Red/Green = 860/660/560 nm)")
print("       Red areas = active healthy vegetation (high NIR reflectance)")
print("=" * 65)

fig, axes = plt.subplots(1, len(all_data), figsize=(4 * len(all_data), 4.5))
if len(all_data) == 1:
    axes = [axes]

for ax, (key, d) in zip(axes, all_data.items()):
    # NIR → R channel,  Red → G channel,  Green → B channel
    # Healthy vegetation reflects NIR heavily → shows up bright red
    rgb_fc, valid = make_rgb(d["cube"], d["wl"], r_nm=860, g_nm=660, b_nm=560)
    ax.imshow(rgb_fc, interpolation='nearest')
    ax.set_title(d["label"], fontsize=9)
    ax.axis('off')

fig.suptitle("False-Colour Composite  (NIR / Red / Green → R / G / B)\n"
             "Bright red = healthy vegetation   |   Brown/cyan = dry soil   |"
             "   Grey = nodata", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "03_rgb_falsecolor.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: 03_rgb_falsecolor.png")

# ─── STEP 5: Overlaid Mean Spectra ───────────────────────────────────────────

print("\n" + "=" * 65)
print("STEP 5 — Mean spectra overlaid (all 4 sites)")
print("=" * 65)
 
# Use constrained_layout instead of tight_layout — more robust
fig, axes = plt.subplots(2, 1, figsize=(16, 9),
                         constrained_layout=True)
 
ax_full = axes[0]  # Full spectral range 420–2445 nm
ax_zoom = axes[1]  # Zoomed VNIR 420–1050 nm
 
for key, d in all_data.items():
    cube  = d["cube"]
    label = d["label"].replace('\n', ' ')
    color = COLORS[key]
    mean_spec = np.nanmean(cube, axis=(1, 2))
    ax_full.plot(d["wl"], mean_spec, color=color, lw=1.5, label=label, alpha=0.9)
    ax_zoom.plot(d["wl"], mean_spec, color=color, lw=1.8, label=label, alpha=0.9)
 
# ── Bad-band shading ──────────────────────────────────────────────────────────
for ax in axes:
    ax.axvspan(850,  1000, color='gray', alpha=0.18, zorder=0)
    ax.axvspan(1340, 1460, color='red',  alpha=0.12, zorder=0)
    ax.axvspan(1790, 1960, color='red',  alpha=0.12, zorder=0)
 
# ── Vertical reference lines on the full spectrum ─────────────────────────────
ref_lines = {
    490:  ("Blue",        0.38),
    560:  ("Green",       0.40),
    660:  ("Red",         0.42),
    720:  ("Red-edge",    0.44),
    860:  ("NIR",         0.46),
    1640: ("SWIR1\nNDMI", 0.46),
    2200: ("Clay\n2200nm",0.46),
}
for nm, (lbl, ypos) in ref_lines.items():
    ax_full.axvline(nm, color='black', lw=0.6, ls='--', alpha=0.35)
    ax_full.text(nm, ypos, lbl, fontsize=6.5, ha='center', va='bottom',
                 color='black', alpha=0.6, rotation=0)
 
# ── Labels and formatting — full spectrum ─────────────────────────────────────
ax_full.set_xlim(420, 2445)
ax_full.set_ylim(0, 0.52)
ax_full.set_ylabel("Reflectance", fontsize=11)
ax_full.set_xlabel("Wavelength (nm)", fontsize=11)
ax_full.legend(fontsize=9, loc='upper right')
ax_full.grid(True, alpha=0.25)
ax_full.set_title(
    "Mean surface reflectance — all 4 sites  (full range 420–2445 nm)\n"
    "Grey = stitch noise (drop)  |  Red = water vapour absorption (drop)",
    fontsize=10
)
 
# ── Text labels for bad-band regions ─────────────────────────────────────────
ax_full.text(925,   0.01, "Stitch\nnoise\n(drop)", fontsize=7,
             ha='center', color='gray', style='italic')
ax_full.text(1400,  0.01, "H₂O\n(drop)", fontsize=7,
             ha='center', color='red', style='italic')
ax_full.text(1875,  0.01, "H₂O\n(drop)", fontsize=7,
             ha='center', color='red', style='italic')
 
# ── Zoomed VNIR panel ─────────────────────────────────────────────────────────
ax_zoom.set_xlim(420, 1050)
ax_zoom.set_ylim(0, 0.52)
ax_zoom.set_ylabel("Reflectance", fontsize=11)
ax_zoom.set_xlabel("Wavelength (nm)", fontsize=11)
ax_zoom.grid(True, alpha=0.25)
ax_zoom.legend(fontsize=9, loc='upper left')
ax_zoom.set_title(
    "Zoomed: VNIR region (420–1050 nm) — vegetation indices computed here",
    fontsize=10
)
 
# ── Non-overlapping annotations on zoom panel ────────────────────────────────
annot_cfg = dict(fontsize=8, color='dimgray',
                 arrowprops=dict(arrowstyle='->', color='dimgray', lw=0.8))
 
ax_zoom.annotate("Chlorophyll absorbs\nblue & red here",
                 xy=(475, 0.06), xytext=(520, 0.16), **annot_cfg)
ax_zoom.annotate("Red-edge:\nvegetation\n'jump'",
                 xy=(715, 0.18), xytext=(640, 0.30), **annot_cfg)
ax_zoom.annotate("NIR plateau\n(cell reflectance)",
                 xy=(870, 0.35), xytext=(920, 0.22), **annot_cfg)
ax_zoom.annotate("Stitch noise\nregion — drop",
                 xy=(930, 0.08), xytext=(980, 0.15), **annot_cfg)
 
plt.savefig(os.path.join(PLOTS_DIR, "04_mean_spectra_overlay.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 04_mean_spectra_overlay.png")

# ─── STEP 6: Per-band variance — which bands carry info ──────────────────────

print("\n" + "=" * 65)
print("STEP 6 — Per-band spatial variance across each scene")
print("       High variance = this wavelength shows spatial structure")
print("       Low variance  = uniform / noise band")
print("=" * 65)

fig, ax = plt.subplots(figsize=(14, 4))

for key, d in all_data.items():
    cube  = d["cube"]
    color = COLORS[key]
    label = d["label"].replace('\n', ' ')
    # Variance per band over valid pixels
    band_var = np.nanvar(cube, axis=(1, 2))
    ax.plot(d["wl"], band_var, color=color, lw=1.2, label=label, alpha=0.85)

ax.axvspan(850,  1000, color='gray', alpha=0.15, label='Stitch noise')
ax.axvspan(1340, 1460, color='red',  alpha=0.15, label='H₂O (drop)')
ax.axvspan(1790, 1960, color='red',  alpha=0.15)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Spatial variance of reflectance")
ax.set_title("Per-band spatial variance — spikes in bad regions confirm noise; "
             "real signal shows in VNIR and SWIR",
             fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.25)
ax.set_xlim(wl_ref.min(), wl_ref.max())
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "05_band_variance.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: 05_band_variance.png")

# ─── STEP 7: Per-pixel spectrum browser — show individual spectra ─────────────

print("\n" + "=" * 65)
print("STEP 7 — Individual pixel spectra sample (first 10 valid pixels per site)")
print("=" * 65)

fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True, sharex=True)
axes_flat = axes.flatten()

for ax, (key, d) in zip(axes_flat, all_data.items()):
    cube  = d["cube"]
    wl    = d["wl"]
    valid = d["valid"]
    color = COLORS[key]

    rows, cols = np.where(valid)
    n_show = min(20, len(rows))
    idx = np.random.choice(len(rows), n_show, replace=False)

    for i in idx:
        spec = cube[:, rows[i], cols[i]]
        ax.plot(wl, spec, color=color, alpha=0.3, lw=0.8)

    # Mean on top
    mean_spec = np.nanmean(cube, axis=(1, 2))
    ax.plot(wl, mean_spec, color='black', lw=2, label='Site mean')

    ax.axvspan(850,  1000, color='gray', alpha=0.15)
    ax.axvspan(1340, 1460, color='red',  alpha=0.10)
    ax.axvspan(1790, 1960, color='red',  alpha=0.10)
    ax.set_title(d["label"].replace('\n', ' '), fontsize=9)
    ax.set_ylim(0, 0.6)
    ax.set_xlim(wl_ref.min(), wl_ref.max())
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7)

axes[1, 0].set_xlabel("Wavelength (nm)")
axes[1, 1].set_xlabel("Wavelength (nm)")
axes[0, 0].set_ylabel("Reflectance")
axes[1, 0].set_ylabel("Reflectance")

fig.suptitle("Individual pixel spectra (faint) vs site mean (black)\n"
             "Spread of faint lines = spatial heterogeneity within the plot",
             fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "06_pixel_spectra_sample.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: 06_pixel_spectra_sample.png")

# ─── SUMMARY ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("DONE — Summary")
print("=" * 65)
print(f"  Output folder: {os.path.abspath(PLOTS_DIR)}")
print()
print("  Files generated:")
print("    01_quality_masks.png      — which pixels survive cloud/shadow masking")
print("    02_rgb_truecolor.png      — how the sites look in visible light")
print("    03_rgb_falsecolor.png     — NIR composite (vegetation = bright red)")
print("    04_mean_spectra_overlay.png — all 4 sites spectra on one chart")
print("    05_band_variance.png      — which bands carry spatial information")
print("    06_pixel_spectra_sample.png — individual pixels vs site mean")
print()
print("  DATE NOTE: Arkadia2 (May) will show very different NIR/Red-edge")
print("  from the October/August scenes. This is expected and correct.")
print("  Do NOT directly compare NDVI across scenes with different dates.")
print("  Soil/clay/anomaly indices are more date-robust — Script 02 handles this.")


from pipeline_utils import save_step
save_step("explore", {
    "plots": [
        {"id": "quality_masks",  "path": "plots/01_explore/01_quality_masks.png",       "title": "Quality Masks"},
        {"id": "rgb_truecolor",  "path": "plots/01_explore/02_rgb_truecolor.png",        "title": "True-Colour RGB"},
        {"id": "rgb_falsecolor", "path": "plots/01_explore/03_rgb_falsecolor.png",       "title": "False-Colour (NIR)"},
        {"id": "mean_spectra",   "path": "plots/01_explore/04_mean_spectra_overlay.png", "title": "Mean Spectra"},
        {"id": "band_variance",  "path": "plots/01_explore/05_band_variance.png",        "title": "Band Variance"},
        {"id": "pixel_spectra",  "path": "plots/01_explore/06_pixel_spectra_sample.png", "title": "Pixel Spectra Sample"},
    ],
    "site_stats": {
        key: {"valid_pixels": int(d["valid"].sum()), "usable_bands": int(good_band_mask(d["wl"]).sum())}
        for key, d in all_data.items()
    }
})