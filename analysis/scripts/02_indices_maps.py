"""
02_indices_maps.py — Spectral Indices & Per-Site Scoring
=========================================================
What this script does:
  1. Loads all 4 scenes (using same loader as script 01)
  2. Removes bad bands
  3. Computes per-pixel spectral indices:
       NDVI  — vegetation amount / biomass
       NDRE  — red-edge vegetation index (more sensitive at high biomass)
       NDMI  — leaf/soil moisture
       BSI   — bare soil fraction
       Clay  — clay mineral index (OH-bond absorption depth at 2200 nm)
  4. Plots each index as a spatial heatmap per site (18×18 px grid)
  5. Computes per-site statistics: mean, std, CV (coefficient of variation)
  6. Builds a final comparison table
  7. IMPORTANT: flags date-sensitive indices clearly

⚠ Date caveats (printed in output):
   NDVI/NDRE/NDMI are phenology-sensitive.
   Arkadia2 (May) will legitimately show higher NDVI than Oct/Aug scenes
   even if the land quality is equivalent — it's just greener in May.
   BSI and Clay Index are date-robust (soil mineral properties don't change seasonally).
   Anomaly detection (script 03) handles this by learning within-population statistics.

Run from: anywhere. Same DATA_ROOT as script 01.
Output:   plots saved to PLOTS_DIR
"""

# 01_explore_data.py and 02_indices_maps.py — top of file
from spectral_common import (
    DATA_ROOT, SCENES, COLORS,
    good_band_mask, load_scene, smooth_cube,
    load_all_scenes, extract_valid_spectra,
    scene_path, band_at,
)

import os, sys
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.signal import savgol_filter
import rasterio


PLOTS_DIR = "../data/plots/01_explore"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── SPECTRAL INDEX FUNCTIONS ─────────────────────────────────────────────────

def safe_ratio(a, b):
    """(a - b) / (a + b) with zero-division guard."""
    denom = a + b
    return np.where(np.abs(denom) > 1e-6, (a - b) / denom, np.nan)


def compute_ndvi(cube, wl):
    """
    NDVI = (NIR_860 - Red_660) / (NIR_860 + Red_660)
    Range: -1 to +1
    Interpretation:
      < 0.1 : water, bare soil, rock, urban
      0.1–0.3 : sparse vegetation, dry crops, shrubs
      0.3–0.6 : moderate vegetation, grasslands, crops mid-season
      > 0.6 : dense healthy crops, forests
    DATE-SENSITIVE: May scene will naturally be higher than Oct scene.
    """
    nir = cube[band_at(wl, 860)]
    red = cube[band_at(wl, 660)]
    return safe_ratio(nir, red)


def compute_ndre(cube, wl):
    """
    NDRE = (NIR_780 - RedEdge_720) / (NIR_780 + RedEdge_720)
    Uses the red-edge (700–730 nm) — the steep slope where chlorophyll
    absorption transitions to NIR reflectance.
    More sensitive than NDVI at high biomass (NDVI saturates above ~0.7).
    Shift in red-edge position toward blue = stress indicator.
    DATE-SENSITIVE.
    """
    nir      = cube[band_at(wl, 780)]
    red_edge = cube[band_at(wl, 720)]
    return safe_ratio(nir, red_edge)


def compute_ndmi(cube, wl):
    """
    NDMI = (NIR_860 - SWIR_1640) / (NIR_860 + SWIR_1640)
    Leaf moisture / canopy water content.
    Higher NDMI = more water in leaves = active growing plant.
    Near 0 or negative = dry leaves, bare soil, stressed plants.
    Useful even on October scenes for soil moisture under the surface.
    MODERATELY date-sensitive.
    """
    nir  = cube[band_at(wl, 860)]
    swir = cube[band_at(wl, 1640)]
    return safe_ratio(nir, swir)


def compute_bsi(cube, wl):
    """
    BSI = ((SWIR_1600 + Red_660) - (NIR_830 + Blue_480)) /
          ((SWIR_1600 + Red_660) + (NIR_830 + Blue_480))
    Bare Soil Index. High = lots of exposed soil / low vegetation cover.
    Complement to NDVI — high BSI + low NDVI = degraded / overgrazed land.
    REASONABLY DATE-ROBUST (soil is soil regardless of season).
    """
    swir = cube[band_at(wl, 1600)]
    red  = cube[band_at(wl, 660)]
    nir  = cube[band_at(wl, 830)]
    blue = cube[band_at(wl, 480)]
    num  = (swir + red) - (nir + blue)
    den  = (swir + red) + (nir + blue)
    return np.where(np.abs(den) > 1e-6, num / den, np.nan)


def compute_clay_index(cube, wl):
    """
    Clay Index: depth of the OH-bond absorption feature at 2200 nm.
    Formula: R_2100 / R_2200 — ratio of shoulder to absorption centre.
    High value = strong absorption at 2200 = clay-rich soil (kaolinite,
    montmorillonite, illite).
    Clay soils swell when wet and crack when dry — higher agricultural
    and foundation risk.
    DATE-ROBUST: mineral composition doesn't change seasonally.
    """
    r2100 = cube[band_at(wl, 2100)]
    r2200 = cube[band_at(wl, 2200)]
    return np.where(r2200 > 1e-4, r2100 / r2200, np.nan)


# ─── LOAD ALL SCENES ─────────────────────────────────────────────────────────

print("=" * 65)
print("Loading and smoothing scenes...")
print("=" * 65)
all_data, wl_ref, good_ref, N_BANDS = load_all_scenes(smooth=True, verbose=True)

if not all_data:
    sys.exit("No scenes loaded.")

# ─── COMPUTE ALL INDICES ─────────────────────────────────────────────────────

INDEX_FUNCS = {
    "NDVI":  (compute_ndvi,       "Vegetation / Biomass",       "RdYlGn",  (-0.1, 0.8), True),
    "NDRE":  (compute_ndre,       "Red-edge Vegetation",        "RdYlGn",  (-0.1, 0.5), True),
    "NDMI":  (compute_ndmi,       "Moisture",                   "RdYlBu",  (-0.4, 0.3), True),
    "BSI":   (compute_bsi,        "Bare Soil (high=bad)",       "RdYlGn_r",(-0.3, 0.3), False),
    "Clay":  (compute_clay_index, "Clay Index (high=risk)",     "YlOrRd",  (0.8, 1.5),  False),
}

# {index_name: {site_key: 2D array}}
index_maps = {idx: {} for idx in INDEX_FUNCS}

print("\nComputing indices...")
for key, d in all_data.items():
    for idx_name, (fn, _, _, _, _) in INDEX_FUNCS.items():
        result = fn(d["cube"], d["wl"])
        result[~d["valid"]] = np.nan
        index_maps[idx_name][key] = result
        mean_val = np.nanmean(result)
        print(f"  {key:10s} {idx_name:6s}: mean={mean_val:+.3f}")

# ─── PLOT SPATIAL MAPS (one figure per index) ─────────────────────────────────

print("\nGenerating spatial index maps...")

for idx_name, (fn, description, cmap_name, vrange, date_sensitive) in INDEX_FUNCS.items():
    n = len(all_data)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 1:
        axes = [axes]

    vmin, vmax = vrange
    cmap = plt.get_cmap(cmap_name)
    cmap.set_bad(color='#888888')  # NaN / nodata → grey

    for ax, (key, d) in zip(axes, all_data.items()):
        idx_map = index_maps[idx_name][key]
        im = ax.imshow(idx_map, cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation='nearest')
        ax.set_title(d["label"], fontsize=9)

        mean_v = np.nanmean(idx_map)
        std_v  = np.nanstd(idx_map)
        ax.text(0.03, 0.06, f"μ={mean_v:.3f}  σ={std_v:.3f}",
                transform=ax.transAxes, fontsize=7.5, color='white',
                bbox=dict(boxstyle='round', fc='black', alpha=0.6))
        ax.axis('off')

    plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04, label=idx_name)

    date_warning = "  ⚠ DATE-SENSITIVE — compare with caution across scenes" if date_sensitive else "  ✓ Date-robust"
    fig.suptitle(f"{idx_name} — {description}\n{date_warning}",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    fname = f"idx_{idx_name.lower()}_maps.png"
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

# ─── SUMMARY COMPARISON TABLE ─────────────────────────────────────────────────

print("\n" + "=" * 65)
print("PER-SITE INDEX SUMMARY")
print("=" * 65)

sites = list(all_data.keys())
idx_names = list(INDEX_FUNCS.keys())

# Print table
header = f"{'Index':8s}  " + "  ".join(f"{k:>18s}" for k in sites)
print(header)
print("-" * len(header))

stats_table = {}  # {site: {index: (mean, std, cv)}}
for k in sites:
    stats_table[k] = {}

for idx_name in idx_names:
    row = f"{idx_name:8s}  "
    for key in sites:
        m  = np.nanmean(index_maps[idx_name][key])
        sd = np.nanstd(index_maps[idx_name][key])
        cv = sd / abs(m) if abs(m) > 1e-4 else np.nan
        stats_table[key][idx_name] = (m, sd, cv)
        row += f"  μ={m:+.3f} σ={sd:.3f}  "
    print(row)

# ─── BAR CHART COMPARISON ─────────────────────────────────────────────────────

print("\nGenerating comparison bar chart...")

fig, axes = plt.subplots(1, len(idx_names), figsize=(4 * len(idx_names), 5))

for ax, idx_name in zip(axes, idx_names):
    means  = [stats_table[k][idx_name][0] for k in sites]
    stds   = [stats_table[k][idx_name][1] for k in sites]
    colors = [COLORS[k] for k in sites]
    labels = [all_data[k]["label"].split('\n')[0] for k in sites]

    bars = ax.bar(labels, means, yerr=stds, capsize=4,
                  color=colors, alpha=0.85, edgecolor='black', linewidth=0.7)
    ax.set_title(idx_name, fontsize=10, fontweight='bold')
    ax.set_ylabel(INDEX_FUNCS[idx_name][1], fontsize=8)
    ax.tick_params(axis='x', labelsize=7)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(0, color='black', lw=0.5)

    _, _, _, date_sensitive, *_ = (*INDEX_FUNCS[idx_name],)
    date_sensitive = INDEX_FUNCS[idx_name][4]
    if date_sensitive:
        ax.set_xlabel("⚠ Date-sensitive", fontsize=7, color='red')

fig.suptitle("Per-site index means ± 1 std  |  Error bars = spatial heterogeneity within plot",
             fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "summary_bar_chart.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: summary_bar_chart.png")

# ─── COEFFICIENT OF VARIATION CHART (Uniformity) ──────────────────────────────

print("Generating uniformity (CV) chart...")

fig, ax = plt.subplots(figsize=(10, 4))

x = np.arange(len(idx_names))
width = 0.2
for i, key in enumerate(sites):
    cvs = [stats_table[key][idx][2] for idx in idx_names]
    label = all_data[key]["label"].replace('\n', ' ')
    ax.bar(x + i * width, cvs, width, label=label,
           color=COLORS[key], alpha=0.85, edgecolor='black', linewidth=0.5)

ax.set_xticks(x + width * (len(sites) - 1) / 2)
ax.set_xticklabels(idx_names)
ax.set_ylabel("Coefficient of Variation (σ/|μ|)")
ax.set_title("Spatial heterogeneity per site per index\n"
             "Lower CV = more uniform plot = more predictable / lower operational risk",
             fontsize=10)
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "uniformity_cv.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: uniformity_cv.png")

# ─── RADAR CHART (one per site) ───────────────────────────────────────────────

print("Generating radar charts...")

# Normalise indices to [0,1] for the radar (higher always = better for investment)
# For each index, we define: is higher better?
# NDVI: higher better. NDRE: higher better. NDMI: higher better.
# BSI: lower better (high bare soil = bad). Clay: lower better.
# Normalize within the 4-site range.

def norm01(vals):
    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-6:
        return [0.5] * len(vals)
    return [(v - mn) / (mx - mn) for v in vals]

radar_labels = ["NDVI\n(biomass)", "NDRE\n(red-edge)", "NDMI\n(moisture)",
                "Uniformity\n(1-CV)", "No Clay\n(1-Clay norm)"]
n_axes = len(radar_labels)
angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
angles += angles[:1]  # close the polygon

raw_scores = {}
for key in sites:
    ndvi_m  = stats_table[key]["NDVI"][0]
    ndre_m  = stats_table[key]["NDRE"][0]
    ndmi_m  = stats_table[key]["NDMI"][0]
    cv_avg  = np.mean([stats_table[key][i][2] for i in ["NDVI", "NDMI"] if not np.isnan(stats_table[key][i][2])])
    clay_m  = stats_table[key]["Clay"][0]
    raw_scores[key] = [ndvi_m, ndre_m, ndmi_m, 1 - cv_avg if not np.isnan(cv_avg) else 0.5, -clay_m]

# Normalise
n_metrics = len(radar_labels)
norm_scores = {}
for mi in range(n_metrics):
    vals = [raw_scores[k][mi] for k in sites]
    normed = norm01(vals)
    for ki, key in enumerate(sites):
        if key not in norm_scores:
            norm_scores[key] = []
        norm_scores[key].append(normed[ki])

fig = plt.figure(figsize=(12, 4))
for si, key in enumerate(sites):
    ax = fig.add_subplot(1, len(sites), si + 1, projection='polar')
    values = norm_scores[key] + [norm_scores[key][0]]  # close polygon
    ax.plot(angles, values, color=COLORS[key], linewidth=2)
    ax.fill(angles, values, color=COLORS[key], alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_title(all_data[key]["label"], fontsize=8, pad=12)
    ax.grid(True, alpha=0.3)

fig.suptitle("Per-site radar chart (normalised within 4-site range)\n"
             "Larger area = stronger profile across metrics", fontsize=10, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "radar_charts.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: radar_charts.png")

print("\n" + "=" * 65)
print("DONE — Script 02")
print("=" * 65)
print(f"  Output: {os.path.abspath(PLOTS_DIR)}")
print()
print("  ⚠ REMINDER ON DATE BIAS:")
print("  Arkadia2 (May 2024) will have genuinely higher NDVI/NDRE/NDMI")
print("  than the October/August scenes, not necessarily because the land")
print("  is better — because it was imaged during active growing season.")
print("  The Clay Index and BSI are your most trustworthy cross-date comparisons.")
print("  Anomaly detection (script 03) learns the within-population structure")
print("  and is less affected by this bias.")


from pipeline_utils import save_step
save_step("indices", {
    "plots": [
        {"id": f"idx_{n.lower()}", "path": f"plots/02_indices/idx_{n.lower()}_maps.png", "title": f"{n} Map"}
        for n in INDEX_FUNCS
    ] + [
        {"id": "summary_bar",  "path": "plots/02_indices/summary_bar_chart.png", "title": "Index Summary"},
        {"id": "uniformity",   "path": "plots/02_indices/uniformity_cv.png",     "title": "Uniformity (CV)"},
        {"id": "radar",        "path": "plots/02_indices/radar_charts.png",       "title": "Radar Charts"},
    ],
    "per_site": {
        key: {
            idx: {
                "mean": round(float(stats_table[key][idx][0]), 4),
                "std":  round(float(stats_table[key][idx][1]), 4),
                "cv":   round(float(stats_table[key][idx][2]), 4) if not np.isnan(stats_table[key][idx][2]) else None,
                "date_sensitive": INDEX_FUNCS[idx][4]
            }
            for idx in INDEX_FUNCS
        }
        for key in sites
    }
})