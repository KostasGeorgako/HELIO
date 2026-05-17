"""
spectral_common.py — Shared definitions for the Makethlon hyperspectral pipeline.

Imported by all four scripts. Contains:
  - Data paths and constants
  - Scene definitions (challenge + extra)
  - EnMAP loader, band masking, smoothing
  - SpectralAE model class
  - extract_valid_spectra helper
"""

import os
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from scipy.signal import savgol_filter

import torch
import torch.nn as nn

# ─── PATHS ───────────────────────────────────────────────────────────────────

DATA_ROOT  = "../data/images_makeathlon/enmap"
PLOTS_BASE = "../data/plots"
MODEL_PATH = "../data/spectral_ae.pt"

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

NODATA     = -9999.0
SCALE      = 10.0        # raw values are reflectance × 10
RANDOM_SEED = 42
VAL_FRAC    = 0.20

# ─── SCENE REGISTRY ──────────────────────────────────────────────────────────

# Challenge images: scored, anomaly-mapped, displayed in all scripts.
# key must equal the subdirectory name under DATA_ROOT.
SCENES = {
    "arkadia":  ("arkadia_20241024_mosaic",  "Arkadia\n(Oct 2024)"),
    "arkadia2": ("arkadia2_20240531_mosaic", "Arkadia 2\n(May 2024)"),
    "magnisia": ("magnisia_20241024_mosaic", "Magnisia\n(Oct 2024)"),
    "veroia":   ("veroia_20250821_mosaic",   "Veroia\n(Aug 2025)"),
}

# Extra temporal images: training pool augmentation only, never scored.
# tuple: (folder_name, parent_site_key)
SCENES_EXTRA = {
    "arkadia_2025":   ("arkadia_20250721_mosaic",  "arkadia"),
    "arkadia2_2025":  ("arkadia2_20250721_mosaic", "arkadia2"),
    "magnisia_2024b": ("magnisia_20240531_mosaic", "magnisia"),
    "magnisia_2025":  ("magnisia_20250721_mosaic", "magnisia"),
    "veroia_2024a":   ("veroia_20240716_mosaic",   "veroia"),
    "veroia_2024b":   ("veroia_20240812_mosaic",   "veroia"),
}

# Per-site plot colours, consistent across all scripts
COLORS = {
    "arkadia":  "#e06c00",
    "arkadia2": "#2e8b22",
    "magnisia": "#1a6bb5",
    "veroia":   "#9b2db5",
}

# ─── BAND MASKING ─────────────────────────────────────────────────────────────

def good_band_mask(wl):
    """
    Returns boolean array (len = n_bands), True = keep this band.

    Drops three noisy regions:
      850–1000 nm  : VNIR/SWIR detector stitch artefacts
      1340–1460 nm : atmospheric water vapour absorption
      1790–1960 nm : atmospheric water vapour absorption
    """
    g = np.ones(len(wl), dtype=bool)
    g[(wl >= 850)  & (wl <= 1000)] = False
    g[(wl >= 1340) & (wl <= 1460)] = False
    g[(wl >= 1790) & (wl <= 1960)] = False
    return g

# ─── SCENE LOADER ────────────────────────────────────────────────────────────

def load_scene(folder_path):
    """
    Load one EnMAP scene folder.

    Parameters
    ----------
    folder_path : str
        Absolute or relative path to the mosaic folder containing
        SPECTRAL_IMAGE.TIF, METADATA.XML, and QL_QUALITY_*.TIF files.

    Returns
    -------
    cube       : (bands, H, W) float32  — nodata → NaN, divided by SCALE
    wl         : (bands,)  float64      — wavelength centres in nm
    valid      : (H, W)    bool         — True where pixel is usable
    meta       : dict                   — startTime, cloudCover, nadirAngle
    """
    tif = os.path.join(folder_path, "SPECTRAL_IMAGE.TIF")
    xml = os.path.join(folder_path, "METADATA.XML")

    with rasterio.open(tif) as src:
        cube = src.read().astype(np.float32)
    cube = np.where(cube == NODATA, np.nan, cube / SCALE)

    tree = ET.parse(xml)
    root = tree.getroot()
    wl = np.array([
        float(e.text) for e in root.iter()
        if e.tag.split('}')[-1] == 'wavelengthCenterOfBand' and e.text
    ])

    def get_tag(tag):
        for e in root.iter():
            if e.tag.split('}')[-1] == tag and e.text:
                return e.text.strip()
        return "N/A"

    meta = {
        "startTime":  get_tag("startTime"),
        "cloudCover": get_tag("cloudCover"),
        "nadirAngle": get_tag("acrossOffNadirAngle"),
    }

    # Quality masks: 0 = clean pixel
    valid = np.ones(cube.shape[1:], dtype=bool)
    for mf in ["QL_QUALITY_CLOUD.TIF", "QL_QUALITY_CLOUDSHADOW.TIF", "QL_QUALITY_HAZE.TIF"]:
        mp = os.path.join(folder_path, mf)
        if os.path.exists(mp):
            with rasterio.open(mp) as src:
                valid &= (src.read(1) == 0)

    # Require ≥70% of good bands to be finite per pixel
    good = good_band_mask(wl)
    finite_per_px = np.sum(np.isfinite(cube[good]), axis=0)
    valid &= (finite_per_px >= int(0.70 * good.sum()))
    cube[:, ~valid] = np.nan

    return cube, wl, valid, meta

# ─── SPECTRAL SMOOTHING ───────────────────────────────────────────────────────

def smooth_cube(cube, good):
    """
    Apply Savitzky-Golay smoothing along the spectral axis for each pixel.

    Parameters
    ----------
    cube : (bands, H, W) float32
    good : (bands,) bool — which bands to smooth (others left unchanged)

    Returns smoothed cube, same shape.
    Window=9, polyorder=2 — standard for EnMAP spectral resolution.
    """
    cube_s = cube.copy()
    _, H, W = cube.shape
    for r in range(H):
        for c in range(W):
            spec = cube[:, r, c]
            if np.any(np.isnan(spec)):
                continue
            if good.sum() >= 9:
                spec[good] = savgol_filter(spec[good], 9, 2)
            cube_s[:, r, c] = spec
    return cube_s

# ─── PIXEL EXTRACTION ────────────────────────────────────────────────────────

def extract_valid_spectra(cube, wl, valid, good):
    """
    Collect all valid pixels from a scene as a flat array.

    Returns
    -------
    spectra : (N, n_good_bands) float32
    coords  : list of (row, col) tuples, same order as spectra rows
    """
    rows, cols = np.where(valid)
    spectra, coords = [], []
    for r, c in zip(rows, cols):
        spec = cube[good, r, c]
        if not np.any(np.isnan(spec)):
            spectra.append(spec)
            coords.append((r, c))
    if spectra:
        return np.array(spectra, dtype=np.float32), coords
    return np.zeros((0, int(good.sum())), dtype=np.float32), []

# ─── SCENE PATH HELPER ───────────────────────────────────────────────────────

def scene_path(site_key, folder_name):
    """Build the full path to a mosaic folder: DATA_ROOT/site_key/folder_name."""
    return os.path.join(DATA_ROOT, site_key, folder_name)

# ─── LOAD ALL CHALLENGE SCENES ───────────────────────────────────────────────

def load_all_scenes(smooth=True, verbose=True):
    """
    Load and optionally smooth all four challenge scenes.

    Returns
    -------
    all_data : dict  key → {cube, wl, valid, meta, label, good}
    wl_ref   : reference wavelength array (from first loaded scene)
    good_ref : reference good-band mask
    N_BANDS  : int, number of usable bands
    """
    all_data = {}
    for key, (fname, label) in SCENES.items():
        fpath = scene_path(key, fname)
        if not os.path.isdir(fpath):
            if verbose:
                print(f"  [{key}] SKIPPED — not found: {fpath}")
            continue
        cube, wl, valid, meta = load_scene(fpath)
        good = good_band_mask(wl)
        if smooth:
            cube = smooth_cube(cube, good)
        all_data[key] = {
            "cube": cube, "wl": wl, "valid": valid,
            "meta": meta, "label": label, "good": good,
        }
        if verbose:
            print(f"  [{key}]  {valid.sum()} valid px  |  {good.sum()} usable bands")

    if not all_data:
        raise RuntimeError(f"No scenes loaded. Check DATA_ROOT: {DATA_ROOT}")

    first = next(iter(all_data.values()))
    wl_ref  = first["wl"]
    good_ref = good_band_mask(wl_ref)
    N_BANDS  = int(good_ref.sum())
    return all_data, wl_ref, good_ref, N_BANDS

# ─── AUTOENCODER ─────────────────────────────────────────────────────────────

class SpectralAE(nn.Module):
    """
    Pixel-wise spectral autoencoder for anomaly detection.

    Architecture: N_BANDS → 96 → 48 → 24 → 48 → 96 → N_BANDS
    Bottleneck (24D) forces the network to learn a compact representation
    of normal Greek agricultural spectra. Pixels that don't fit the learned
    manifold produce high reconstruction error — those are the anomalies.

    Design choices:
      LeakyReLU   : avoids dying ReLU in narrow bottleneck layer
      BatchNorm   : stabilises training on small datasets
      Dropout     : regularises encoder (not decoder — decoder must reconstruct faithfully)
      No encoder→decoder skip connections : would bypass the bottleneck and
                                            defeat the anomaly detection mechanism
    """
    def __init__(self, n_bands, bottleneck=24, dropout=0.15):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_bands, 96), nn.BatchNorm1d(96), nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(96,      48), nn.BatchNorm1d(48), nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(48, bottleneck),                  nn.LeakyReLU(0.1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 48), nn.BatchNorm1d(48), nn.LeakyReLU(0.1),
            nn.Linear(48,         96), nn.BatchNorm1d(96), nn.LeakyReLU(0.1),
            nn.Linear(96,    n_bands),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

    def reconstruction_error(self, x):
        """Per-pixel MSE between input and reconstruction. No gradient."""
        with torch.no_grad():
            return ((self.forward(x) - x) ** 2).mean(dim=1).cpu().numpy()

# ─── MODEL PERSISTENCE ───────────────────────────────────────────────────────

def load_model_checkpoint(device):
    """
    Load the trained SpectralAE and all associated metadata from MODEL_PATH.

    Returns
    -------
    model        : SpectralAE, eval mode
    scaler       : fitted StandardScaler (reconstructed from checkpoint arrays)
    ckpt         : full checkpoint dict (train_losses, val_losses, indices, etc.)
    """
    from sklearn.preprocessing import StandardScaler

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}\n"
            f"Run 03_anomaly_ml.py first."
        )

    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)  # ← fix

    model = SpectralAE(ckpt["n_bands"], bottleneck=ckpt["bottleneck"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    scaler = StandardScaler()
    scaler.mean_           = ckpt["scaler_mean"]
    scaler.scale_          = ckpt["scaler_std"]
    scaler.var_            = ckpt["scaler_std"] ** 2
    scaler.n_features_in_  = ckpt["n_bands"]

    return model, scaler, ckpt

# ─── BAND UTILITY ────────────────────────────────────────────────────────────

def band_at(wl, nm):
    """Index of the band closest to target wavelength nm."""
    return int(np.argmin(np.abs(wl - nm)))


# ─── SPECTRAL INDEX LIBRARY ───────────────────────────────────────────────────

def compute_ndvi(cube, wl):
    nir = cube[band_at(wl, 860)]; red = cube[band_at(wl, 660)]
    d = nir + red
    return np.where(d > 1e-6, (nir - red) / d, np.nan)

def compute_ndre(cube, wl):
    nir = cube[band_at(wl, 780)]; re = cube[band_at(wl, 720)]
    d = nir + re
    return np.where(d > 1e-6, (nir - re) / d, np.nan)

def compute_ndmi(cube, wl):
    nir = cube[band_at(wl, 860)]; swir = cube[band_at(wl, 1640)]
    d = nir + swir
    return np.where(d > 1e-6, (nir - swir) / d, np.nan)

def compute_evi(cube, wl):
    nir  = cube[band_at(wl, 860)]
    red  = cube[band_at(wl, 660)]
    blue = cube[band_at(wl, 480)]
    d = nir + 6*red - 7.5*blue + 1
    return np.where(np.abs(d) > 1e-6, 2.5 * (nir - red) / d, np.nan)

def compute_bsi(cube, wl):
    swir = cube[band_at(wl, 1600)]; red  = cube[band_at(wl, 660)]
    nir  = cube[band_at(wl, 830)];  blue = cube[band_at(wl, 480)]
    num = (swir + red) - (nir + blue)
    den = (swir + red) + (nir + blue)
    return np.where(np.abs(den) > 1e-6, num / den, np.nan)

def compute_clay_index(cube, wl):
    r2100 = cube[band_at(wl, 2100)]; r2200 = cube[band_at(wl, 2200)]
    return np.where(r2200 > 1e-4, r2100 / r2200, np.nan)

def compute_iron_oxide(cube, wl):
    """R_700 / R_500 — elevated = iron-rich soil (hematite, goethite)."""
    return np.where(
        cube[band_at(wl, 500)] > 1e-4,
        cube[band_at(wl, 700)] / cube[band_at(wl, 500)],
        np.nan
    )

def compute_carbonate_index(cube, wl):
    """R_2340 / R_2300 — carbonate/limestone content."""
    r2300 = cube[band_at(wl, 2300)]
    return np.where(
        r2300 > 1e-4,
        cube[band_at(wl, 2340)] / r2300,
        np.nan
    )

def compute_salinity_index(cube, wl):
    """R_1600 / R_820 — salt-affected soils show elevated SWIR/NIR ratio."""
    nir = cube[band_at(wl, 820)]
    return np.where(
        nir > 1e-4,
        cube[band_at(wl, 1600)] / nir,
        np.nan
    )

def compute_all_indices(cube, wl, valid):
    """
    Compute all indices for one scene. Returns dict of {name: 2D map}.
    Invalid pixels set to NaN.
    """
    funcs = {
        "ndvi":       compute_ndvi,
        "ndre":       compute_ndre,
        "ndmi":       compute_ndmi,
        "evi":        compute_evi,
        "bsi":        compute_bsi,
        "clay":       compute_clay_index,
        "iron_oxide": compute_iron_oxide,
        "carbonate":  compute_carbonate_index,
        "salinity":   compute_salinity_index,
    }
    result = {}
    for name, fn in funcs.items():
        m = fn(cube, wl).astype(np.float32)
        m[~valid] = np.nan
        result[name] = m
    return result

def summarise_indices(index_maps, valid):
    """
    Per-site summary stats for all indices.
    Returns dict of {name: {mean, std, cv}}.
    """
    summary = {}
    for name, m in index_maps.items():
        vals = m[valid & np.isfinite(m)]
        if len(vals) == 0:
            summary[name] = {"mean": np.nan, "std": np.nan, "cv": np.nan}
            continue
        mean = float(np.nanmean(vals))
        std  = float(np.nanstd(vals))
        cv   = std / (abs(mean) + 1e-6)
        summary[name] = {"mean": round(mean, 4), "std": round(std, 4), "cv": round(cv, 4)}
    return summary