import rasterio
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import os

NODATA    = -9999.0
SCALE     = 10.0        # HuggingFace tool outputs reflectance × 10
plots_dir = "../data/plots"
os.makedirs(plots_dir, exist_ok=True)

site = "arkadia_20241024_mosaic"
base = f"../data/images_makeathlon/{site}"

# ── Load + scale ──────────────────────────────────────────────────
with rasterio.open(f"{base}/SPECTRAL_IMAGE.TIF") as src:
    cube = src.read()

cube_masked = np.where(cube == NODATA, np.nan, cube / SCALE)
print(f"Reflectance range (valid): {np.nanmin(cube_masked):.4f} – {np.nanmax(cube_masked):.4f}")
# Expect roughly 0.02 – 0.45

# ── Wavelengths ───────────────────────────────────────────────────
tree = ET.parse(f"{base}/METADATA.XML")
root = tree.getroot()
wl = np.array([
    float(e.text) for e in root.iter()
    if e.tag.split('}')[-1] == 'wavelengthCenterOfBand' and e.text
])

def band_at(nm):
    return int(np.argmin(np.abs(wl - nm)))

# ── RGB — mask nodata properly ────────────────────────────────────
r, g, b = band_at(660), band_at(560), band_at(490)

rgb = np.stack([cube_masked[r], cube_masked[g], cube_masked[b]], axis=-1)

# Valid pixel mask: True where ALL three bands are finite
valid = np.all(np.isfinite(rgb), axis=-1)

lo  = np.nanpercentile(rgb[valid], 2)
hi  = np.nanpercentile(rgb[valid], 98)
rgb_disp = np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1)

# Nodata → grey (0.5) so it's visually distinct from real dark pixels
rgb_disp[~valid] = 0.5

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(rgb_disp, interpolation='nearest')   # no smoothing — pixels are real
ax.set_title(f"{site}\nRGB (660/560/490 nm) | {valid.mean()*100:.0f}% valid pixels")
ax.axis('off')
# Annotate pixel count so it's obvious this is expected
ax.text(0.02, 0.02, f"{cube.shape[1]}×{cube.shape[2]} px @ 30m GSD",
        transform=ax.transAxes, fontsize=7, color='white',
        bbox=dict(boxstyle='round', fc='black', alpha=0.5))
plt.tight_layout()
plt.savefig(f"{plots_dir}/{site}_rgb.png", dpi=150)
plt.close()

# ── Mean spectrum ─────────────────────────────────────────────────
mean_spectrum = np.nanmean(cube_masked, axis=(1, 2))

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(wl, mean_spectrum, lw=1.2)
ax.axvspan(1340, 1460, color='red',    alpha=0.15, label='water absorption — drop')
ax.axvspan(1790, 1960, color='orange', alpha=0.15, label='water absorption — drop')
ax.set_ylim(0, 0.6)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Reflectance")
ax.set_title(f"Mean spectrum — {site}  (Oct 24, dry/senescent land)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{plots_dir}/{site}_mean_spectrum.png", dpi=150)
plt.close()

print("Done.")