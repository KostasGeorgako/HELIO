"""
00_diagnose_masks.py — Figure out what the QL files actually contain
Run this first. It prints the exact values inside every QL file for one site.
"""
import os
import numpy as np
import rasterio

DATA_ROOT = "../data/images_makeathlon"
SITE      = "arkadia_20241024_mosaic"          # change if this folder is missing
folder    = os.path.join(DATA_ROOT, SITE)

QL_FILES = [
    "QL_QUALITY_CLOUD.TIF",
    "QL_QUALITY_CLOUDSHADOW.TIF",
    "QL_QUALITY_HAZE.TIF",
    "QL_QUALITY_SNOW.TIF",
    "QL_QUALITY.TIF",
    "QL_QUALITY_TESTFLAGS.TIF",
    "SPECTRAL_IMAGE.TIF",          # also check raw cube values
]

print(f"Inspecting: {folder}\n")

for fname in QL_FILES:
    path = os.path.join(folder, fname)
    if not os.path.exists(path):
        print(f"  {fname:40s}  NOT FOUND")
        continue

    with rasterio.open(path) as src:
        bands   = src.count
        dtype   = src.dtypes[0]
        nodata  = src.nodata
        data    = src.read()          # all bands, shape (bands, H, W)

    print(f"  {fname}")
    print(f"    bands={bands}  dtype={dtype}  nodata_header={nodata}")
    print(f"    shape={data.shape}")

    if fname == "SPECTRAL_IMAGE.TIF":
        # Just show band 1 stats to confirm reflectance values
        b1 = data[0].astype(float)
        valid = b1[b1 != -9999.0]
        print(f"    band-1 range: {b1.min():.2f} – {b1.max():.2f}")
        if len(valid) > 0:
            print(f"    band-1 valid (≠-9999) range: {valid.min():.4f} – {valid.max():.4f}")
            print(f"    band-1 valid pixels: {len(valid)} / {b1.size}")
        else:
            print(f"    ALL pixels are -9999 in band 1 ← cube itself is nodata!")
    else:
        # For QL files: show unique values so we know the encoding
        for bi in range(bands):
            layer = data[bi]
            unique_vals = np.unique(layer)
            print(f"    band {bi+1}: unique values = {unique_vals}  "
                  f"min={layer.min()}  max={layer.max()}")
    print()