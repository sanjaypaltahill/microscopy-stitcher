#!/usr/bin/env python
"""Visual QC for MIRACL's CLARITY segmentation. Produces pictures, not numbers.

Usage:
    python qc_overlay.py RAW_DIR SEG_BIN_TIF OUT_DIR

    RAW_DIR      folder of per-slice signal-channel tifs (the input to seg clar)
    SEG_BIN_TIF  segmentation_<type>/seg_bin_<type>.tif written by `miracl seg clar`
    OUT_DIR      where to write the PNGs

Writes:
    overlay_singleslice.png  three Z-slices, detected cells in red
    overlay_slab.png         the same slices as 7-plane max-projections, which is
                             a fairer recall check because each cell shows once

Also prints intensity stats for the raw slice and for the macro's `norm.tif`
intermediate if it is present. If norm.tif is nearly all 0 or all 255, the Fiji
macro's 16-bit -> 8-bit conversion clipped the data: switch SEG_TYPE to cfos2,
which rescales before converting.
"""
import sys, glob, os
import numpy as np, tifffile
from scipy import ndimage as ndi
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW_DIR, SEG_BIN, OUTDIR = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUTDIR, exist_ok=True)

raws = sorted(glob.glob(os.path.join(RAW_DIR, "*.tif")))
if not raws:
    sys.exit(f"no tifs in {RAW_DIR}")

seg_tf = tifffile.TiffFile(SEG_BIN)
seg_pages = seg_tf.pages
if len(seg_pages) != len(raws):
    print(f"WARNING: {len(seg_pages)} segmentation planes vs {len(raws)} raw slices")

read_seg = lambda z: np.asarray(seg_pages[z].asarray()) > 0

# --- sanity numbers ---
mid = len(raws) // 2
r = tifffile.imread(raws[mid])
s = read_seg(min(mid, len(seg_pages) - 1))
print("raw slice:", r.dtype, "min", r.min(), "max", r.max(),
      "p99.5", float(np.percentile(r, 99.5)))
print("seg slice: cell pixels", int(s.sum()), f"({100 * s.mean():.4f}% of plane)")

norm = os.path.join(os.path.dirname(SEG_BIN), "norm.tif")
if os.path.isfile(norm):
    with tifffile.TiffFile(norm) as tf:
        n = np.asarray(tf.pages[min(mid, len(tf.pages) - 1)].asarray())
    print("norm.tif slice:", n.dtype, "min", n.min(), "max", n.max(),
          "mean", float(n.mean()),
          "| saturated(255)", f"{100 * np.mean(n >= 255):.2f}%",
          "| zero", f"{100 * np.mean(n == 0):.2f}%")


def overlay(centers, W, outpng, crop=700):
    fig, ax = plt.subplots(len(centers), 2, figsize=(12, 6 * len(centers)))
    ax = np.atleast_2d(ax)
    for i, zc in enumerate(centers):
        idx = range(max(0, zc - W), min(len(raws), zc + W + 1))
        rr = np.max([tifffile.imread(raws[z]).astype(np.float32) for z in idx], axis=0)
        ss = np.any([read_seg(z) for z in idx if z < len(seg_pages)], axis=0)
        dens = ndi.uniform_filter(ss.astype(np.float32), size=200)   # densest window
        cy, cx = np.unravel_index(np.argmax(dens), dens.shape)
        y0 = max(0, min(cy - crop // 2, rr.shape[0] - crop))
        x0 = max(0, min(cx - crop // 2, rr.shape[1] - crop))
        rc = rr[y0:y0 + crop, x0:x0 + crop]
        sc = ndi.binary_dilation(ss[y0:y0 + crop, x0:x0 + crop], iterations=2)
        base = np.clip(rc / max(np.percentile(rc, 99.5), 1), 0, 1)
        rgb = np.stack([base] * 3, -1); rgb[sc] = [1, 0, 0]
        lab = f"z={zc}" + (f"+-{W} max-proj" if W else "")
        ax[i, 0].imshow(base, cmap="gray"); ax[i, 0].set_title(f"{lab} raw signal"); ax[i, 0].axis("off")
        ax[i, 1].imshow(rgb); ax[i, 1].set_title(f"{lab} detected cells (red)"); ax[i, 1].axis("off")
    plt.tight_layout(); plt.savefig(outpng, dpi=130); plt.close(); print("wrote", outpng)


n = len(raws)
zc = [n // 4, n // 2, 3 * n // 4]
overlay(zc, 0, f"{OUTDIR}/overlay_singleslice.png")
overlay(zc, 3, f"{OUTDIR}/overlay_slab.png")
seg_tf.close()
