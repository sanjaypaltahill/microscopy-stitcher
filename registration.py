## Sensitivity analysis of parameters

"""
<<<<<<< Updated upstream
Image Registration & Stitching — Monje Lab
================================================
Uses skimage.registration.phase_cross_correlation to estimate the best
(dy, dx) shift between overlapping tile edges, then applies that shift
as a pure-translation AffineTransform via skimage.transform.warp.

After registration, tiles are stitched using raised-cosine (sinusoidal)
blending. Zero-padded rows/columns (added to reconcile dimension mismatches
from non-zero dy between rows) are excluded from the sinusoidal calculation.

--mode flag:
  test   — Processes first two tiles (columns 0 and 1 of a reference row,
            Z slice 0). Saves red/green/yellow overlay PNGs (naive vs corrected)
            so you can inspect alignment quality. Prints a movement report.

  pair_h — Stitch just two tiles horizontally (col --col_idx and col_idx+1,
            row --row_idx, z --z_slice).
  pair_v — Stitch just two tiles vertically   (row --row_idx and row_idx+1,
            col --col_idx, z --z_slice).
  row    — Stitch a single full row (row --row_idx, all cols, z --z_slice).
  col    — Stitch a single full column (col --col_idx, all rows, z --z_slice).
  real   — Full stitch: all tiles, all channels.
             With --z_slice N : only that Z slice.
             Without --z_slice : all Z slices (full volume).

New flags:
  --row_idx    Which row to use in pair_h / row / test modes  (default: 0)
  --col_idx    Which col to use in pair_v / col / test modes  (default: 0)
  --z_slice    Pin to a specific Z index for any mode.
               In 'real' mode: omit to stitch every Z (full volume),
               or pass a value to stitch only that single slice.

Dependencies:
    pip install numpy tifffile scipy scikit-image Pillow matplotlib
=======
registration.py — Monje Lab Stitcher
======================================
Phase-cross-correlation (PCC) shift estimation with continuous background
subtraction (no binary masks) and a tile-fraction shift constraint.

Design decisions
-----------------
* PCC on overlap strips (not full tiles):
  For a horizontal join, PCC operates on the right-edge strip of tile A and
  the left-edge strip of tile B, each overlap_px wide.  For a vertical join,
  the bottom strip of A and top strip of B, each overlap_px tall.

  Running PCC on the full tile is tempting but wrong: the dominant phase peak
  would correspond to the strongest repeating spatial frequency in the tissue
  (e.g. cell bodies at ~900 px spacing), not the seam offset.  Restricting to
  the overlap strips keeps the correlation space small and physically
  meaningful.

* Background subtraction (not masking):
  Before PCC, the per-strip percentile baseline is subtracted and negative
  values are clipped to zero.  This suppresses camera offset and
  autofluorescence continuously — no Otsu, no binary mask, no foreground
  fg-overlap gate.

* Shift constraint with peak-space fallback:
  After PCC, the shift is accepted only if |dy| ≤ max_shift_frac × strip_h
  and |dx| ≤ max_shift_frac × strip_w (and both ≤ max_shift).  If the
  primary result fails, the phase-correlation surface is re-examined at
  progressively coarser upsample factors (different peaks may dominate), then
  via a downsampled coarse PCC.  If nothing passes the constraint, (0, 0) is
  returned.

Public API
----------
subtract_background(img, percentile=5)

estimate_shift_horizontal(img_a, img_b, overlap_px, ..., max_shift_frac=0.5)
    Returns (dy, dx, overlap_px).

estimate_shift_vertical(img_a, img_b, overlap_px, ..., max_shift_frac=0.5)
    Returns (dy, dx, overlap_px).

compute_mse(img_a, img_b)
compute_mutual_information(img_a, img_b, bins=64)
apply_shift_skimage(img, dy, dx, output_shape)
>>>>>>> Stashed changes
"""

import os
import sys
import re
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
<<<<<<< Updated upstream
import matplotlib.patches as mpatches

try:
    import tifffile
except ImportError:
    sys.exit("Missing: pip install tifffile")

try:
    from skimage.registration import phase_cross_correlation
except ImportError:
    sys.exit("Missing: pip install scikit-image>=0.19")

from skimage.transform import AffineTransform, warp


# ============================================================
#  FILENAME PARSING
# ============================================================

def parse_filename(fname):
    """
    Expected pattern:  <prefix>[ROW x COL]_C<CH>_z<Z>.ome.tif
    Returns (row, col, channel, z, prefix) or None if no match.
    """
    m = re.match(r"^(.*?)\[(\d+) x (\d+)\]_C(\d+)_z(\d+)\.ome\.tif$", fname)
    if not m:
        return None
    prefix = m.group(1).rstrip("_ ") or "registered"
    row, col, ch, z = map(int, m.groups()[1:])
    return row, col, ch, z, prefix
=======
import os

from skimage.registration import phase_cross_correlation
from skimage.transform import AffineTransform, warp, rescale
from io_utils import load_tile


# ─────────────────────────────────────────────
#  BACKGROUND SUBTRACTION
# ─────────────────────────────────────────────

def subtract_background(img: np.ndarray, percentile: float = 5) -> np.ndarray:
    """
    Subtract the *percentile*-th percentile pixel value from the array and
    clip negatives to zero.  Returns float32.
    """
    baseline = float(np.percentile(img, percentile))
    return np.clip(img - baseline, 0.0, None).astype(np.float32)


# ─────────────────────────────────────────────
#  INTERNAL PCC WRAPPER
# ─────────────────────────────────────────────

def _pcc(ref: np.ndarray, moving: np.ndarray, upsample: int):
    """
    phase_cross_correlation(normalization=None).
    Crops to common shape if needed.
    Returns (dy, dx, error).
    """
    if ref.shape != moving.shape:
        rows = min(ref.shape[0], moving.shape[0])
        cols = min(ref.shape[1], moving.shape[1])
        ref    = ref[:rows, :cols]
        moving = moving[:rows, :cols]
>>>>>>> Stashed changes


# ============================================================
#  I/O HELPERS
# ============================================================

def load_tile(path):
    """Load a tile TIF and return it as a 2-D float32 array."""
    img = tifffile.imread(path).astype(np.float32)
    while img.ndim > 2:
        img = img[0]
    return img


def save_tiff(img, path):
    """Normalise to uint16 and write a TIFF."""
    vmin, vmax = img.min(), img.max()
    if vmax > vmin:
        norm = ((img - vmin) / (vmax - vmin) * 65535).astype(np.uint16)
    else:
        norm = np.zeros_like(img, dtype=np.uint16)
    tifffile.imwrite(path, norm, photometric="minisblack")
    print(f"    Saved TIFF: {path}  ({norm.shape[1]}x{norm.shape[0]} px)")


# ============================================================
#  SHIFT ESTIMATION (CORRECTED)
# ============================================================

def estimate_shift_horizontal(img_a, img_b, overlap_px, search_margin=100, upsample=10):
    """
    For horizontal (left-right) tile pairs.
    Returns (dy, dx) indicating how much to shift tile B to align with tile A.
    """
    h_a, w_a = img_a.shape
    h_b, w_b = img_b.shape
    
    # Reference: rightmost overlap_px columns from tile A
    ref_strip = img_a[:, -overlap_px:]
    
    # Moving: leftmost overlap_px columns from tile B (what we want to align)
    # We search a bit wider to allow for misalignment
    search_width = min(overlap_px + search_margin, w_b)
    moving_strip = img_b[:, :search_width]
    
    # Match dimensions
    min_h = min(ref_strip.shape[0], moving_strip.shape[0])
    min_w = min(ref_strip.shape[1], moving_strip.shape[1])
    ref_strip = ref_strip[:min_h, :min_w]
    moving_strip = moving_strip[:min_h, :min_w]
    
    if ref_strip.std() < 1e-3 or moving_strip.std() < 1e-3:
        print("    Warning: Blank strip -- using (dy=0, dx=0).")
        return 0.0, 0.0
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # reference_image = ref_strip (from tile A)
        # moving_image = moving_strip (from tile B)
        # Returns: shift to apply to moving_image to align with reference_image
        shift, error, _ = phase_cross_correlation(
            ref_strip,      # reference (fixed)
            moving_strip,   # moving (what we're aligning)
            upsample_factor=upsample,
            normalization="phase",
        )
    
    dy = float(shift[0])
    dx = float(shift[1])
    
    print(f"    PCC shift: dy={dy:+.2f}, dx={dx:+.2f}, error={error:.4f}")
    
    return dy, dx

def estimate_shift_vertical(img_a, img_b, overlap_px, search_margin=100, upsample=10):
    """
    For vertical (top-bottom) tile pairs.
    Returns (dy, dx) indicating how much to shift tile B to align with tile A.
    """
    h_a, w_a = img_a.shape
    h_b, w_b = img_b.shape
    
    # Reference: bottom overlap_px rows from tile A
    ref_strip = img_a[-overlap_px:, :]
    
    # Moving: top rows from tile B (search a bit taller)
    search_height = min(overlap_px + search_margin, h_b)
    moving_strip = img_b[:search_height, :]
    
    # Match dimensions
    min_h = min(ref_strip.shape[0], moving_strip.shape[0])
    min_w = min(ref_strip.shape[1], moving_strip.shape[1])
    ref_strip = ref_strip[:min_h, :min_w]
    moving_strip = moving_strip[:min_h, :min_w]
    
    if ref_strip.std() < 1e-3 or moving_strip.std() < 1e-3:
        print("    Warning: Blank strip -- using (dy=0, dx=0).")
        return 0.0, 0.0
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shift, error, _ = phase_cross_correlation(
            ref_strip,      # reference (fixed) 
            moving_strip,   # moving (what we're aligning)
            upsample_factor=upsample,
            normalization="phase",
        )
<<<<<<< Updated upstream
    
    dy = float(shift[0])
    dx = float(shift[1])
    
    print(f"    PCC shift: dy={dy:+.2f}, dx={dx:+.2f}, error={error:.4f}")
    
    return dy, dx


# ============================================================
#  WARP (skimage AffineTransform)
# ============================================================

def apply_shift_skimage(img, dy, dx, output_shape):
    """
    Translate img by (dy, dx) using skimage's AffineTransform + warp.
    Pure translation -- no rotation, no shear.
    skimage convention: translation=(x, y) = (col, row).
    """
=======
    return float(shift[0]), float(shift[1]), float(error)


# ─────────────────────────────────────────────
#  SHIFT CONSTRAINT + FALLBACK
# ─────────────────────────────────────────────

def _pcc_with_shift_limit(ref: np.ndarray, moving: np.ndarray,
                          max_dy: float, max_dx: float,
                          upsample: int,
                          label: str = "") -> tuple:
    """
    Run PCC and return the first result whose shift is within bounds.

    Fallback sequence
    -----------------
    1. PCC at requested upsample.
    2. PCC at upsample//2, upsample//4, 1  (different peak may dominate).
    3. Coarse downsampled (1/8) PCC, result scaled back up.
    4. If nothing passes → warn and return (0.0, 0.0, 1.0).

    Note: changing upsample does NOT change which phase peak is the global
    maximum — it only changes sub-pixel resolution.  Steps 1-3 therefore all
    find the same integer-pixel peak; the real purpose of the fallback chain
    is step 3 (coarse scale), which physically cannot produce a large shift
    because the image is small.  Steps 1-2 are kept for the sub-pixel
    accuracy ladder in the normal in-range case.

    Returns
    -------
    dy, dx, error : float, float, float
    """
    for up in [upsample, max(1, upsample // 2), max(1, upsample // 4), 1]:
        dy, dx, error = _pcc(ref, moving, up)
        if abs(dy) <= max_dy and abs(dx) <= max_dx:
            if up != upsample:
                print(f"    [{label}] accepted at upsample={up}: "
                      f"dy={dy:+.2f}, dx={dx:+.2f}")
            return dy, dx, error

    # Coarse downsampled fallback — operates at 1/8 resolution, so the
    # maximum possible shift in full-res pixels is strip_dim/8 * 8 = strip_dim,
    # but in practice the coarse result is usually small and within bounds.
    scale = 8
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref_d = rescale(ref,    1.0 / scale, anti_aliasing=True)
        mov_d = rescale(moving, 1.0 / scale, anti_aliasing=True)
    dy_c, dx_c, error = _pcc(ref_d, mov_d, 1)
    dy = dy_c * scale
    dx = dx_c * scale
    if abs(dy) <= max_dy and abs(dx) <= max_dx:
        print(f"    [{label}] coarse fallback accepted: "
              f"dy={dy:+.2f}, dx={dx:+.2f}")
        return dy, dx, error

    print(
        f"    WARNING [{label}]: all PCC attempts exceeded shift limit "
        f"(max_dy={max_dy:.0f}px, max_dx={max_dx:.0f}px). "
        f"Last coarse result: dy={dy:+.2f}, dx={dx:+.2f}. Using (0, 0)."
    )
    return 0.0, 0.0, 1.0


# ─────────────────────────────────────────────
#  PUBLIC SHIFT ESTIMATION
# ─────────────────────────────────────────────

def estimate_shift_horizontal(img_a, img_b, overlap_px,
                               fudge=0, upsample=10, max_shift=500,
                               multiscale=True,
                               bg_percentile=5,
                               max_shift_frac=0.5,
                               # legacy kwargs accepted but unused
                               global_thresh=None,
                               min_overlap_frac=None,
                               max_overlap_frac=None):
    """
    Estimate (dy, dx, overlap_px) to align tile B to the right of tile A.

    PCC is run on the right overlap_px columns of img_a vs the left
    overlap_px columns of img_b, after per-strip background subtraction.

    The shift constraint is max_shift_frac × strip dimension (and ≤ max_shift).

    Returns
    -------
    dy, dx, overlap_px : float, float, int
        overlap_px is passed through unchanged.
    """
    h_a, w_a = img_a.shape
    h_b, w_b = img_b.shape
    rows = min(h_a, h_b)
    ov   = min(overlap_px, w_a, w_b)

    # Extract overlap strips
    strip_a = img_a[:rows, -ov:]   # right edge of A
    strip_b = img_b[:rows,  :ov]   # left  edge of B

    # Background-subtract each strip independently
    ref    = subtract_background(strip_a, bg_percentile)
    moving = subtract_background(strip_b, bg_percentile)

    # Shift bounds: fraction of strip dimensions
    max_dy = min(max_shift, max_shift_frac * rows)
    max_dx = min(max_shift, max_shift_frac * ov)

    dy, dx, error = _pcc_with_shift_limit(
        ref, moving, max_dy, max_dx, upsample, label="horizontal"
    )
    print(f"    PCC horizontal: dy={dy:+.2f}, dx={dx:+.2f}, error={error:.4f}  "
          f"(strip {rows}×{ov}px, max_dy={max_dy:.0f}, max_dx={max_dx:.0f})")
    return dy, dx, overlap_px


def estimate_shift_vertical(img_a, img_b, overlap_px,
                             fudge=0, upsample=10, max_shift=500,
                             multiscale=True,
                             bg_percentile=5,
                             max_shift_frac=0.5,
                             # legacy kwargs accepted but unused
                             global_thresh=None,
                             min_overlap_frac=None,
                             max_overlap_frac=None):
    """
    Estimate (dy, dx, overlap_px) to align tile B below tile A.

    PCC is run on the bottom overlap_px rows of img_a vs the top
    overlap_px rows of img_b, after per-strip background subtraction.

    Returns
    -------
    dy, dx, overlap_px : float, float, int
    """
    h_a, w_a = img_a.shape
    h_b, w_b = img_b.shape
    cols = min(w_a, w_b)
    ov   = min(overlap_px, h_a, h_b)

    strip_a = img_a[-ov:, :cols]   # bottom edge of A
    strip_b = img_b[ :ov, :cols]   # top    edge of B

    ref    = subtract_background(strip_a, bg_percentile)
    moving = subtract_background(strip_b, bg_percentile)

    max_dy = min(max_shift, max_shift_frac * ov)
    max_dx = min(max_shift, max_shift_frac * cols)

    dy, dx, error = _pcc_with_shift_limit(
        ref, moving, max_dy, max_dx, upsample, label="vertical"
    )
    print(f"    PCC vertical:   dy={dy:+.2f}, dx={dx:+.2f}, error={error:.4f}  "
          f"(strip {ov}×{cols}px, max_dy={max_dy:.0f}, max_dx={max_dx:.0f})")
    return dy, dx, overlap_px


# ─────────────────────────────────────────────
#  WARP  (test_mode diagnostics only)
# ─────────────────────────────────────────────

def apply_shift_skimage(img, dy, dx, output_shape):
>>>>>>> Stashed changes
    tform = AffineTransform(translation=(dx, dy))
    warped = warp(
        img,
        inverse_map=tform.inverse,
        output_shape=output_shape,
        order=1,
        mode="constant",
        cval=0.0,
        preserve_range=True,
    )
    return warped.astype(np.float32)


<<<<<<< Updated upstream
# ============================================================
#  MOVEMENT REPORT
# ============================================================

def print_movement_report(pair_label, direction, dy, dx,
                           nominal_dy, nominal_dx,
                           canvas_y, canvas_x):
    corr_dy = dy - nominal_dy
    corr_dx = dx - nominal_dx
    print(f"\n  Tile pair  : {pair_label}  ({direction} neighbour)")
    print(f"     Nominal placement  : dy={nominal_dy:+5d}  dx={nominal_dx:+5d} px")
    print(f"     PCC correction     : dy={corr_dy:+5.1f}  dx={corr_dx:+5.1f} px")
    print(f"     Final canvas pos   : y={canvas_y:.1f}  x={canvas_x:.1f} px")
    if abs(corr_dy) < 0.5 and abs(corr_dx) < 0.5:
        print(f"     No correction needed -- tiles aligned perfectly at nominal.")
    else:
        print(f"     Vertical correction   : {corr_dy:+.1f} px")
        print(f"     Horizontal correction : {corr_dx:+.1f} px")


# ============================================================
#  GRID POSITION SOLVER
# ============================================================

def compute_positions(tiles, z, n_rows, n_cols,
                      overlap_x, overlap_y,
                      tile_h, tile_w,
                      max_shift, fudge, ref_ch, upsample):
    """
    Walk the grid left->right then top->bottom, estimating pairwise shifts
    and accumulating absolute (y, x) canvas positions.
    """
    pos = np.zeros((n_rows, n_cols, 2), dtype=np.float64)

    print("\n  -- Horizontal passes (left -> right) --")
    for r in range(n_rows):
        for c in range(1, n_cols):
            path_a = tiles.get((r, c - 1, z, ref_ch))
            path_b = tiles.get((r, c,     z, ref_ch))

            nominal_dx = tile_w - overlap_x
            nominal_dy = 0

            if path_a is None or path_b is None:
                pos[r, c] = pos[r, c - 1] + [0, nominal_dx]
                print(f"    ({r},{c-1})->({r},{c}): MISSING tile -- using nominal")
                continue

            img_a = load_tile(path_a)
            img_b = load_tile(path_b)

            dy, dx = estimate_shift_horizontal(img_a, img_b, overlap_x, 
                                              search_margin=fudge, upsample=upsample)

            pos[r, c, 0] = pos[r, c - 1, 0] + nominal_dy + dy
            pos[r, c, 1] = pos[r, c - 1, 1] + nominal_dx + dx

            print_movement_report(
                f"({r},{c-1})->({r},{c})", "horizontal",
                nominal_dy + dy, nominal_dx + dx,
                nominal_dy, nominal_dx,
                pos[r, c, 0], pos[r, c, 1],
            )

    print("\n  -- Vertical passes (top -> bottom) --")
    for c in range(n_cols):
        for r in range(1, n_rows):
            path_a = tiles.get((r - 1, c, z, ref_ch))
            path_b = tiles.get((r,     c, z, ref_ch))

            nominal_dy = tile_h - overlap_y
            nominal_dx = 0

            if path_a is None or path_b is None:
                pos[r, c] = pos[r - 1, c] + [nominal_dy, 0]
                print(f"    ({r-1},{c})->({r},{c}): MISSING tile -- using nominal")
                continue

            img_a = load_tile(path_a)
            img_b = load_tile(path_b)

            dy, dx = estimate_shift_vertical(img_a, img_b, overlap_y,
                                            search_margin=fudge, upsample=upsample)

            pos[r, c, 0] = pos[r - 1, c, 0] + nominal_dy + dy
            pos[r, c, 1] = pos[r - 1, c, 1] + nominal_dx + dx

            print_movement_report(
                f"({r-1},{c})->({r},{c})", "vertical",
                nominal_dy + dy, nominal_dx + dx,
                nominal_dy, nominal_dx,
                pos[r, c, 0], pos[r, c, 1],
            )

    print("\n  == Solved canvas positions ==")
    print(f"  {'Tile':>10}   {'y (px)':>10}   {'x (px)':>10}")
    for r in range(n_rows):
        for c in range(n_cols):
            print(f"  ({r:02d},{c:02d})     "
                  f"{pos[r,c,0]:>10.1f}   {pos[r,c,1]:>10.1f}")

    return pos


# ============================================================
#  PADDING HELPERS
# ============================================================

def pad_to_same_width(img_a, img_b):
    """Pad the narrower image on the right with zeros so both have equal width."""
    wa, wb = img_a.shape[1], img_b.shape[1]
    if wa == wb:
        return img_a, img_b
    target = max(wa, wb)
    def _pad_w(img, w):
        pad = target - w
        return np.pad(img, ((0, 0), (0, pad)), mode="constant", constant_values=0)
    return _pad_w(img_a, wa), _pad_w(img_b, wb)


def pad_to_same_height(img_a, img_b):
    """Pad the shorter image on the bottom with zeros so both have equal height."""
    ha, hb = img_a.shape[0], img_b.shape[0]
    if ha == hb:
        return img_a, img_b
    target = max(ha, hb)
    def _pad_h(img, h):
        pad = target - h
        return np.pad(img, ((0, pad), (0, 0)), mode="constant", constant_values=0)
    return _pad_h(img_a, ha), _pad_h(img_b, hb)


# ============================================================
#  SINUSOIDAL (RAISED-COSINE) BLENDING
# ============================================================

def blend_sinusoidal_x(left_ol, right_ol):
    """Raised-cosine blend along horizontal axis.

    ramp is broadcast to the full (H, W) shape of the overlap strips
    before applying the zero-pixel masks, avoiding index shape mismatches.
    """
    h, n = left_ol.shape
    t    = np.linspace(0.0, 1.0, n, dtype=np.float32)
    row  = 0.5 * (1.0 + np.cos(np.pi * t))          # (n,)  1 -> 0
    # Broadcast to full overlap shape so boolean indexing works element-wise
    ramp = np.broadcast_to(row, (h, n)).copy()        # (H, n), writeable

    # Where the left tile is zero, give full weight to the right (and vice-versa)
    ramp[left_ol  == 0] = 0.0
    ramp[right_ol == 0] = 1.0
    return ramp * left_ol + (1.0 - ramp) * right_ol


def blend_sinusoidal_y(top_ol, bot_ol):
    """Raised-cosine blend along vertical axis.

    Same broadcast fix as blend_sinusoidal_x but for the vertical direction.
    """
    n, w = top_ol.shape
    t    = np.linspace(0.0, 1.0, n, dtype=np.float32)
    col  = 0.5 * (1.0 + np.cos(np.pi * t))           # (n,)  1 -> 0
    ramp = np.broadcast_to(col[:, None], (n, w)).copy()  # (n, W), writeable

    ramp[top_ol == 0] = 0.0
    ramp[bot_ol == 0] = 1.0
    return ramp * top_ol + (1.0 - ramp) * bot_ol


# ============================================================
#  CORE STITCH PRIMITIVES
# ============================================================

def stitch_pair_horizontal(left, right, overlap_px):
    """Stitch two tiles side-by-side with sinusoidal blending."""
    left, right = pad_to_same_height(left, right)
    h, wl = left.shape
    wr = right.shape[1]
    overlap_px = min(overlap_px, wl, wr)

    left_body  = left[:,  :-overlap_px]
    left_ol    = left[:,  -overlap_px:]
    right_ol   = right[:, :overlap_px]
    right_body = right[:, overlap_px:]

    blended = blend_sinusoidal_x(left_ol, right_ol)
    result  = np.concatenate([left_body, blended, right_body], axis=1)
    print(f"      stitch_h: left={left.shape} right={right.shape} "
          f"overlap={overlap_px} -> result={result.shape}")
    return result


def stitch_pair_vertical(top, bottom, overlap_px):
    """Stitch two tiles top-to-bottom with sinusoidal blending."""
    top, bottom = pad_to_same_width(top, bottom)
    ht = top.shape[0]
    hb = bottom.shape[0]
    overlap_px = min(overlap_px, ht, hb)

    top_body    = top[:-overlap_px, :]
    top_ol      = top[-overlap_px:, :]
    bottom_ol   = bottom[:overlap_px, :]
    bottom_body = bottom[overlap_px:, :]

    top_ol, bottom_ol = pad_to_same_height(top_ol, bottom_ol)
    blended = blend_sinusoidal_y(top_ol, bottom_ol)
    result  = np.concatenate([top_body, blended, bottom_body], axis=0)
    print(f"      stitch_v: top={top.shape} bot={bottom.shape} "
          f"overlap={overlap_px} -> result={result.shape}")
    return result


# ============================================================
#  ROW AND FULL-SLICE STITCHING
# ============================================================

def stitch_row(tiles, z, ch, row_idx, n_cols, overlap_x, dy_corrections=None):
    """Stitch all columns in a single row horizontally."""
    if dy_corrections is None:
        dy_corrections = {}

    path = tiles.get((row_idx, 0, z, ch))
    if path is None:
        raise ValueError(f"Missing tile ({row_idx}, 0, z={z}, ch={ch})")
    row_img = load_tile(path)

    for c in range(1, n_cols):
        path_b = tiles.get((row_idx, c, z, ch))
        if path_b is None:
            print(f"    Warning: Missing tile ({row_idx},{c}) -- skipping column {c}")
            continue

        img_b = load_tile(path_b)
        dy = dy_corrections.get(c, 0)

        if dy != 0:
            abs_dy = abs(dy)
            if dy > 0:
                img_b   = np.pad(img_b,   ((abs_dy, 0), (0, 0)), mode="constant", constant_values=0)
                row_img = np.pad(row_img, ((0, abs_dy), (0, 0)), mode="constant", constant_values=0)
            else:
                img_b   = np.pad(img_b,   ((0, abs_dy), (0, 0)), mode="constant", constant_values=0)
                row_img = np.pad(row_img, ((abs_dy, 0), (0, 0)), mode="constant", constant_values=0)
            print(f"    dy={dy:+d} padding applied at col {c}: "
                  f"row_img->{row_img.shape}  img_b->{img_b.shape}")

        row_img = stitch_pair_horizontal(row_img, img_b, overlap_x)

    return row_img


def stitch_column(tiles, z, ch, col_idx, n_rows, overlap_y, dx_corrections=None):
    """Stitch all rows in a single column vertically."""
    if dx_corrections is None:
        dx_corrections = {}

    path = tiles.get((0, col_idx, z, ch))
    if path is None:
        raise ValueError(f"Missing tile (0, {col_idx}, z={z}, ch={ch})")
    col_img = load_tile(path)

    for r in range(1, n_rows):
        path_b = tiles.get((r, col_idx, z, ch))
        if path_b is None:
            print(f"    Warning: Missing tile ({r},{col_idx}) -- skipping row {r}")
            continue

        img_b = load_tile(path_b)
        dx = dx_corrections.get(r, 0)

        if dx != 0:
            abs_dx = abs(dx)
            if dx > 0:
                img_b   = np.pad(img_b,   ((0, 0), (abs_dx, 0)), mode="constant", constant_values=0)
                col_img = np.pad(col_img, ((0, 0), (0, abs_dx)), mode="constant", constant_values=0)
            else:
                img_b   = np.pad(img_b,   ((0, 0), (0, abs_dx)), mode="constant", constant_values=0)
                col_img = np.pad(col_img, ((0, 0), (abs_dx, 0)), mode="constant", constant_values=0)
            print(f"    dx={dx:+d} padding applied at row {r}: "
                  f"col_img->{col_img.shape}  img_b->{img_b.shape}")

        col_img = stitch_pair_vertical(col_img, img_b, overlap_y)

    return col_img


def stitch_full_slice(tiles, z, ch, n_rows, n_cols,
                      overlap_x, overlap_y,
                      dy_per_col=None, dx_per_row_col=None):
    """Stitch the complete 2D slice for one channel and one Z."""
    if dy_per_col is None:
        dy_per_col = {}
    if dx_per_row_col is None:
        dx_per_row_col = {}

    print(f"\n  Building rows for ch={ch}, z={z}...")
    row_images = []
    for r in range(n_rows):
        print(f"    Row {r}:")
        dy_corr = {c: dy_per_col.get((r, c), 0) for c in range(1, n_cols)}
        row_img = stitch_row(tiles, z, ch, r, n_cols, overlap_x, dy_corr)
        row_images.append(row_img)
        print(f"    Row {r} stitched -> {row_img.shape}")

    print(f"\n  Stacking rows vertically for ch={ch}, z={z}...")
    final = row_images[0]
    for r in range(1, n_rows):
        print(f"    Joining row {r-1} + row {r}:")
        final = stitch_pair_vertical(final, row_images[r], overlap_y)

    return final

def stitch_full_slice_naive(tiles, z, ch, n_rows, n_cols, overlap_x, overlap_y):
    """
    Stitch the complete 2D slice WITHOUT registration corrections.
    Uses only nominal overlap values.
    """
    print(f"\n  Building rows (NAIVE mode) for ch={ch}, z={z}...")
    row_images = []
    for r in range(n_rows):
        print(f"    Row {r}:")
        row_img = stitch_row(tiles, z, ch, r, n_cols, overlap_x, dy_corrections=None)
        row_images.append(row_img)
        print(f"    Row {r} stitched -> {row_img.shape}")

    print(f"\n  Stacking rows vertically (NAIVE mode) for ch={ch}, z={z}...")
    final = row_images[0]
    for r in range(1, n_rows):
        print(f"    Joining row {r-1} + row {r}:")
        final = stitch_pair_vertical(final, row_images[r], overlap_y)

    return final

# ============================================================
#  EXTRACT REGISTRATION CORRECTIONS FROM SOLVED POSITIONS
# ============================================================

def extract_corrections(pos, tile_h, tile_w, overlap_x, overlap_y):
    """Convert canvas positions into per-join dy/dx correction dicts."""
    n_rows, n_cols = pos.shape[:2]
    dy_per_col = {}
    dx_per_row_col = {}

    for r in range(n_rows):
        for c in range(1, n_cols):
            nominal_dx = tile_w - overlap_x
            nominal_dy = 0
            actual_dy  = pos[r, c, 0] - pos[r, c - 1, 0]
            dy_per_col[(r, c)] = int(round(actual_dy - nominal_dy))

    for c in range(n_cols):
        for r in range(1, n_rows):
            nominal_dx = 0
            actual_dx  = pos[r, c, 1] - pos[r - 1, c, 1]
            dx_per_row_col[(r, c)] = int(round(actual_dx - nominal_dx))

    return dy_per_col, dx_per_row_col


# ============================================================
#  RED / GREEN / YELLOW OVERLAY  (test mode diagnostic)
# ============================================================

def save_rg_overlay(img_fixed, img_moving, out_path,
                    title="Alignment check", zoom_region=None):
    """Composite two greyscale images as red/green false-colour overlay."""
    def norm_u8(arr):
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
        return np.zeros_like(arr, dtype=np.uint8)

    r_ch = norm_u8(img_fixed)
    g_ch = norm_u8(img_moving)
    b_ch = np.zeros_like(r_ch)
    rgb  = np.stack([r_ch, g_ch, b_ch], axis=-1)

    n_plots = 3 if zoom_region is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6), facecolor="#111")
    fig.suptitle(title, color="white", fontsize=13, y=1.01)

    axes[0].imshow(rgb)
    axes[0].set_title("Red=fixed | Green=moving | Yellow=overlap",
                      color="white", fontsize=9)
    axes[0].axis("off")

    if zoom_region is not None:
        y0, y1, x0, x1 = zoom_region
        from matplotlib.patches import Rectangle
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                         linewidth=2, edgecolor="cyan", facecolor="none")
        axes[0].add_patch(rect)

    diff = img_fixed.astype(np.float32) - img_moving.astype(np.float32)
    vmax = np.percentile(np.abs(diff), 99) or 1.0
    axes[1].imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title("Difference map (zero = perfect overlap)",
                      color="white", fontsize=9)
    axes[1].axis("off")

    if zoom_region is not None:
        y0, y1, x0, x1 = zoom_region
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                         linewidth=2, edgecolor="cyan", facecolor="none")
        axes[1].add_patch(rect)
        rgb_zoom = rgb[y0:y1, x0:x1]
        axes[2].imshow(rgb_zoom)
        axes[2].set_title(f"ZOOM: Overlap region\n({x1-x0}x{y1-y0} px)",
                          color="white", fontsize=9)
        axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#111")
    plt.close(fig)
    print(f"\n  Red/green overlay saved: {out_path}")


# ============================================================
#  TEST MODE
# ============================================================

def run_test_overlay(tiles, z, ref_ch, overlap_x, tile_h, tile_w,
                     max_shift, fudge, upsample, out_dir,
                     row_idx=0, col_idx=0):
    """
    Test mode: naive vs corrected alignment for tiles at
    (row_idx, col_idx) and (row_idx, col_idx+1).
    """
    print("\n" + "=" * 60)
    print(f"  TEST MODE -- row={row_idx}, cols {col_idx} and {col_idx+1}, Z={z}")
    print("=" * 60)

    path_a = tiles.get((row_idx, col_idx,     z, ref_ch))
    path_b = tiles.get((row_idx, col_idx + 1, z, ref_ch))

    if path_a is None or path_b is None:
        # Try row 0 as a fallback so the user gets some output
        alt_a = tiles.get((0, col_idx,     z, ref_ch))
        alt_b = tiles.get((0, col_idx + 1, z, ref_ch))
        if alt_a and alt_b:
            print(f"  Warning: row {row_idx} not found -- falling back to row 0.")
            path_a, path_b, row_idx = alt_a, alt_b, 0
        else:
            print(f"  Error: Could not find tiles at row={row_idx} "
                  f"cols={col_idx}/{col_idx+1} (or row=0) "
                  f"for ref channel {ref_ch} at Z={z}.")
            return

    img_a = load_tile(path_a)
    img_b = load_tile(path_b)
    print(f"\n  Tile A : {os.path.basename(path_a)}  {img_a.shape}")
    print(f"  Tile B : {os.path.basename(path_b)}  {img_b.shape}")

    print(f"\n  Overlap px: {overlap_x}, Search margin (fudge): {fudge}")
    dy, dx = estimate_shift_horizontal(img_a, img_b, overlap_x, 
                                       search_margin=fudge, upsample=upsample)

    nominal_dx = tile_w - overlap_x
    print(f"\n  Nominal horizontal offset : {nominal_dx} px")
    print(f"  PCC correction estimate   : dy={dy:+.2f}  dx={dx:+.2f} px")

    canvas_h = tile_h + abs(int(dy)) + 10
    canvas_w = tile_w + nominal_dx + abs(int(dx)) + 10

    # NAIVE
    canvas_naive_a = np.zeros((canvas_h, canvas_w), np.float32)
    canvas_naive_b = np.zeros((canvas_h, canvas_w), np.float32)
    canvas_naive_a[:tile_h, :tile_w] = img_a
    x0_naive = nominal_dx
    x1_naive = min(x0_naive + tile_w, canvas_w)
    canvas_naive_b[:tile_h, x0_naive:x1_naive] = img_b[:, :x1_naive - x0_naive]

    zoom_margin = 300
    zoom_x0 = max(0, tile_w - zoom_margin)
    zoom_x1 = min(canvas_w, tile_w + zoom_margin)
    zoom_y0 = max(0, canvas_h // 2 - zoom_margin)
    zoom_y1 = min(canvas_h, canvas_h // 2 + zoom_margin)

    tag = f"r{row_idx}_c{col_idx}-{col_idx+1}_z{z:04d}"
    naive_path = os.path.join(out_dir, f"test_{tag}_NAIVE.png")
    save_rg_overlay(canvas_naive_a, canvas_naive_b, naive_path,
                    title=f"NAIVE placement -- row {row_idx} cols {col_idx}/{col_idx+1}  Z={z}",
                    zoom_region=(zoom_y0, zoom_y1, zoom_x0, zoom_x1))

    # CORRECTED
    canvas_corr_a = np.zeros((canvas_h, canvas_w), np.float32)
    canvas_corr_b = np.zeros((canvas_h, canvas_w), np.float32)
    canvas_corr_a[:tile_h, :tile_w] = img_a
    shifted_b = apply_shift_skimage(img_b, dy, nominal_dx + dx, (canvas_h, canvas_w))
    canvas_corr_b += shifted_b

    corr_path = os.path.join(out_dir, f"test_{tag}_CORRECTED.png")
    save_rg_overlay(canvas_corr_a, canvas_corr_b, corr_path,
                    title=f"CORRECTED placement -- row {row_idx} cols {col_idx}/{col_idx+1}  Z={z}",
                    zoom_region=(zoom_y0, zoom_y1, zoom_x0, zoom_x1))

    # STITCHED
    stitched = stitch_pair_horizontal(img_a, img_b, overlap_x)
    stitch_path = os.path.join(out_dir, f"test_{tag}_STITCHED.tif")
    save_tiff(stitched, stitch_path)

    print(f"\n  Test outputs written to: {out_dir}")
    print(f"     test_{tag}_NAIVE.png")
    print(f"     test_{tag}_CORRECTED.png")
    print(f"     test_{tag}_STITCHED.tif")


# ============================================================
#  PAIR / ROW / COL MODES
# ============================================================

def run_pair_h(tiles, z, ref_ch, overlap_x, tile_h, tile_w,
               max_shift, fudge, upsample, out_dir,
               row_idx=0, col_idx=0):
    """Stitch (row_idx, col_idx) and (row_idx, col_idx+1) horizontally."""
    print(f"\n-- PAIR HORIZONTAL mode  (row={row_idx}, cols {col_idx}+{col_idx+1}, Z={z}) --")
    for c in [col_idx, col_idx + 1]:
        if tiles.get((row_idx, c, z, ref_ch)) is None:
            print(f"  Error: Missing tile ({row_idx},{c}). Aborting.")
            return

    img_a = load_tile(tiles[(row_idx, col_idx,     z, ref_ch)])
    img_b = load_tile(tiles[(row_idx, col_idx + 1, z, ref_ch)])
    
    dy, dx = estimate_shift_horizontal(img_a, img_b, overlap_x, 
                                       search_margin=fudge, upsample=upsample)
    print(f"  PCC shift: dy={dy:.2f}, dx={dx:.2f}")

    result = stitch_pair_horizontal(img_a, img_b, overlap_x)
    out = os.path.join(out_dir,
          f"pair_h_r{row_idx}_c{col_idx}-{col_idx+1}_z{z:04d}_C{ref_ch:02d}.tif")
    save_tiff(result, out)


def run_pair_v(tiles, z, ref_ch, overlap_y, tile_h, tile_w,
               max_shift, fudge, upsample, out_dir,
               row_idx=0, col_idx=0):
    """Stitch (row_idx, col_idx) and (row_idx+1, col_idx) vertically."""
    print(f"\n-- PAIR VERTICAL mode  (col={col_idx}, rows {row_idx}+{row_idx+1}, Z={z}) --")
    for r in [row_idx, row_idx + 1]:
        if tiles.get((r, col_idx, z, ref_ch)) is None:
            print(f"  Error: Missing tile ({r},{col_idx}). Aborting.")
            return

    img_a = load_tile(tiles[(row_idx,     col_idx, z, ref_ch)])
    img_b = load_tile(tiles[(row_idx + 1, col_idx, z, ref_ch)])
    
    dy, dx = estimate_shift_vertical(img_a, img_b, overlap_y,
                                     search_margin=fudge, upsample=upsample)
    print(f"  PCC shift: dy={dy:.2f}, dx={dx:.2f}")

    result = stitch_pair_vertical(img_a, img_b, overlap_y)
    out = os.path.join(out_dir,
          f"pair_v_c{col_idx}_r{row_idx}-{row_idx+1}_z{z:04d}_C{ref_ch:02d}.tif")
    save_tiff(result, out)


def run_single_row(tiles, z, ref_ch, n_cols, overlap_x, out_dir, row_idx=0):
    """Stitch all columns of the chosen row horizontally and save."""
    print(f"\n-- SINGLE ROW mode  (row={row_idx}, Z={z}) --")
    result = stitch_row(tiles, z, ref_ch, row_idx, n_cols, overlap_x)
    out = os.path.join(out_dir, f"row{row_idx}_z{z:04d}_C{ref_ch:02d}.tif")
    save_tiff(result, out)


def run_single_col(tiles, z, ref_ch, n_rows, overlap_y, out_dir, col_idx=0):
    """Stitch all rows of the chosen column vertically and save."""
    print(f"\n-- SINGLE COLUMN mode  (col={col_idx}, Z={z}) --")
    result = stitch_column(tiles, z, ref_ch, col_idx, n_rows, overlap_y)
    out = os.path.join(out_dir, f"col{col_idx}_z{z:04d}_C{ref_ch:02d}.tif")
    save_tiff(result, out)


# ============================================================
#  GRID OVERLAY VISUALISATION
# ============================================================

def _save_grid_overlay(pos, tile_h, tile_w, n_rows, n_cols, z, out_path):
    p = pos.copy()
    p[:, :, 0] -= p[:, :, 0].min()
    p[:, :, 1] -= p[:, :, 1].min()

    H = int(np.ceil(p[:, :, 0].max())) + tile_h
    W = int(np.ceil(p[:, :, 1].max())) + tile_w

    fig, ax = plt.subplots(figsize=(max(6, W // 200), max(4, H // 200)))
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.set_title(f"Tile grid overlay -- Z {z:04d}", fontsize=9)
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")

    cmap = matplotlib.colormaps["tab10"].resampled(n_rows)
    for r in range(n_rows):
        for c in range(n_cols):
            y0 = p[r, c, 0]; x0 = p[r, c, 1]
            rect = mpatches.Rectangle(
                (x0, y0), tile_w, tile_h,
                linewidth=1.0, edgecolor=cmap(r),
                facecolor=(*cmap(r)[:3], 0.08),
            )
            ax.add_patch(rect)
            ax.text(x0 + tile_w / 2, y0 + tile_h / 2, f"({r},{c})",
                    ha="center", va="center", fontsize=6, color=cmap(r))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Grid overlay saved: {out_path}")


# ============================================================
#  MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Monje Lab tile stitcher -- skimage registration + sinusoidal stitching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes
-----
  test   -- Naive vs corrected overlay for a chosen tile pair. Good for param tuning.
  pair_h -- Stitch (row_idx, col_idx) + (row_idx, col_idx+1) horizontally.
  pair_v -- Stitch (row_idx, col_idx) + (row_idx+1, col_idx) vertically.
  row    -- Stitch all columns of --row_idx.
  col    -- Stitch all rows of --col_idx.
  real   -- Full stitch: all tiles, all channels.
             Add --z_slice N for a single Z slice; omit for the full volume.

Positional-selection flags
--------------------------
  --row_idx N   Row to use in test / pair_h / pair_v / row modes  (default: 0)
  --col_idx N   Col to use in test / pair_h / pair_v / col modes  (default: 0)
  --z_slice  N  Pin to a specific Z index.
                In 'real' mode: single slice; omit for all Z (full volume).
                In other modes: overrides the default (first available Z).

Examples
--------
  # Check alignment at row 5, col 3 junction:
  python registration.py --input_dir /data --overlap 10 --mode test --row_idx 5 --col_idx 3

  # Stitch just row 4, at Z=12:
  python registration.py --input_dir /data --overlap 10 --mode row --row_idx 4 --z_slice 12

  # Stitch column 2 (all rows), first available Z:
  python registration.py --input_dir /data --overlap 10 --mode col --col_idx 2

  # Full volume stitch (all Z slices):
  python registration.py --input_dir /data --overlap 10 --mode real

  # Full stitch of a single Z slice only:
  python registration.py --input_dir /data --overlap 10 --mode real --z_slice 5
        """,
    )
    ap.add_argument("--input_dir",   required=True)
    ap.add_argument("--output_dir",  default=None)
    ap.add_argument("--overlap",     type=int, required=True,
                    help="Nominal overlap as %% of tile size (e.g. 10)")
    ap.add_argument("--mode",
                    choices=["test", "pair_h", "pair_v", "row", "col", "real"],
                    default="real")
    ap.add_argument("--max_shift",   type=int, default=200,
                    help="Warning threshold for unexpectedly large shifts (px)")
    ap.add_argument("--fudge",       type=int, default=500,
                    help="Extra search margin beyond overlap region (px)")
    ap.add_argument("--ref_channel", type=int, default=None)
    ap.add_argument("--upsample",    type=int, default=1)
    ap.add_argument("--visualize",   action="store_true",
                    help="(real mode) Save a grid-overlay PNG per Z slice")

    # ── Tile-selection flags ────────────────────────────────────────
    ap.add_argument("--row_idx", type=int, default=0,
                    help="Row index for test/pair_h/pair_v/row modes (default: 0)")
    ap.add_argument("--col_idx", type=int, default=0,
                    help="Col index for test/pair_h/pair_v/col modes (default: 0)")
    ap.add_argument("--z_slice", type=int, default=None,
                    help=(
                        "Z index to process. "
                        "In 'real' mode: single slice (omit for all Z / full volume). "
                        "In other modes: overrides the default first-Z."
                    ))

    args = ap.parse_args()

    root_out = args.output_dir or args.input_dir
    frac     = args.overlap / 100.0

    # ── Discover tiles ──────────────────────────────────────────────
    files = [f for f in os.listdir(args.input_dir) if f.endswith(".ome.tif")]
    tiles = {}
    zs, chs = set(), set()
    prefix  = None

    for f in files:
        parsed = parse_filename(f)
        if parsed:
            r, c, ch, z, pfx = parsed
            tiles[(r, c, z, ch)] = os.path.join(args.input_dir, f)
            zs.add(z); chs.add(ch)
            if prefix is None:
                prefix = pfx

    if not tiles:
        sys.exit("No tiles found -- filenames must match:\n"
                 "  <prefix>[ROW x COL]_C<CH>_z<Z>.ome.tif")

    zs  = sorted(zs)
    chs = sorted(chs)
    ref = args.ref_channel if args.ref_channel is not None else chs[0]

    # ── Resolve Z scope ─────────────────────────────────────────────
    if args.z_slice is not None:
        if args.z_slice not in zs:
            sys.exit(f"--z_slice {args.z_slice} not found. "
                     f"Available Z values: {zs}")
        work_z  = args.z_slice
        work_zs = [args.z_slice]
    else:
        work_z  = zs[0]   # default for quick modes
        work_zs = zs       # all Z for real mode

    # ── Sample tile dimensions ──────────────────────────────────────
    sample_path = next(
        (v for (r, c, z_, ch), v in tiles.items() if z_ == work_z and ch == ref), None)
    if sample_path is None:
        sys.exit(f"Cannot find any tile for ref channel {ref} at Z={work_z}.")
    sample = load_tile(sample_path)
    tile_h, tile_w = sample.shape
    ov_x = max(1, int(round(frac * tile_w)))
    ov_y = max(1, int(round(frac * tile_h)))

    # ── Print summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Monje Lab -- Image Registration & Stitching")
    print("=" * 60)
    print(f"  Mode       : {args.mode}")
    print(f"  All Z      : {zs}")
    if args.z_slice is not None:
        print(f"  Working Z  : {work_z}  (pinned via --z_slice)")
    elif args.mode == "real":
        print(f"  Working Z  : all {len(zs)} slices (full volume)")
    else:
        print(f"  Working Z  : {work_z}  (first available)")
    print(f"  Channels   : {chs}  (ref={ref})")
    print(f"  Overlap    : {args.overlap}%  -> x={ov_x} px, y={ov_y} px")
    print(f"  Tile size  : {tile_h} x {tile_w} px")
    print(f"  Max shift  : +/-{args.max_shift} px")
    print(f"  Fudge      : +{args.fudge} px")
    print(f"  Upsample   : {args.upsample}x -> {1/args.upsample:.2f} px accuracy")
    if args.mode in ("test", "pair_h", "pair_v", "row"):
        print(f"  --row_idx  : {args.row_idx}")
    if args.mode in ("test", "pair_h", "pair_v", "col"):
        print(f"  --col_idx  : {args.col_idx}")
    print()

    # ── Output directory ────────────────────────────────────────────
    out_dir = os.path.join(root_out, (prefix or "output") + "_registered")
    os.makedirs(out_dir, exist_ok=True)

    # ── Grid dimensions (at work_z) ─────────────────────────────────
    keys_wz   = [(r, c) for (r, c, z_, ch) in tiles if z_ == work_z]
    n_rows_wz = max(r for r, c in keys_wz) + 1
    n_cols_wz = max(c for r, c in keys_wz) + 1

    # ────────────────────────────────────────────────────────────────
    #  QUICK MODES
    # ────────────────────────────────────────────────────────────────
    if args.mode == "test":
        run_test_overlay(
            tiles, work_z, ref,
            overlap_x=ov_x, tile_h=tile_h, tile_w=tile_w,
            max_shift=args.max_shift, fudge=args.fudge, upsample=args.upsample,
            out_dir=out_dir, row_idx=args.row_idx, col_idx=args.col_idx,
        )
        print("\nTest mode done. Review the PNGs, then re-run with --mode real.")
        return

    if args.mode == "pair_h":
        run_pair_h(tiles, work_z, ref, ov_x, tile_h, tile_w,
                   args.max_shift, args.fudge, args.upsample, out_dir,
                   row_idx=args.row_idx, col_idx=args.col_idx)
        return

    if args.mode == "pair_v":
        run_pair_v(tiles, work_z, ref, ov_y, tile_h, tile_w,
                   args.max_shift, args.fudge, args.upsample, out_dir,
                   row_idx=args.row_idx, col_idx=args.col_idx)
        return

    if args.mode == "row":
        run_single_row(tiles, work_z, ref, n_cols_wz, ov_x, out_dir,
                       row_idx=args.row_idx)
        return

    if args.mode == "col":
        run_single_col(tiles, work_z, ref, n_rows_wz, ov_y, out_dir,
                       col_idx=args.col_idx)
        return

    # ────────────────────────────────────────────────────────────────
    #  REAL MODE
    # ────────────────────────────────────────────────────────────────
    if args.z_slice is not None:
        print(f"  real mode: single Z slice ({work_z})")
    else:
        print(f"  real mode: all {len(work_zs)} Z slices (full volume)")

    ch_dirs = {}
    ch_dirs_naive = {}
    for ch in chs:
        d = os.path.join(out_dir, f"Channel_{ch:02d}")
        os.makedirs(d, exist_ok=True)
        ch_dirs[ch] = d
        
        # Create naive output directory
        d_naive = os.path.join(out_dir, f"Channel_{ch:02d}_NAIVE")
        os.makedirs(d_naive, exist_ok=True)
        ch_dirs_naive[ch] = d_naive

    if args.visualize:
        viz_dir = os.path.join(out_dir, "grid_overlays")
        os.makedirs(viz_dir, exist_ok=True)

    for z in work_zs:
        print(f"\n{'='*60}")
        print(f"  Z = {z:04d}")
        print(f"{'='*60}")

        keys = [(r, c) for (r, c, z_, ch) in tiles if z_ == z]
        if not keys:
            print("  No tiles for this Z -- skipping.")
            continue

        n_rows = max(r for r, c in keys) + 1
        n_cols = max(c for r, c in keys) + 1
        print(f"  Grid: {n_rows} rows x {n_cols} cols")

        pos = compute_positions(
            tiles, z, n_rows, n_cols,
            ov_x, ov_y, tile_h, tile_w,
            args.max_shift, args.fudge, ref, args.upsample,
        )
        dy_per_col, dx_per_row_col = extract_corrections(
            pos, tile_h, tile_w, ov_x, ov_y)

        if args.visualize:
            _save_grid_overlay(pos, tile_h, tile_w, n_rows, n_cols, z,
                               os.path.join(viz_dir, f"grid_z{z:04d}.png"))

        for ch in chs:
            # NAIVE stitching (no corrections)
            print(f"\n  Stitching channel {ch} (NAIVE - no registration)...")
            img_naive = stitch_full_slice_naive(
                tiles, z, ch, n_rows, n_cols, ov_x, ov_y
            )
            out_name_naive = f"stitched_z{z:04d}_C{ch:02d}_NAIVE.tif"
            save_tiff(img_naive, os.path.join(ch_dirs_naive[ch], out_name_naive))
            
            # CORRECTED stitching (with registration)
            print(f"\n  Stitching channel {ch} (CORRECTED - with registration)...")
            img_corrected = stitch_full_slice(
                tiles, z, ch, n_rows, n_cols, ov_x, ov_y,
                dy_per_col=dy_per_col, dx_per_row_col=dx_per_row_col,
            )
            out_name_corrected = f"stitched_z{z:04d}_C{ch:02d}_CORRECTED.tif"
            save_tiff(img_corrected, os.path.join(ch_dirs[ch], out_name_corrected))

    print("\nAll done!")


if __name__ == "__main__":
    main()
=======
# ─────────────────────────────────────────────
#  DIAGNOSTIC HELPERS
# ─────────────────────────────────────────────

def compute_mse(img_a, img_b):
    if img_a.shape != img_b.shape:
        raise ValueError(
            f"compute_mse: shape mismatch {img_a.shape} vs {img_b.shape}"
        )
    return float(np.mean(np.square(img_a.astype(np.float64)
                                   - img_b.astype(np.float64))))


def compute_mutual_information(img_a, img_b, bins=64):
    a = img_a.ravel().astype(np.float64)
    b = img_b.ravel().astype(np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    hist_ab, _, _ = np.histogram2d(a, b, bins=bins)
    hist_ab = hist_ab + 1e-12
    p_ab = hist_ab / hist_ab.sum()
    p_a  = p_ab.sum(axis=1, keepdims=True)
    p_b  = p_ab.sum(axis=0, keepdims=True)
    mi = float(np.sum(p_ab * np.log2(p_ab / (p_a * p_b))))
    return max(mi, 0.0)


# ─────────────────────────────────────────────
#  LEGACY STUBS
# ─────────────────────────────────────────────

def compute_global_threshold(tiles, sample_frac=0.10, n_bins=512):
    """Removed: background subtraction replaces binary masking."""
    return None


def compute_foreground_mask(strip, global_thresh=None):
    """Removed: use subtract_background instead."""
    return np.ones(strip.shape, dtype=bool)


def save_mask_png(img, mask, out_path):
    """Removed: mask PNGs are no longer produced."""
    pass
>>>>>>> Stashed changes
