"""
<<<<<<< Updated upstream
Automatic 3D Image Stitching — Monje Lab
========================================
Detects tile grid, channels, and Z slices from OME-TIFF filenames,
stitches each Z slice into a 2D image per channel, and saves individually.
Filename suffix format expected: ...[RR x CC]_C<ch>_z<zzzz>.ome.tif
=======
stitching.py — Monje Lab Stitcher
===================================
Row, column, and full-slice stitching.

Changes vs previous version
-----------------------------
* global_thresh / binary masking removed throughout.
  Background subtraction is now handled inside registration.py before PCC;
  stitching.py never needs to touch the threshold.

* max_shift_frac forwarded to estimate_shift_* so the shift constraint is
  respected consistently.

* bg_percentile forwarded so callers can tune background subtraction.

* overlap_px is passed through from the shift estimators unchanged (no
  overlap-search); the blend zone always uses the nominal overlap.
>>>>>>> Stashed changes
"""

import os
import sys
import re
import argparse
import numpy as np
<<<<<<< Updated upstream

try:
    from PIL import Image
except ImportError:
    sys.exit("Please install Pillow: pip install Pillow")

try:
    import tifffile
    USE_TIFFFILE = True
except ImportError:
    USE_TIFFFILE = False
    print("tifffile not found; falling back to Pillow.")
=======
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from io_utils import load_tile
from blending import stitch_pair
from registration import (
    estimate_shift_horizontal,
    estimate_shift_vertical,
    apply_shift_skimage,
    compute_mse,
    compute_mutual_information,
)
>>>>>>> Stashed changes


# -------------------
# Overlap blending
# -------------------
def blend_weighted_x(left_ol, right_ol):
    n = left_ol.shape[1]
    ramp = np.linspace(1.0, 0.0, n, dtype=np.float32)[None, :]
    return ramp * left_ol + (1.0 - ramp) * right_ol

<<<<<<< Updated upstream

def blend_weighted_y(top_ol, bot_ol):
    n = top_ol.shape[0]
    ramp = np.linspace(1.0, 0.0, n, dtype=np.float32)[:, None]
    return ramp * top_ol + (1.0 - ramp) * bot_ol


def blend_sinusoidal_x(left_ol, right_ol):
    """Raised-cosine (sinusoidal) blend along the horizontal axis.

    Uses w(t) = 0.5 * (1 + cos(π·t)) for t ∈ [0, 1], which gives a smooth
    S-curve with zero derivatives at both endpoints. This avoids the
    brightness kinks that linear ('weighted') blending can leave at seam
    boundaries, making it a better choice for tiles with significant
    illumination variation across the overlap region.
    """
    n = left_ol.shape[1]
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[None, :]
    ramp = 0.5 * (1.0 + np.cos(np.pi * t))   # 1 → 0, smooth

    # CHANGE: If left_ol =0, then give all the weight to right_ol, so make ramp 0
    ## If right_ol =0, then make ramp = 1 
    ## On individual pixels (matrix way)
    ramp[left_ol == 0] = 0 ## Changed
    ramp[right_ol == 0] = 1 ## Changed
    return ramp * left_ol + (1.0 - ramp) * right_ol


def blend_sinusoidal_y(top_ol, bot_ol):
    """Raised-cosine (sinusoidal) blend along the vertical axis.

    See blend_sinusoidal_x for a full description of the weighting curve.
    """
    n = top_ol.shape[0]
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    ramp = 0.5 * (1.0 + np.cos(np.pi * t))
    return ramp * top_ol + (1.0 - ramp) * bot_ol


def blend_average(left, right):
    return 0.5 * (left + right)


def blend_majority(left, right):
    return np.maximum(left, right)


def stitch_horizontal(left, right, overlap_px, method):
    left_body, left_ol = left[:, :-overlap_px], left[:, -overlap_px:]
    right_ol, right_body = right[:, :overlap_px], right[:, overlap_px:]

    if method == "weighted":
        blended = blend_weighted_x(left_ol, right_ol)
    elif method == "sinusoidal":
        blended = blend_sinusoidal_x(left_ol, right_ol)
    elif method == "average":
        blended = blend_average(left_ol, right_ol)
    elif method == "majority":
        blended = blend_majority(left_ol, right_ol)
    else:
        raise ValueError(f"Unknown method: {method}")

    return np.concatenate([left_body, blended, right_body], axis=1)


def stitch_vertical(top, bottom, overlap_px, method):
    top_body, top_ol = top[:-overlap_px, :], top[-overlap_px:, :]
    bottom_ol, bottom_body = bottom[:overlap_px, :], bottom[overlap_px:, :]

    if method == "weighted":
        blended = blend_weighted_y(top_ol, bottom_ol)
    elif method == "sinusoidal":
        blended = blend_sinusoidal_y(top_ol, bottom_ol)
    elif method == "average":
        blended = blend_average(top_ol, bottom_ol)
    elif method == "majority":
        blended = blend_majority(top_ol, bottom_ol)
    else:
        raise ValueError(f"Unknown method: {method}")

    return np.concatenate([top_body, blended, bottom_body], axis=0)


# -------------------
# Image I/O
# -------------------
def save_image(img, path):
    vmin, vmax = img.min(), img.max()
    norm = ((img - vmin) / (vmax - vmin) * 65535).astype(np.uint16) if vmax > vmin else np.zeros_like(img, np.uint16)
    if USE_TIFFFILE:
        tifffile.imwrite(path, norm, photometric='minisblack')
    else:
        Image.fromarray(norm).save(path)
    print(f"  Saved: {path} ({norm.shape[1]} x {norm.shape[0]} px)")


def load_tile(path):
    if USE_TIFFFILE:
        img = tifffile.imread(path).astype(np.float32)
    else:
        img = np.array(Image.open(path)).astype(np.float32)
    while img.ndim > 2:
        img = img[0]
    return img


def parse_filename(fname):
    """
    Extract row, col, channel, z, and filename prefix.

    Only the suffix is matched — the prefix before '[RR x CC]' can be anything.
    Expected suffix format: [RR x CC]_C<ch>_z<zzzz>.ome.tif
    Example: 260128_anything_prefix[00 x 00]_C00_z0100.ome.tif

    Returns:
        (row, col, channel, z, prefix) or None if the filename doesn't match.
    """
    pattern = re.compile(
        r"^(.*?)\[(\d+) x (\d+)\]_C(\d+)_z(\d+)\.ome\.tif$"
    )
    m = pattern.match(fname)
    if not m:
        return None
    prefix_raw = m.group(1)
    row, col, channel, z = map(int, m.groups()[1:])
    # Strip trailing underscores/spaces so the folder name is clean
    prefix = prefix_raw.rstrip("_ ") or "stitched"
    return row, col, channel, z, prefix


# -------------------
# Main
# -------------------
def main():
    parser = argparse.ArgumentParser(description="Automatic 3D stitching of OME-TIFF tiles")
    parser.add_argument("--input_dir", required=True,
                        help="Folder containing tiles")
    parser.add_argument("--output_dir", default=None,
                        help="Root folder for stitched output. A sub-folder named after the "
                             "filename prefix is created here. Defaults to input_dir if not set.")
    parser.add_argument("--overlap", type=int, required=True,
                        help="Tile overlap as an integer percentage (e.g. 20 for 20%%)")
    parser.add_argument("--method", choices=["weighted", "sinusoidal", "average", "majority"],
                        default="weighted",
                        help=(
                            "Overlap blending method. "
                            "'weighted': linear ramp (fast, slight edge artefacts). "
                            "'sinusoidal': raised-cosine ramp (smoother seams, recommended for uneven illumination). "
                            "'average': equal 50/50 mix. "
                            "'majority': max-value (bright-field / binary masks)."
                        ))
    args = parser.parse_args()

    overlap_fraction = args.overlap / 100.0
    root_out = args.output_dir if args.output_dir else args.input_dir

    files = [f for f in os.listdir(args.input_dir) if f.endswith(".ome.tif")]
    tiles = {}
    z_slices, channels = set(), set()
    detected_prefix = None

    # Parse all files
    for f in files:
        parsed = parse_filename(f)
        if parsed:
            row, col, channel, z, prefix = parsed
            tiles[(row, col, z, channel)] = os.path.join(args.input_dir, f)
            z_slices.add(z)
            channels.add(channel)
            if detected_prefix is None:
                detected_prefix = prefix  # capture prefix from the first matched file

    if not tiles:
        sys.exit("No matching OME-TIFF tiles found.")

    z_slices = sorted(z_slices)
    channels = sorted(channels)

    # Build parent output folder:  <root_out>/<prefix>/
    parent_dir = os.path.join(root_out, detected_prefix)
    os.makedirs(parent_dir, exist_ok=True)

    print(f"Detected Z slices : {z_slices}")
    print(f"Detected channels : {channels}")
    print(f"Detected prefix   : {detected_prefix}")
    print(f"Overlap           : {args.overlap}%")
    print(f"Blend method      : {args.method}")
    print(f"Output parent dir : {parent_dir}")

    # Create one output folder per channel inside the parent folder
    channel_dirs = {}
    for ch in channels:
        ch_dir = os.path.join(parent_dir, f"channel_{ch}")
        os.makedirs(ch_dir, exist_ok=True)
        channel_dirs[ch] = ch_dir
    print(f"\nCreated channel folders: {list(channel_dirs.values())}")

    for z in z_slices:
        print(f"\n--- Z slice {z:04d} ---")

        rows_at_z = [r for r, c, z_, ch in tiles if z_ == z]
        cols_at_z = [c for r, c, z_, ch in tiles if z_ == z]
        n_rows, n_cols = max(rows_at_z) + 1, max(cols_at_z) + 1
        print(f"  Grid: {n_rows} rows x {n_cols} cols")

        # Compute overlap in pixels from one representative tile
        sample_path = tiles[(0, 0, z, channels[0])]
        sample_tile = load_tile(sample_path)
        tile_h, tile_w = sample_tile.shape
        overlap_x = max(1, int(round(overlap_fraction * tile_w)))
        overlap_y = max(1, int(round(overlap_fraction * tile_h)))

        for ch in channels:
            print(f"  channel_{ch} ...")

            # Stitch each row horizontally
            rows_stitched = []
            for r in range(n_rows):
                row_img = load_tile(tiles[(r, 0, z, ch)])
                for c in range(1, n_cols):
                    row_img = stitch_horizontal(
                        row_img, load_tile(tiles[(r, c, z, ch)]), overlap_x, args.method
                    )
                rows_stitched.append(row_img)

            # Stitch rows vertically
            final_img_2D = rows_stitched[0]
            for r_img in rows_stitched[1:]:
                final_img_2D = stitch_vertical(final_img_2D, r_img, overlap_y, args.method)

            # Save 2D tif into the channel's subfolder
            out_name = f"stitched_z{z:04d}_C{ch:02d}.tif"
            out_path = os.path.join(channel_dirs[ch], out_name)
            save_image(final_img_2D, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
=======
@dataclass
class ShiftPlan:
    # (row, col) → (dy, dx, overlap_px)
    h_shifts: Dict[Tuple[int, int], Tuple[float, float, int]] = field(default_factory=dict)
    # row_idx → (dy, dx, overlap_px)
    v_shifts: Dict[int, Tuple[float, float, int]]             = field(default_factory=dict)


# ─────────────────────────────────────────────
#  OVERLAP STAT RECORD
# ─────────────────────────────────────────────

@dataclass
class OverlapStat:
    pair:       str
    mse_before: float = 0.0
    mse_after:  float = 0.0
    mi_before:  float = 0.0
    mi_after:   float = 0.0


def _pct_change(before: float, after: float, invert: bool = False) -> str:
    if before == 0.0:
        return "     N/A"
    if invert:
        pct = (after - before) / before * 100.0
    else:
        pct = (before - after) / before * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:7.1f}%"


def print_stats_table(stats: List[OverlapStat]) -> None:
    col_w = 105
    print("\n" + "=" * col_w)
    print("  ALIGNMENT STATISTICS")
    print("  MSE = mean-squared pixel error in the overlap zone  (lower = better)")
    print("  MI  = mutual information in the overlap zone        (higher = better)")
    print("=" * col_w)
    header = (
        f"  {'Pair':<28}  "
        f"{'MSE before':>11}  {'MSE after':>11}  {'MSE % chg':>10}  "
        f"{'MI before':>9}  {'MI after':>9}  {'MI % chg':>9}"
    )
    print(header)
    print("  " + "-" * (col_w - 2))
    for s in stats:
        print(
            f"  {s.pair:<28}  "
            f"{s.mse_before:>11.1f}  {s.mse_after:>11.1f}  "
            f"{_pct_change(s.mse_before, s.mse_after):>10}  "
            f"{s.mi_before:>9.4f}  {s.mi_after:>9.4f}  "
            f"{_pct_change(s.mi_before, s.mi_after, invert=True):>9}"
        )
    print("=" * col_w + "\n")


# ─────────────────────────────────────────────
#  SHIFT COMPUTATION
# ─────────────────────────────────────────────

def compute_shifts(tiles, z, ref_ch, n_rows, n_cols,
                   overlap_x, overlap_y,
                   fudge=0, upsample=10, max_shift=500,
                   multiscale=True,
                   bg_percentile=5,
                   max_shift_frac=0.5,
                   # legacy kwarg accepted but ignored
                   global_thresh=None,
                   mask_dir=None):
    """
    Compute horizontal and vertical shifts for all tile pairs.

    Horizontal: pairwise (r, c-1) → (r, c)  for each row.
    Vertical:   pairwise (r-1, c) → (r, c)  per column, then median
                across all columns as the consensus for that row pair.

    Returns ShiftPlan.
    """
    plan = ShiftPlan()
    print(f"\n  Computing shifts on ref_ch={ref_ch}, z={z} …")

    # ── Horizontal passes ─────────────────────────────────────────
    print("\n  -- Horizontal passes --")
    for r in range(n_rows):
        for c in range(1, n_cols):
            path_a = tiles.get((r, c - 1, z, ref_ch))
            path_b = tiles.get((r, c,     z, ref_ch))
            if path_a is None or path_b is None:
                plan.h_shifts[(r, c)] = (0.0, 0.0, overlap_x)
                print(f"    ({r},{c-1})→({r},{c}): MISSING — using (0, 0, {overlap_x})")
                continue

            img_a = load_tile(path_a)
            img_b = load_tile(path_b)

            print(f"    H ({r},{c-1})→({r},{c})")
            dy, dx, ov = estimate_shift_horizontal(
                img_a, img_b, overlap_x,
                fudge=fudge, upsample=upsample,
                max_shift=max_shift, multiscale=multiscale,
                bg_percentile=bg_percentile,
                max_shift_frac=max_shift_frac,
            )
            plan.h_shifts[(r, c)] = (dy, dx, ov)

    # ── Vertical passes (per-column tile pairs, then median) ──────
    print("\n  -- Vertical passes (per-column tile pairs) --")
    for r in range(1, n_rows):
        col_dys = []
        col_dxs = []
        col_ovs = []

        for c in range(n_cols):
            path_a = tiles.get((r - 1, c, z, ref_ch))
            path_b = tiles.get((r,     c, z, ref_ch))
            if path_a is None or path_b is None:
                print(f"    V ({r-1},{c})→({r},{c}): MISSING — skipping column")
                continue

            img_a = load_tile(path_a)
            img_b = load_tile(path_b)

            print(f"    V ({r-1},{c})→({r},{c})")
            dy, dx, ov = estimate_shift_vertical(
                img_a, img_b, overlap_y,
                fudge=fudge, upsample=upsample,
                max_shift=max_shift, multiscale=multiscale,
                bg_percentile=bg_percentile,
                max_shift_frac=max_shift_frac,
            )
            col_dys.append(dy)
            col_dxs.append(dx)
            col_ovs.append(ov)

        if col_dys:
            med_dy = float(np.median(col_dys))
            med_dx = float(np.median(col_dxs))
            med_ov = int(round(np.median(col_ovs)))
            print(f"    V row {r-1}→{r}: median dy={med_dy:+.2f}  dx={med_dx:+.2f}  "
                  f"overlap={med_ov}px  "
                  f"(from {len(col_dys)} columns: "
                  f"dy={[round(v,1) for v in col_dys]}, "
                  f"dx={[round(v,1) for v in col_dxs]})")
            plan.v_shifts[r] = (med_dy, med_dx, med_ov)
        else:
            plan.v_shifts[r] = (0.0, 0.0, overlap_y)
            print(f"    V row {r-1}→{r}: no valid columns — using (0, 0, {overlap_y})")

    return plan


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def _pad_horizontal_for_shift(accum, incoming, dx):
    if abs(dx) < 0.5:
        return accum, incoming
    abs_d = int(round(abs(dx)))
    if dx > 0:
        accum    = np.pad(accum,    ((0, 0), (0, abs_d)), constant_values=0)
        incoming = np.pad(incoming, ((0, 0), (abs_d, 0)), constant_values=0)
    else:
        accum    = np.pad(accum,    ((0, 0), (abs_d, 0)), constant_values=0)
        incoming = np.pad(incoming, ((0, 0), (0, abs_d)), constant_values=0)
    return accum, incoming


def _pad_vertical_for_shift(accum, incoming, dy):
    if abs(dy) < 0.5:
        return accum, incoming
    abs_d = int(round(abs(dy)))
    if dy > 0:
        accum    = np.pad(accum,    ((0, abs_d), (0, 0)), constant_values=0)
        incoming = np.pad(incoming, ((abs_d, 0), (0, 0)), constant_values=0)
    else:
        accum    = np.pad(accum,    ((abs_d, 0), (0, 0)), constant_values=0)
        incoming = np.pad(incoming, ((0, abs_d), (0, 0)), constant_values=0)
    return accum, incoming


def _align_for_stitch(accum, incoming, dy, dx, axis):
    if axis == "h":
        return _pad_vertical_for_shift(accum, incoming, dy)
    if axis == "v":
        return _pad_horizontal_for_shift(accum, incoming, dx)
    raise ValueError(f"axis must be 'h' or 'v', got {axis!r}")


def _apply_row_shifts(tiles, z, ch, row_idx, n_cols, overlap_x, plan, blend):
    path = tiles.get((row_idx, 0, z, ch))
    if path is None:
        raise ValueError(f"Missing tile ({row_idx}, 0, z={z}, ch={ch})")

    accum = load_tile(path)
    for c in range(1, n_cols):
        path_b = tiles.get((row_idx, c, z, ch))
        if path_b is None:
            continue
        incoming = load_tile(path_b)
        dy, dx, ov = plan.h_shifts.get((row_idx, c), (0.0, 0.0, overlap_x))

        accum, incoming = _align_for_stitch(accum, incoming, dy, dx, axis="h")
        accum = stitch_pair(accum, incoming, ov, axis="h", blend=blend)
    return accum


# ─────────────────────────────────────────────
#  METRIC HELPERS
# ─────────────────────────────────────────────

def _metrics_for_offset(img_a, img_b, offset_y, offset_x):
    yb = int(round(offset_y))
    xb = int(round(offset_x))
    y0 = max(0, yb);  y1 = min(img_a.shape[0], yb + img_b.shape[0])
    x0 = max(0, xb);  x1 = min(img_a.shape[1], xb + img_b.shape[1])
    if y1 <= y0 or x1 <= x0:
        return 0.0, 0.0
    sa = img_a[y0:y1, x0:x1].astype(np.float64)
    sb = img_b[y0 - yb:y1 - yb, x0 - xb:x1 - xb].astype(np.float64)
    mse = float(np.mean(np.square(sa - sb)))
    mi  = compute_mutual_information(sa.astype(np.float32), sb.astype(np.float32))
    return mse, mi


def _metrics_naive_h(img_a, img_b, overlap_x):
    return _metrics_for_offset(img_a, img_b, 0.0, img_a.shape[1] - overlap_x)


def _metrics_registered_h(img_a, img_b, dy, dx, overlap_x):
    return _metrics_for_offset(img_a, img_b, dy, img_a.shape[1] - overlap_x + dx)


def _metrics_naive_v(row_a, row_b, overlap_y):
    return _metrics_for_offset(row_a, row_b, row_a.shape[0] - overlap_y, 0.0)


def _metrics_registered_v(row_a, row_b, dy, dx, overlap_y):
    return _metrics_for_offset(row_a, row_b, row_a.shape[0] - overlap_y + dy, dx)


# ─────────────────────────────────────────────
#  STAT CREATION
# ─────────────────────────────────────────────

def _overlap_stat_h(img_a, img_b, overlap_x, dy, dx, ov, r, c):
    mse_before, mi_before = _metrics_naive_h(img_a, img_b, overlap_x)
    mse_after,  mi_after  = _metrics_registered_h(img_a, img_b, dy, dx, ov)
    return OverlapStat(
        pair=f"H r{r} c{c-1}-{c}",
        mse_before=mse_before, mse_after=mse_after,
        mi_before=mi_before,   mi_after=mi_after,
    )


def _overlap_stat_v(row_a, row_b, overlap_y, dy, dx, ov, r):
    mse_before, mi_before = _metrics_naive_v(row_a, row_b, overlap_y)
    mse_after,  mi_after  = _metrics_registered_v(row_a, row_b, dy, dx, ov)
    return OverlapStat(
        pair=f"V rows {r-1}-{r} (full rows)",
        mse_before=mse_before, mse_after=mse_after,
        mi_before=mi_before,   mi_after=mi_after,
    )


# ─────────────────────────────────────────────
#  FULL-SLICE STITCHING
# ─────────────────────────────────────────────

def stitch_full_slice(tiles, z, ch, n_rows, n_cols,
                      overlap_x, overlap_y, plan, blend="average",
                      tile_h=None, tile_w=None, collect_stats=False,
                      # legacy kwargs accepted but ignored
                      global_thresh=None, mask_dir=None):
    stats: List[OverlapStat] = []

    print(f"\n  Building rows for ch={ch}, z={z} …")
    row_images = []
    for r in range(n_rows):
        row_img = _apply_row_shifts(tiles, z, ch, r, n_cols, overlap_x, plan, blend)
        row_images.append(row_img)

        if collect_stats and tile_h is not None and tile_w is not None:
            for c in range(1, n_cols):
                path_a = tiles.get((r, c-1, z, ch))
                path_b = tiles.get((r, c,   z, ch))
                if path_a is None or path_b is None:
                    continue
                img_a = load_tile(path_a)
                img_b = load_tile(path_b)
                dy, dx, ov = plan.h_shifts.get((r, c), (0.0, 0.0, overlap_x))
                stats.append(_overlap_stat_h(img_a, img_b, overlap_x, dy, dx, ov, r, c))

    print(f"\n  Stacking rows for ch={ch}, z={z} …")
    canvas = row_images[0]
    for r in range(1, n_rows):
        incoming = row_images[r]
        dy, dx, ov = plan.v_shifts.get(r, (0.0, 0.0, overlap_y))

        if collect_stats:
            stats.append(_overlap_stat_v(canvas, incoming, overlap_y, dy, dx, ov, r))

        canvas, incoming = _align_for_stitch(canvas, incoming, dy, dx, axis="v")
        canvas = stitch_pair(canvas, incoming, ov, axis="v", blend=blend)

    return canvas, stats


# ─────────────────────────────────────────────
#  NAIVE FULL-SLICE STITCHING
# ─────────────────────────────────────────────

def stitch_full_slice_naive(tiles, z, ch, n_rows, n_cols,
                            overlap_x, overlap_y, blend="average",
                            tile_h=None, tile_w=None,
                            collect_stats=False,
                            # legacy kwargs accepted but ignored
                            global_thresh=None, mask_dir=None):
    stats: List[OverlapStat] = []

    print(f"\n  Building rows (NAIVE) for ch={ch}, z={z} …")
    row_images = []
    for r in range(n_rows):
        accum = load_tile(tiles[(r, 0, z, ch)])
        for c in range(1, n_cols):
            incoming = load_tile(tiles[(r, c, z, ch)])
            if collect_stats:
                left_tile = load_tile(tiles[(r, c - 1, z, ch)])
                mse, mi = _metrics_naive_h(left_tile, incoming, overlap_x)
                stats.append(OverlapStat(
                    pair=f"H r{r} c{c-1}-{c}",
                    mse_before=mse, mse_after=mse,
                    mi_before=mi,   mi_after=mi,
                ))
            accum = stitch_pair(accum, incoming, overlap_x, axis="h", blend=blend)
        row_images.append(accum)

    print(f"\n  Stacking rows (NAIVE) for ch={ch}, z={z} …")
    canvas = row_images[0]
    for r in range(1, n_rows):
        incoming = row_images[r]
        if collect_stats:
            mse, mi = _metrics_naive_v(row_images[r - 1], row_images[r], overlap_y)
            stats.append(OverlapStat(
                pair=f"V rows {r-1}-{r} (full rows)",
                mse_before=mse, mse_after=mse,
                mi_before=mi,   mi_after=mi,
            ))
        canvas = stitch_pair(canvas, incoming, overlap_y, axis="v", blend=blend)

    return canvas, stats
>>>>>>> Stashed changes
