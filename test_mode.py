"""
test_mode.py — Monje Lab Stitcher
===================================
Test-mode diagnostics.  Four orientation modes are supported:

  horizontal   — single tile pair, horizontal join
  vertical     — single tile pair, vertical join
  row_vertical — stitch full rows then inspect the vertical join between two
                 adjacent rows
  full_slice   — stitch ALL rows for the Z slice, save each stitched row as a
                 TIFF, then produce side-by-side overlays for every vertical
                 join so the complete stitching sequence can be inspected.

Outputs per run
---------------
  Horizontal / Vertical (single pair):
    <tag>_NAIVE.png         — placement without correction
    <tag>_CORRECTED.png     — placement after PCC correction
    <tag>_STITCHED.tif      — blended stitch

  Row-vertical / Full-slice:
    row<N>_STITCHED.tif           — stitched row image (corrected)
    row<N>_NAIVE.tif              — stitched row image (naive/no-correction)
    join_r<N>-<N+1>_NAIVE.png     — overlay before correction
    join_r<N>-<N+1>_CORRECTED.png — overlay after correction

Stats table
-----------
Printed to stdout after every run.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from io_utils import load_tile, save_tiff
from registration import (
    estimate_shift_horizontal,
    estimate_shift_vertical,
    apply_shift_skimage,
    compute_mse,
    compute_mutual_information,
)
from blending import stitch_pair
from stitching import (
    _align_for_stitch,
    _metrics_naive_h,
    _metrics_registered_h,
    _metrics_naive_v,
    _metrics_registered_v,
)


# ─────────────────────────────────────────────
#  OVERLAY HELPER
# ─────────────────────────────────────────────

def save_rg_overlay(img_fixed, img_moving, out_path,
                    title="Alignment check", zoom_region=None):
    """
    Save a three-panel diagnostic PNG.

    Panels
    ------
    0 : False-colour composite  (red=fixed, green=moving, yellow=overlap)
    1 : Signed difference map
    2 : Zoomed crop of the seam region  (only when zoom_region is provided)
    """
    def _norm_u8(arr):
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
        return np.zeros_like(arr, dtype=np.uint8)

    r_ch = _norm_u8(img_fixed)
    g_ch = _norm_u8(img_moving)
    b_ch = np.zeros_like(r_ch)
    rgb  = np.stack([r_ch, g_ch, b_ch], axis=-1)

    n_panels = 3 if zoom_region is not None else 2
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(7 * n_panels, 6),
                             facecolor="#111")
    fig.suptitle(title, color="white", fontsize=13, y=1.01)

    axes[0].imshow(rgb)
    axes[0].set_title("Red=fixed | Green=moving | Yellow=overlap",
                      color="white", fontsize=9)
    axes[0].axis("off")

    if zoom_region is not None:
        y0, y1, x0, x1 = zoom_region
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                         linewidth=2, edgecolor="cyan", facecolor="none")
        axes[0].add_patch(rect)

    diff = img_fixed.astype(np.float32) - img_moving.astype(np.float32)
    vmax = np.percentile(np.abs(diff), 99) or 1.0
    axes[1].imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title("Difference map  (zero = perfect overlap)",
                      color="white", fontsize=9)
    axes[1].axis("off")

    if zoom_region is not None:
        y0, y1, x0, x1 = zoom_region
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                         linewidth=2, edgecolor="cyan", facecolor="none")
        axes[1].add_patch(rect)

        axes[2].imshow(rgb[y0:y1, x0:x1])
        axes[2].set_title(f"Zoom: seam region  ({x1-x0}×{y1-y0} px)",
                          color="white", fontsize=9)
        axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#111")
    plt.close(fig)
    print(f"  Overlay saved: {out_path}")


# ─────────────────────────────────────────────
#  METRIC HELPERS
# ─────────────────────────────────────────────

def _metrics_h(img_a, img_b, overlap_px):
    return _metrics_naive_h(img_a, img_b, overlap_px)


def _metrics_v(row_a, row_b, overlap_px):
    return _metrics_naive_v(row_a, row_b, overlap_px)


# ─────────────────────────────────────────────
#  STATS TABLE HELPER
# ─────────────────────────────────────────────

def _pct_change(before, after):
    if before == 0.0:
        return "     N/A"
    pct = (after - before) / before * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:7.1f}%"


def _print_stats_table(rows):
    col_w = 105
    print("\n" + "=" * col_w)
    print("  ALIGNMENT STATISTICS")
    print("  MSE = mean-squared error in the overlap zone  (lower = better)")
    print("  MI  = mutual information in the overlap zone  (higher = better)")
    print("  % chg = (after - before) / before × 100")
    print("=" * col_w)
    header = (
        f"  {'Pair':<28}  "
        f"{'MSE before':>11}  {'MSE after':>11}  {'MSE % chg':>10}  "
        f"{'MI before':>9}  {'MI after':>9}  {'MI % chg':>9}"
    )
    print(header)
    print("  " + "-" * (col_w - 2))
    for r in rows:
        print(
            f"  {r['pair']:<28}  "
            f"{r['mse_before']:>11.1f}  {r['mse_after']:>11.1f}  "
            f"{_pct_change(r['mse_before'], r['mse_after']):>10}  "
            f"{r['mi_before']:>9.4f}  {r['mi_after']:>9.4f}  "
            f"{_pct_change(r['mi_before'], r['mi_after']):>9}"
        )
    print("=" * col_w + "\n")


# ─────────────────────────────────────────────
#  CANVAS HELPERS FOR OVERLAYS
# ─────────────────────────────────────────────

def _pct_overlap_h_canvas(img_a, img_b, tile_h, tile_w, nom_dx):
    canvas_h = tile_h + 20
    canvas_w = tile_w + nom_dx + 20
    ca = np.zeros((canvas_h, canvas_w), np.float32)
    cb = np.zeros((canvas_h, canvas_w), np.float32)
    ca[:tile_h, :tile_w] = img_a
    x1 = min(nom_dx + tile_w, canvas_w)
    cb[:tile_h, nom_dx:x1] = img_b[:, :x1 - nom_dx]
    return ca, cb


def _pct_overlap_v_canvas(row_a, row_b, nominal_dy, dy=0.0, dx=0.0):
    rh_a, rw_a = row_a.shape
    rh_b, rw_b = row_b.shape
    canvas_h = rh_a + nominal_dy + abs(int(dy)) + 20
    canvas_w = max(rw_a, rw_b) + abs(int(dx)) + 20
    ca = np.zeros((canvas_h, canvas_w), np.float32)
    cb = np.zeros((canvas_h, canvas_w), np.float32)
    ca[:rh_a, :rw_a] = row_a
    if dy == 0.0 and dx == 0.0:
        y1 = min(nominal_dy + rh_b, canvas_h)
        cb[nominal_dy:y1, :rw_b] = row_b[:y1 - nominal_dy, :]
    else:
        cb = apply_shift_skimage(row_b, nominal_dy + dy, dx, (canvas_h, canvas_w))
    return ca, cb


# ─────────────────────────────────────────────
#  HORIZONTAL TEST
# ─────────────────────────────────────────────

def _test_horizontal(tiles, z, ref_ch,
                     overlap_x, tile_h, tile_w,
                     fudge, upsample, max_shift,
                     out_dir, row_idx, col_idx, blend="average",
                     bg_percentile=5, max_shift_frac=0.5,
                     # legacy
                     global_thresh=None):
    tag = f"H_r{row_idx}_c{col_idx}-{col_idx+1}_z{z:04d}"
    print(f"\n  TEST — HORIZONTAL  row={row_idx}, cols {col_idx}↔{col_idx+1}, Z={z}")

    path_a = tiles.get((row_idx, col_idx,     z, ref_ch))
    path_b = tiles.get((row_idx, col_idx + 1, z, ref_ch))
    if path_a is None or path_b is None:
        print("  Error: one or both test tiles are missing.")
        return

    img_a = load_tile(path_a)
    img_b = load_tile(path_b)
    print(f"  Tile A: {os.path.basename(path_a)}  {img_a.shape}")
    print(f"  Tile B: {os.path.basename(path_b)}  {img_b.shape}")

    dy, dx, ov = estimate_shift_horizontal(
        img_a, img_b, overlap_x,
        fudge=fudge, upsample=upsample, max_shift=max_shift,
        bg_percentile=bg_percentile, max_shift_frac=max_shift_frac,
    )
    nominal_dx = tile_w - overlap_x
    print(f"  Nominal dx={nominal_dx}  PCC correction: dy={dy:+.2f}, dx={dx:+.2f}")

    canvas_h = tile_h + abs(int(dy)) + 20
    canvas_w = tile_w + nominal_dx + abs(int(dx)) + 20

    zm = 300
    zoom = (
        max(0, canvas_h // 2 - zm), min(canvas_h, canvas_h // 2 + zm),
        max(0, tile_w - zm),        min(canvas_w, tile_w + zm),
    )

    # NAIVE overlay
    ca, cb = _pct_overlap_h_canvas(img_a, img_b, tile_h, tile_w, nominal_dx)
    ca_full = np.zeros((canvas_h, canvas_w), np.float32)
    cb_full = np.zeros((canvas_h, canvas_w), np.float32)
    ca_full[:ca.shape[0], :ca.shape[1]] = ca
    cb_full[:cb.shape[0], :cb.shape[1]] = cb
    save_rg_overlay(ca_full, cb_full,
                    os.path.join(out_dir, f"test_{tag}_NAIVE.png"),
                    title="NAIVE horizontal placement", zoom_region=zoom)

    mse_before, mi_before = _metrics_h(img_a, img_b, overlap_x)

    # CORRECTED overlay
    ca2 = np.zeros((canvas_h, canvas_w), np.float32)
    ca2[:tile_h, :tile_w] = img_a
    cb2 = apply_shift_skimage(img_b, dy, nominal_dx + dx, (canvas_h, canvas_w))
    save_rg_overlay(ca2, cb2,
                    os.path.join(out_dir, f"test_{tag}_CORRECTED.png"),
                    title="CORRECTED horizontal placement", zoom_region=zoom)

    mse_after, mi_after = _metrics_registered_h(img_a, img_b, dy, dx, ov)

    # Stitched TIFF
    img_a_p, img_b_p = _align_for_stitch(img_a, img_b, dy, dx, axis="h")
    stitched = stitch_pair(img_a_p, img_b_p, ov, axis="h", blend=blend)
    save_tiff(stitched, os.path.join(out_dir, f"test_{tag}_STITCHED.tif"))
    print(f"\n  Outputs written to: {out_dir}")

    _print_stats_table([{
        "pair":       tag,
        "mse_before": mse_before, "mse_after":  mse_after,
        "mi_before":  mi_before,  "mi_after":   mi_after,
    }])


# ─────────────────────────────────────────────
#  VERTICAL TEST  (single tile pair)
# ─────────────────────────────────────────────

def _test_vertical(tiles, z, ref_ch,
                   overlap_y, tile_h, tile_w,
                   fudge, upsample, max_shift,
                   out_dir, row_idx, col_idx, blend="average",
                   bg_percentile=5, max_shift_frac=0.5,
                   global_thresh=None):
    tag = f"V_c{col_idx}_r{row_idx}-{row_idx+1}_z{z:04d}"
    print(f"\n  TEST — VERTICAL  col={col_idx}, rows {row_idx}↔{row_idx+1}, Z={z}")

    path_a = tiles.get((row_idx,     col_idx, z, ref_ch))
    path_b = tiles.get((row_idx + 1, col_idx, z, ref_ch))
    if path_a is None or path_b is None:
        print("  Error: one or both test tiles are missing.")
        return

    img_a = load_tile(path_a)
    img_b = load_tile(path_b)
    print(f"  Tile A: {os.path.basename(path_a)}  {img_a.shape}")
    print(f"  Tile B: {os.path.basename(path_b)}  {img_b.shape}")

    dy, dx, ov = estimate_shift_vertical(
        img_a, img_b, overlap_y,
        fudge=fudge, upsample=upsample, max_shift=max_shift,
        bg_percentile=bg_percentile, max_shift_frac=max_shift_frac,
    )
    nominal_dy = tile_h - overlap_y
    print(f"  Nominal dy={nominal_dy}  PCC correction: dy={dy:+.2f}, dx={dx:+.2f}")

    canvas_h = tile_h + nominal_dy + abs(int(dy)) + 20
    canvas_w = tile_w + abs(int(dx)) + 20

    zm = 300
    zoom = (
        max(0, tile_h - zm), min(canvas_h, tile_h + zm),
        max(0, canvas_w // 2 - zm), min(canvas_w, canvas_w // 2 + zm),
    )

    # NAIVE overlay
    ca = np.zeros((canvas_h, canvas_w), np.float32)
    cb = np.zeros((canvas_h, canvas_w), np.float32)
    ca[:tile_h, :tile_w] = img_a
    y1 = min(nominal_dy + tile_h, canvas_h)
    cb[nominal_dy:y1, :tile_w] = img_b[:y1 - nominal_dy, :]
    save_rg_overlay(ca, cb,
                    os.path.join(out_dir, f"test_{tag}_NAIVE.png"),
                    title="NAIVE vertical placement", zoom_region=zoom)

    mse_before, mi_before = _metrics_v(img_a, img_b, overlap_y)

    # CORRECTED overlay
    ca2 = np.zeros((canvas_h, canvas_w), np.float32)
    ca2[:tile_h, :tile_w] = img_a
    cb2 = apply_shift_skimage(img_b, nominal_dy + dy, dx, (canvas_h, canvas_w))
    save_rg_overlay(ca2, cb2,
                    os.path.join(out_dir, f"test_{tag}_CORRECTED.png"),
                    title="CORRECTED vertical placement", zoom_region=zoom)

    mse_after, mi_after = _metrics_registered_v(img_a, img_b, dy, dx, ov)

    # Stitched TIFF
    img_a_p, img_b_p = _align_for_stitch(img_a, img_b, dy, dx, axis="v")
    stitched = stitch_pair(img_a_p, img_b_p, ov, axis="v", blend=blend)
    save_tiff(stitched, os.path.join(out_dir, f"test_{tag}_STITCHED.tif"))
    print(f"\n  Outputs written to: {out_dir}")

    _print_stats_table([{
        "pair":       tag,
        "mse_before": mse_before, "mse_after":  mse_after,
        "mi_before":  mi_before,  "mi_after":   mi_after,
    }])


# ─────────────────────────────────────────────
#  ROW-VERTICAL TEST
# ─────────────────────────────────────────────

def _test_row_vertical(tiles, z, ref_ch,
                       overlap_x, overlap_y, tile_h, tile_w,
                       fudge, upsample, max_shift,
                       out_dir, row_idx, n_cols, blend="average",
                       bg_percentile=5, max_shift_frac=0.5,
                       global_thresh=None):
    from stitching import ShiftPlan, _apply_row_shifts

    tag = f"RV_r{row_idx}-{row_idx+1}_z{z:04d}"
    print(f"\n  TEST — ROW-VERTICAL  rows {row_idx}↔{row_idx+1}, Z={z}, {n_cols} cols")

    plan = ShiftPlan()
    stats_rows = []

    for r in [row_idx, row_idx + 1]:
        for c in range(1, n_cols):
            path_a = tiles.get((r, c - 1, z, ref_ch))
            path_b = tiles.get((r, c,     z, ref_ch))
            if path_a is None or path_b is None:
                plan.h_shifts[(r, c)] = (0.0, 0.0, overlap_x)
                continue

            img_a = load_tile(path_a)
            img_b = load_tile(path_b)

            mse_before, mi_before = _metrics_h(img_a, img_b, overlap_x)

            dy, dx, ov = estimate_shift_horizontal(
                img_a, img_b, overlap_x,
                fudge=fudge, upsample=upsample, max_shift=max_shift,
                bg_percentile=bg_percentile, max_shift_frac=max_shift_frac,
            )
            plan.h_shifts[(r, c)] = (dy, dx, ov)

            mse_after, mi_after = _metrics_registered_h(img_a, img_b, dy, dx, ov)

            stats_rows.append({
                "pair":       f"H r{r} c{c-1}-{c}",
                "mse_before": mse_before, "mse_after": mse_after,
                "mi_before":  mi_before,  "mi_after":  mi_after,
            })

    # Stitch the two full rows
    print(f"\n  Stitching row {row_idx} (corrected) …")
    row_a = _apply_row_shifts(tiles, z, ref_ch, row_idx,   n_cols, overlap_x, plan, blend)
    print(f"\n  Stitching row {row_idx+1} (corrected) …")
    row_b = _apply_row_shifts(tiles, z, ref_ch, row_idx+1, n_cols, overlap_x, plan, blend)

    def _stitch_row_naive(row_r):
        path = tiles.get((row_r, 0, z, ref_ch))
        if path is None:
            raise ValueError(f"Missing tile (r={row_r}, c=0, z={z}, ch={ref_ch})")
        accum = load_tile(path)
        for c in range(1, n_cols):
            p = tiles.get((row_r, c, z, ref_ch))
            if p is None:
                continue
            accum = stitch_pair(accum, load_tile(p), overlap_x, axis="h", blend=blend)
        return accum

    row_a_naive = _stitch_row_naive(row_idx)
    row_b_naive = _stitch_row_naive(row_idx + 1)

    save_tiff(row_a,       os.path.join(out_dir, f"test_{tag}_row{row_idx}_STITCHED.tif"))
    save_tiff(row_b,       os.path.join(out_dir, f"test_{tag}_row{row_idx+1}_STITCHED.tif"))
    save_tiff(row_a_naive, os.path.join(out_dir, f"test_{tag}_row{row_idx}_NAIVE.tif"))
    save_tiff(row_b_naive, os.path.join(out_dir, f"test_{tag}_row{row_idx+1}_NAIVE.tif"))

    # Estimate vertical shift
    dy, dx, ov = estimate_shift_vertical(
        row_a, row_b, overlap_y,
        fudge=fudge, upsample=upsample, max_shift=max_shift,
        bg_percentile=bg_percentile, max_shift_frac=max_shift_frac,
    )
    nominal_dy = tile_h - overlap_y
    print(f"  Nominal dy={nominal_dy}  PCC correction: dy={dy:+.2f}, dx={dx:+.2f}")

    rh_a, rw_a = row_a.shape
    rh_b, rw_b = row_b.shape
    canvas_h = rh_a + nominal_dy + abs(int(dy)) + 20
    canvas_w = max(rw_a, rw_b) + abs(int(dx)) + 20

    zm = 400
    zoom = (
        max(0, rh_a - zm), min(canvas_h, rh_a + zm),
        max(0, canvas_w // 2 - zm), min(canvas_w, canvas_w // 2 + zm),
    )

    # NAIVE overlay
    ca = np.zeros((canvas_h, canvas_w), np.float32)
    cb = np.zeros((canvas_h, canvas_w), np.float32)
    ca[:rh_a, :rw_a] = row_a
    y1 = min(nominal_dy + rh_b, canvas_h)
    cb[nominal_dy:y1, :rw_b] = row_b[:y1 - nominal_dy, :]
    save_rg_overlay(ca, cb,
                    os.path.join(out_dir, f"test_{tag}_NAIVE.png"),
                    title=f"NAIVE row-vertical placement  (rows {row_idx}↔{row_idx+1})",
                    zoom_region=zoom)
    mse_v_before, mi_v_before = _metrics_v(row_a, row_b, overlap_y)

    # CORRECTED overlay
    ca2 = np.zeros((canvas_h, canvas_w), np.float32)
    ca2[:rh_a, :rw_a] = row_a
    cb2 = apply_shift_skimage(row_b, nominal_dy + dy, dx, (canvas_h, canvas_w))
    save_rg_overlay(ca2, cb2,
                    os.path.join(out_dir, f"test_{tag}_CORRECTED.png"),
                    title=f"CORRECTED row-vertical placement  (rows {row_idx}↔{row_idx+1})",
                    zoom_region=zoom)
    mse_v_after, mi_v_after = _metrics_registered_v(row_a, row_b, dy, dx, ov)

    # Stitched TIFF
    row_a_p, row_b_p = _align_for_stitch(row_a, row_b, dy, dx, axis="v")
    stitched = stitch_pair(row_a_p, row_b_p, ov, axis="v", blend=blend)
    save_tiff(stitched, os.path.join(out_dir, f"test_{tag}_STITCHED.tif"))
    print(f"\n  Outputs written to: {out_dir}")

    stats_rows.append({
        "pair":       f"V rows {row_idx}-{row_idx+1} (full rows)",
        "mse_before": mse_v_before, "mse_after": mse_v_after,
        "mi_before":  mi_v_before,  "mi_after":  mi_v_after,
    })
    _print_stats_table(stats_rows)


# ─────────────────────────────────────────────
#  FULL-SLICE TEST
# ─────────────────────────────────────────────

def _test_full_slice(tiles, z, ref_ch,
                     overlap_x, overlap_y, tile_h, tile_w,
                     fudge, upsample, max_shift,
                     out_dir, n_rows, n_cols, blend="average",
                     bg_percentile=5, max_shift_frac=0.5,
                     global_thresh=None):
    """
    Stitch every row for this Z slice (corrected and naive), save all row
    TIFFs, produce overlays for every consecutive vertical join.
    """
    from stitching import ShiftPlan, _apply_row_shifts

    print(f"\n  TEST — FULL SLICE  Z={z}, {n_rows} rows × {n_cols} cols")

    plan = ShiftPlan()
    stats_rows = []

    # Horizontal shifts for all rows
    for r in range(n_rows):
        for c in range(1, n_cols):
            path_a = tiles.get((r, c - 1, z, ref_ch))
            path_b = tiles.get((r, c,     z, ref_ch))
            if path_a is None or path_b is None:
                plan.h_shifts[(r, c)] = (0.0, 0.0, overlap_x)
                continue

            img_a = load_tile(path_a)
            img_b = load_tile(path_b)

            mse_before, mi_before = _metrics_h(img_a, img_b, overlap_x)

            dy, dx, ov = estimate_shift_horizontal(
                img_a, img_b, overlap_x,
                fudge=fudge, upsample=upsample, max_shift=max_shift,
                bg_percentile=bg_percentile, max_shift_frac=max_shift_frac,
            )
            plan.h_shifts[(r, c)] = (dy, dx, ov)

            mse_after, mi_after = _metrics_registered_h(img_a, img_b, dy, dx, ov)
            stats_rows.append({
                "pair":       f"H r{r} c{c-1}-{c}",
                "mse_before": mse_before, "mse_after": mse_after,
                "mi_before":  mi_before,  "mi_after":  mi_after,
            })

    # Stitch all rows
    corrected_rows = []
    naive_rows     = []

    def _stitch_row_naive(row_r):
        path = tiles.get((row_r, 0, z, ref_ch))
        if path is None:
            raise ValueError(f"Missing tile (r={row_r}, c=0, z={z}, ch={ref_ch})")
        accum = load_tile(path)
        for c in range(1, n_cols):
            p = tiles.get((row_r, c, z, ref_ch))
            if p is None:
                continue
            accum = stitch_pair(accum, load_tile(p), overlap_x, axis="h", blend=blend)
        return accum

    for r in range(n_rows):
        print(f"\n  Stitching row {r} (corrected) …")
        row_corr = _apply_row_shifts(tiles, z, ref_ch, r, n_cols,
                                     overlap_x, plan, blend)
        corrected_rows.append(row_corr)
        save_tiff(row_corr,
                  os.path.join(out_dir, f"test_FS_z{z:04d}_row{r:02d}_STITCHED.tif"))

        print(f"  Stitching row {r} (naive) …")
        row_naive = _stitch_row_naive(r)
        naive_rows.append(row_naive)
        save_tiff(row_naive,
                  os.path.join(out_dir, f"test_FS_z{z:04d}_row{r:02d}_NAIVE.tif"))

    # Vertical join overlays
    for r in range(1, n_rows):
        row_a = corrected_rows[r - 1]
        row_b = corrected_rows[r]

        dy, dx, ov = estimate_shift_vertical(
            row_a, row_b, overlap_y,
            fudge=fudge, upsample=upsample, max_shift=max_shift,
            bg_percentile=bg_percentile, max_shift_frac=max_shift_frac,
        )
        nominal_dy = tile_h - overlap_y

        rh_a, rw_a = row_a.shape
        rh_b, rw_b = row_b.shape
        canvas_h = rh_a + nominal_dy + abs(int(dy)) + 20
        canvas_w = max(rw_a, rw_b) + abs(int(dx)) + 20

        zm = 400
        zoom = (
            max(0, rh_a - zm), min(canvas_h, rh_a + zm),
            max(0, canvas_w // 2 - zm), min(canvas_w, canvas_w // 2 + zm),
        )

        # NAIVE
        ca = np.zeros((canvas_h, canvas_w), np.float32)
        cb = np.zeros((canvas_h, canvas_w), np.float32)
        ca[:rh_a, :rw_a] = row_a
        y1 = min(nominal_dy + rh_b, canvas_h)
        cb[nominal_dy:y1, :rw_b] = row_b[:y1 - nominal_dy, :]
        save_rg_overlay(
            ca, cb,
            os.path.join(out_dir, f"test_FS_z{z:04d}_join_r{r-1:02d}-{r:02d}_NAIVE.png"),
            title=f"NAIVE vertical join  rows {r-1}↔{r}  Z={z}",
            zoom_region=zoom,
        )
        mse_v_before, mi_v_before = _metrics_v(row_a, row_b, overlap_y)

        # CORRECTED
        ca2 = np.zeros((canvas_h, canvas_w), np.float32)
        ca2[:rh_a, :rw_a] = row_a
        cb2 = apply_shift_skimage(row_b, nominal_dy + dy, dx, (canvas_h, canvas_w))
        save_rg_overlay(
            ca2, cb2,
            os.path.join(out_dir, f"test_FS_z{z:04d}_join_r{r-1:02d}-{r:02d}_CORRECTED.png"),
            title=f"CORRECTED vertical join  rows {r-1}↔{r}  Z={z}",
            zoom_region=zoom,
        )
        mse_v_after, mi_v_after = _metrics_registered_v(row_a, row_b, dy, dx, ov)

        plan.v_shifts[r] = (dy, dx, ov)
        stats_rows.append({
            "pair":       f"V rows {r-1}-{r} (full rows)",
            "mse_before": mse_v_before, "mse_after": mse_v_after,
            "mi_before":  mi_v_before,  "mi_after":  mi_v_after,
        })

    # Final stitched slice TIFF
    canvas = corrected_rows[0]
    for r in range(1, n_rows):
        dy, dx, ov = plan.v_shifts.get(r, (0.0, 0.0, overlap_y))
        canvas, incoming = _align_for_stitch(canvas, corrected_rows[r], dy, dx, axis="v")
        canvas = stitch_pair(canvas, incoming, ov, axis="v", blend=blend)
    save_tiff(canvas, os.path.join(out_dir, f"test_FS_z{z:04d}_FULL_SLICE.tif"))

    print(f"\n  Outputs written to: {out_dir}")
    _print_stats_table(stats_rows)


# ─────────────────────────────────────────────
#  PUBLIC ENTRY POINT
# ─────────────────────────────────────────────

def run_test(tiles, z, ref_ch,
             overlap_x, overlap_y,
             tile_h, tile_w,
             fudge, upsample, max_shift,
             out_dir,
             row_idx=0, col_idx=0,
             n_rows=None, n_cols=None,
             orientation="horizontal",
             blend="average",
             bg_percentile=5,
             max_shift_frac=0.5,
             # legacy
             global_thresh=None):
    """
    Dispatch to the appropriate test helper based on *orientation*.

    Parameters
    ----------
    bg_percentile  : percentile for continuous background subtraction (default 5)
    max_shift_frac : max allowed shift as a fraction of tile size (default 0.5)

    orientation options
    -------------------
    "horizontal"   — single tile pair, horizontal join
    "vertical"     — single tile pair, vertical join
    "row_vertical" — stitch full rows then inspect the vertical join
    "full_slice"   — stitch ALL rows and produce overlays for every join
    """
    print("\n" + "=" * 60)
    kw = dict(bg_percentile=bg_percentile, max_shift_frac=max_shift_frac)
    if orientation == "horizontal":
        _test_horizontal(tiles, z, ref_ch,
                         overlap_x, tile_h, tile_w,
                         fudge, upsample, max_shift,
                         out_dir, row_idx, col_idx, blend, **kw)
    elif orientation == "vertical":
        _test_vertical(tiles, z, ref_ch,
                       overlap_y, tile_h, tile_w,
                       fudge, upsample, max_shift,
                       out_dir, row_idx, col_idx, blend, **kw)
    elif orientation == "row_vertical":
        if n_cols is None:
            print("  Error: n_cols required for row_vertical test mode.")
        else:
            _test_row_vertical(tiles, z, ref_ch,
                               overlap_x, overlap_y, tile_h, tile_w,
                               fudge, upsample, max_shift,
                               out_dir, row_idx, n_cols, blend, **kw)
    elif orientation == "full_slice":
        if n_rows is None or n_cols is None:
            print("  Error: n_rows and n_cols required for full_slice test mode.")
        else:
            _test_full_slice(tiles, z, ref_ch,
                             overlap_x, overlap_y, tile_h, tile_w,
                             fudge, upsample, max_shift,
                             out_dir, n_rows, n_cols, blend, **kw)
    else:
        print(f"  Error: unknown orientation {orientation!r}")
    print("=" * 60)
    print("\nTest done. Review the PNGs and TIFFs, then re-run with --mode real.")
