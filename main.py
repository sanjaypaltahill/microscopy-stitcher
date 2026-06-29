"""
main.py — Monje Lab Stitcher  (entry point)
============================================
Two modes:

  test  — Diagnostic stitching (calls test_mode.py).  Choose an orientation
          (horizontal / vertical / row_vertical / full_slice) to inspect
          individual joins before committing to a full stitch.

  real  — Full volume stitching.  Computes PCC shifts on the reference
          channel once, then applies identical transforms to every channel.

Registration approach
---------------------
Each tile pair undergoes continuous background subtraction (percentile-based,
controlled by --bg_percentile) before PCC runs on the FULL tile images — no
overlap strip extraction, no binary masking.

The returned shift is accepted only if both components are within
--max_shift_frac × the relevant tile dimension (default 50 %).  If the
primary PCC result exceeds this bound, progressively coarser upsample
factors are tried, then a downsampled full-frame fallback.  If nothing passes
the bound, the shift is set to (0, 0).

Run with --help for full usage information.
"""

import os
import sys
import argparse
import numpy as np

from io_utils import discover_tiles, grid_dims, load_tile, save_tiff
from stitching import (
    compute_shifts,
    stitch_full_slice,
    stitch_full_slice_naive,
    print_stats_table,
    OverlapStat,
)
from test_mode import run_test
from visualization import save_grid_overlay


# ─────────────────────────────────────────────
#  ARG PARSING
# ─────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="Monje Lab tile stitcher — PCC with background subtraction and shift constraints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Required ─────────────────────────────
    p.add_argument("--input_dir",  required=True,
                   help="Folder containing .ome.tif tiles")
    p.add_argument("--overlap",    type=int, required=True,
                   help="Nominal overlap as %% of tile size (e.g. 10 = 10%%)")

    # ── Optional I/O ─────────────────────────
    p.add_argument("--output_dir", default=None,
                   help="Root folder for all outputs (default: same as input_dir)")

    # ── Mode ─────────────────────────────────
    p.add_argument("--mode", choices=["test", "real"], default="real",
                   help="'test' = diagnostic outputs; 'real' = full stitch")

    # ── Real-mode options ─────────────────────
    p.add_argument("--visualize", action="store_true",
                   help="Save grid_overlays/ PNGs showing nominal tile positions")
    p.add_argument("--blend", choices=["average", "sinusoidal"], default="average",
                   help="Blend kernel for the overlap zone")

    # ── Registration tuning ───────────────────
    p.add_argument("--max_shift", type=int, default=500,
                   help="Absolute PCC shift cap in pixels — secondary guard on top of "
                        "--max_shift_frac (default: 500)")
    p.add_argument("--max_shift_frac", type=float, default=0.5,
                   help="Maximum allowed shift as a fraction of the tile dimension "
                        "(default: 0.5 = 50%%).  Shifts exceeding this are rejected and "
                        "the next-best PCC estimate is tried.")
    p.add_argument("--bg_percentile", type=float, default=5.0,
                   help="Percentile used for per-tile background subtraction before PCC "
                        "(default: 5).  Lower values remove less background; higher values "
                        "are more aggressive.")
    p.add_argument("--fudge",     type=int, default=0,
                   help="Legacy registration option kept for compatibility (default: 0)")
    p.add_argument("--ref_channel", type=int, default=None,
                   help="Channel used for registration (default: first found)")
    p.add_argument("--upsample",  type=int, default=10,
                   help="PCC upsample factor; sub-pixel accuracy = 1/upsample (default: 10)")
    p.add_argument("--no_multiscale", action="store_true",
                   help="Disable the full-frame fallback PCC (faster but fails on large shifts)")

    # ── Slice / tile selection ────────────────
    p.add_argument("--z_slice", type=int, default=None,
                   help="Pin to a single Z index; omit to process every Z")

    # ── Test-mode options ─────────────────────
    p.add_argument("--orientation",
                   choices=["horizontal", "vertical", "row_vertical", "full_slice"],
                   default="horizontal",
                   help="Test mode only: which join(s) to inspect")
    p.add_argument("--row_idx", type=int, default=0,
                   help="Test mode: row of the anchor tile (default: 0)")
    p.add_argument("--col_idx", type=int, default=0,
                   help="Test mode: column of the anchor tile (default: 0)")

    return p


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    args = build_parser().parse_args()

    root_out = args.output_dir or args.input_dir
    frac     = args.overlap / 100.0

    # ── Discover tiles ─────────────────────────────────────────────
    tiles, zs, chs, prefix = discover_tiles(args.input_dir)
    ref_ch = args.ref_channel if args.ref_channel is not None else chs[0]

    # ── Z selection ────────────────────────────────────────────────
    if args.z_slice is not None:
        if args.z_slice not in zs:
            sys.exit(f"Z {args.z_slice} not found")
        work_z  = args.z_slice
        work_zs = [args.z_slice]
    else:
        work_z  = zs[0]
        work_zs = zs

    sample_path = next(
        (v for (r, c, z, ch), v in tiles.items() if z == work_z and ch == ref_ch),
        None,
    )
    if sample_path is None:
        sys.exit("No sample tile found")

    sample         = load_tile(sample_path)
    tile_h, tile_w = sample.shape

    ov_x = max(1, int(round(frac * tile_w)))
    ov_y = max(1, int(round(frac * tile_h)))

    n_rows, n_cols = grid_dims(tiles, work_z)
    multiscale     = not args.no_multiscale

    # ── Print run summary ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Monje Lab — Stitcher")
    print("=" * 60)
    print(f"  Mode            : {args.mode}")
    print(f"  Z slice(s)      : {work_zs}")
    print(f"  Grid            : {n_rows} × {n_cols}")
    print(f"  Tile size       : {tile_h} × {tile_w}")
    print(f"  Overlap         : {args.overlap}%  ({ov_x}px H, {ov_y}px V)")
    print(f"  Orientation     : {args.orientation}")
    print(f"  Blend           : {args.blend}")
    print(f"  Ref channel     : {ref_ch}")
    print(f"  Max shift (abs) : {args.max_shift} px")
    print(f"  Max shift (frac): {args.max_shift_frac:.0%}  "
          f"({args.max_shift_frac * tile_w:.0f}px H, "
          f"{args.max_shift_frac * tile_h:.0f}px V)")
    print(f"  BG percentile   : {args.bg_percentile}")
    print(f"  Fudge           : {args.fudge} px")
    print(f"  Upsample        : {args.upsample}")
    print(f"  Multiscale      : {multiscale}")
    print("=" * 60)

    out_dir = os.path.join(root_out, (prefix or "output") + "_registered")
    os.makedirs(out_dir, exist_ok=True)

    # ─────────────────────────────────────────
    # TEST MODE
    # ─────────────────────────────────────────
    if args.mode == "test":
        run_test(
            tiles=tiles,
            z=work_z,
            ref_ch=ref_ch,
            overlap_x=ov_x,
            overlap_y=ov_y,
            tile_h=tile_h,
            tile_w=tile_w,
            fudge=args.fudge,
            upsample=args.upsample,
            max_shift=args.max_shift,
            out_dir=out_dir,
            row_idx=args.row_idx,
            col_idx=args.col_idx,
            n_rows=n_rows,
            n_cols=n_cols,
            orientation=args.orientation,
            blend=args.blend,
            bg_percentile=args.bg_percentile,
            max_shift_frac=args.max_shift_frac,
        )
        print("\nTest complete.")
        return

    # ─────────────────────────────────────────
    # REAL MODE
    # ─────────────────────────────────────────
    ch_dirs       = {}
    ch_dirs_naive = {}

    for ch in chs:
        d       = os.path.join(out_dir, f"Channel_{ch:02d}")
        d_naive = os.path.join(out_dir, f"Channel_{ch:02d}_NAIVE")
        os.makedirs(d,       exist_ok=True)
        os.makedirs(d_naive, exist_ok=True)
        ch_dirs[ch]       = d
        ch_dirs_naive[ch] = d_naive

    for z in work_zs:
        print(f"\n===== Z={z} =====")

        n_rows_z, n_cols_z = grid_dims(tiles, z)

        # Compute PCC shifts once on the reference channel.
        plan = compute_shifts(
            tiles, z, ref_ch,
            n_rows_z, n_cols_z,
            ov_x, ov_y,
            fudge=args.fudge,
            upsample=args.upsample,
            max_shift=args.max_shift,
            multiscale=multiscale,
            bg_percentile=args.bg_percentile,
            max_shift_frac=args.max_shift_frac,
        )

        for ch in chs:
            print(f"\nChannel {ch}")

            img_naive, naive_stats = stitch_full_slice_naive(
                tiles, z, ch,
                n_rows_z, n_cols_z,
                ov_x, ov_y,
                blend=args.blend,
                tile_h=tile_h,
                tile_w=tile_w,
                collect_stats=(ch == ref_ch),
            )

            img_corr, corr_stats = stitch_full_slice(
                tiles, z, ch,
                n_rows_z, n_cols_z,
                ov_x, ov_y,
                blend=args.blend,
                plan=plan,
                tile_h=tile_h,
                tile_w=tile_w,
                collect_stats=(ch == ref_ch),
            )

            save_tiff(img_naive,
                      os.path.join(ch_dirs_naive[ch], f"z{z:04d}_NAIVE.tif"))
            save_tiff(img_corr,
                      os.path.join(ch_dirs[ch], f"z{z:04d}_CORRECTED.tif"))

            if ch == ref_ch:
                merged = []
                for naive, corr in zip(naive_stats, corr_stats):
                    merged.append(OverlapStat(
                        pair=naive.pair,
                        mse_before=naive.mse_before,
                        mse_after=corr.mse_after,
                        mi_before=naive.mi_before,
                        mi_after=corr.mi_after,
                    ))
                print_stats_table(merged)

        if args.visualize:
            vis_dir = os.path.join(out_dir, "grid_overlays")
            os.makedirs(vis_dir, exist_ok=True)
            pos = _build_nominal_positions(n_rows_z, n_cols_z, tile_h, tile_w,
                                           ov_x, ov_y)
            save_grid_overlay(pos, tile_h, tile_w, n_rows_z, n_cols_z, z,
                               os.path.join(vis_dir, f"grid_z{z:04d}.png"))

    print("\nDone.")


# ─────────────────────────────────────────────
#  HELPER: NOMINAL POSITION ARRAY
# ─────────────────────────────────────────────

def _build_nominal_positions(n_rows, n_cols, tile_h, tile_w, ov_x, ov_y):
    pos = np.zeros((n_rows, n_cols, 2), dtype=np.float32)
    step_x = tile_w - ov_x
    step_y = tile_h - ov_y
    for r in range(n_rows):
        for c in range(n_cols):
            pos[r, c, 0] = r * step_y
            pos[r, c, 1] = c * step_x
    return pos


if __name__ == "__main__":
    main()
