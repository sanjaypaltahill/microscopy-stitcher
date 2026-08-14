#!/usr/bin/env python3
"""Pad a folder of stitched z-slices to a common size.

Registered/stitched z-slices often come out a few pixels different in height and
width (the mosaic bounding box shifts a little from slice to slice). To stack
them into one volume every slice must share the same (height, width).

This script implements the PADDING strategy (never truncating): it finds the
largest height and width across the folder, then zero-pads every slice on its
BOTTOM and RIGHT edges up to that size. No pixel is ever discarded.

Usage:
    python padding.py INPUT_DIR OUTPUT_DIR
    python padding.py INPUT_DIR OUTPUT_DIR --glob '*_registered.tif'

It prints, per slice, how many rows/cols were added, and the maximum added over
the whole folder so you can sanity-check that the padding is small/reasonable.
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile


def read_hw(path):
    """(height, width) of a TIFF's first page, without loading the pixels."""
    with tifffile.TiffFile(str(path)) as tif:
        page = tif.pages[0]
        return int(page.shape[0]), int(page.shape[1])


def pad_to(img, target_hw):
    """Zero-pad the bottom & right edges up to target (height, width).

    Only the first two axes (height, width) are padded; any trailing axis
    (e.g. channels for an (H, W, C) image) is left untouched.
    """
    th, tw = target_hw
    h, w = img.shape[:2]
    if h > th or w > tw:
        raise ValueError(f"image {img.shape[:2]} is larger than pad target {target_hw}")
    pad = [(0, th - h), (0, tw - w)] + [(0, 0)] * (img.ndim - 2)
    return np.pad(img, pad, mode="constant", constant_values=0)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Zero-pad a folder of stitched z-slices to a common "
                    "(largest) size by adding rows/cols on the bottom & right.")
    ap.add_argument("input_dir",
                    help="Folder of stitched z-slice images to equalize")
    ap.add_argument("output_dir",
                    help="Folder to write the equal-size padded slices into "
                         "(created if needed)")
    ap.add_argument("--glob", default="*.tif",
                    help="Filename pattern to match inside input_dir "
                         "(default: '*.tif')")
    return ap.parse_args()


def main():
    p = parse_args()
    input_dir = Path(p.input_dir)
    output_dir = Path(p.output_dir)

    if not input_dir.is_dir():
        raise SystemExit(f"Input folder does not exist: {input_dir}")

    paths = sorted(input_dir.glob(p.glob))
    if not paths:
        raise SystemExit(f"No images matching '{p.glob}' in {input_dir}")

    # --- Pass 1: find the target (largest) size from headers only. ------------
    shapes = np.array([read_hw(pt) for pt in paths])   # (N, 2): height, width
    target_h, target_w = int(shapes[:, 0].max()), int(shapes[:, 1].max())

    print(f"Input folder : {input_dir}")
    print(f"Output folder: {output_dir}")
    print(f"Slices found : {len(paths)}  (matching '{p.glob}')")
    print(f"Height       : min={shapes[:, 0].min()}  max={target_h}")
    print(f"Width        : min={shapes[:, 1].min()}  max={target_w}")
    print(f"Pad target   : {target_h} rows x {target_w} cols\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Pass 2: pad every slice and write it out, preserving dtype/name. ------
    max_rows_added = 0
    max_cols_added = 0
    max_rows_file = max_cols_file = None

    for pt in paths:
        img = tifffile.imread(str(pt), is_ome=False)
        h, w = img.shape[:2]
        rows_added, cols_added = target_h - h, target_w - w

        padded = pad_to(img, (target_h, target_w))
        out_path = output_dir / pt.name
        tifffile.imwrite(str(out_path), padded)

        if rows_added > max_rows_added:
            max_rows_added, max_rows_file = rows_added, pt.name
        if cols_added > max_cols_added:
            max_cols_added, max_cols_file = cols_added, pt.name

        print(f"  {pt.name}: {h}x{w} -> {target_h}x{target_w}  "
              f"(+{rows_added} rows, +{cols_added} cols)")

    # --- Sanity-check summary. ------------------------------------------------
    print(f"\nDone. Wrote {len(paths)} equal-size slices to: {output_dir}")
    print(f"Most rows added to any slice: {max_rows_added} "
          f"({max_rows_file})")
    print(f"Most cols added to any slice: {max_cols_added} "
          f"({max_cols_file})")


if __name__ == "__main__":
    main()
