"""registrationQC.py

Quality-control dry run for registration.py. Does everything
full_slice_with_metrics.ipynb does -- the visual diagnostics (edge-focus box +
Hann window, tissue masks, naive-vs-registered R/G/Y overlays, the final
naive-vs-registered mosaic) AND the MSE / mutual-information seam scoring -- but
headless, on THREE representative z-slices instead of one hand-picked plane: the
25th, 50th and 75th percentile of the detected z-stack.

The point is to look before you leap: run this first, flip through the figures and
the metrics table, and only then turn registration.py loose on the whole volume.

Every registration decision comes from registration.py itself (this script imports
its helpers rather than re-implementing them), so what you QC here is exactly what
the full run will do. This script only ADDS the evaluation layer.

Results land in one folder per percentile under the output root:
    25th_percentile_QC/  50th_percentile_QC/  75th_percentile_QC/
each holding the figures, mosaic_registered.tif / mosaic_naive.tif, metrics.csv
and log.txt.

Run with the project python, e.g.:
    /opt/miniconda3/bin/python registrationQC.py "/Users/spaltahill/test_images"
    /opt/miniconda3/bin/python registrationQC.py <input_dir> --output-dir <out> \
        --reference-channel 0 --percentiles 25,50,75
"""

# -- Imports -------------------------------------------------------
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless: figures are written to disk, never shown
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tifffile
from skimage.filters import threshold_otsu

# The registration engine under test -- imported, never re-implemented, so a change
# to registration.py is immediately reflected in what this QC measures.
from registration import (composite_line, edge_focus_box, hann2d, load_grid,
                          norm_disp, register_column, register_row,
                          scan_input_folder, to_input_dtype)

matplotlib.rcParams['figure.dpi'] = 120
FIG_DPI = 120


# -- Output plumbing -----------------------------------------------

class _Tee:
    """Mirror stdout into a log file so each QC folder keeps its own console trace
    (the per-seam shift and metric printouts are half the value of the run)."""

    def __init__(self, path):
        self.file = open(path, 'w')
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def save_fig(fig, out_dir, name):
    """Write a figure into the QC folder and drop it (headless runs leak figures
    fast -- a 3x3 grid produces dozens)."""
    fig.savefig(Path(out_dir) / name, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)


# -- Display decimation --------------------------------------------
# Figures are saved at FIG_DPI into panels a few thousand pixels wide, so building
# them from full-resolution arrays allocates hundreds of MB of detail the renderer
# immediately throws away -- a vertical-seam overlay of two row images is a
# 4605x5507x3 float32 canvas (~300 MB), built twice per seam. Decimating first is
# invisible in the PNG and is what keeps peak memory in single-digit GB.
#
# This affects DISPLAY ONLY. Every MSE/MI number is computed from the full-
# resolution arrays; nothing here touches registration or the metrics.
DISPLAY_MAX_PX = 2400


def display_step(*imgs):
    """Integer stride that keeps a display canvas near DISPLAY_MAX_PX on its long
    axis. 1 (no decimation) for anything already small enough."""
    biggest = max(max(i.shape) for i in imgs)
    return max(1, int(np.ceil(biggest / DISPLAY_MAX_PX)))


def _disp(img, step=1):
    """Decimate then contrast-stretch for display. Order matters: norm_disp's
    percentile arithmetic promotes float32 to float64, so stretching first would
    allocate a double-precision copy of the FULL image."""
    return norm_disp(img[::step, ::step])


# -- Red/Green/Yellow overlay helpers ------------------------------
# A -> red channel, B -> green channel, so their overlap reads yellow.

def _overlay_canvas(img_a, img_b, b_y0, b_x0, step=1):
    """R/G overlay canvas, decimated by `step`. Placement offsets are scaled by the
    same factor, so the canvas is a faithful (just coarser) picture of the
    placement -- at these strides a 1 px shift error lands well under one output
    pixel either way."""
    a_d, b_d = _disp(img_a, step), _disp(img_b, step)
    h_a, w_a = a_d.shape
    h_b, w_b = b_d.shape
    b_y0, b_x0 = int(round(b_y0 / step)), int(round(b_x0 / step))
    oy, ox = max(0, -b_y0), max(0, -b_x0)           # shift so nothing is clipped
    H = max(h_a, b_y0 + h_b) + oy
    W = max(w_a, b_x0 + w_b) + ox
    ch_a = np.zeros((H, W), 'float32')
    ch_b = np.zeros((H, W), 'float32')
    ch_a[oy:oy+h_a, ox:ox+w_a] = a_d
    ch_b[b_y0+oy:b_y0+oy+h_b, b_x0+ox:b_x0+ox+w_b] = b_d
    return np.stack([ch_a, ch_b, np.zeros_like(ch_a)], -1)


def _legend(fig):
    fig.legend(handles=[mpatches.Patch(color=(1, 0, 0), label='A only'),
                        mpatches.Patch(color=(0, 1, 0), label='B only'),
                        mpatches.Patch(color=(1, 1, 0), label='Overlap (both)')],
               loc='upper right', fontsize=8, framealpha=0.6)


def compare_overlay_h(img_a, img_b, overlap, dy, dx, suptitle, out_dir, name):
    """Horizontal pair (A=left, B=right): B left edge at x = w_a - overlap + dx."""
    w_a = img_a.shape[1]
    s = display_step(img_a, img_b)
    naive = _overlay_canvas(img_a, img_b, 0, w_a - overlap, s)
    reg = _overlay_canvas(img_a, img_b, int(round(dy)),
                          w_a - overlap + int(round(dx)), s)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].imshow(naive, aspect='auto')
    axes[0].set_title('NAIVE  (dy=0, dx=0)', fontsize=9)
    axes[0].axis('off')
    axes[1].imshow(reg, aspect='auto')
    axes[1].set_title(f'REGISTERED  (dy={dy:+.1f}, dx={dx:+.1f})', fontsize=9)
    axes[1].axis('off')
    _legend(fig)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    save_fig(fig, out_dir, name)


def compare_overlay_v(img_a, img_b, overlap, dy, dx, suptitle, out_dir, name):
    """Vertical pair (A=top, B=bottom): B top edge at y = h_a - overlap + dy."""
    h_a = img_a.shape[0]
    s = display_step(img_a, img_b)
    naive = _overlay_canvas(img_a, img_b, h_a - overlap, 0, s)
    reg = _overlay_canvas(img_a, img_b, h_a - overlap + int(round(dy)),
                          int(round(dx)), s)
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    axes[0].imshow(naive, aspect='auto')
    axes[0].set_title('NAIVE  (dy=0, dx=0)', fontsize=9)
    axes[0].axis('off')
    axes[1].imshow(reg, aspect='auto')
    axes[1].set_title(f'REGISTERED  (dy={dy:+.1f}, dx={dx:+.1f})', fontsize=9)
    axes[1].axis('off')
    _legend(fig)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    save_fig(fig, out_dir, name)


# -- Per-seam diagnostic panels ------------------------------------
# These reproduce the notebook's inline figures. They are drawn AFTER registration
# from the shifts registration.py returned, re-deriving the edge-focus box with the
# same call the engine made -- so the panels show the real box, not a lookalike.

def plot_seam_box(a, b, overlap, lo, hi, axis, label, out_dir, prefix, thresh):
    """Two figures for one seam: (1) both images with the edge-focus box drawn on
    the overlap plus the Hann window that was applied inside it, and (2) the same
    images beside their tissue masks -- the sanity check that the box landed on
    tissue and not on background."""
    if axis == 'h':
        box_a = (a.shape[1] - overlap, lo, overlap, hi - lo)   # (x, y, w, h)
        box_b = (0, lo, overlap, hi - lo)
        win = hann2d(a[lo:hi, -overlap:].shape)
        aspect, kind = None, 'horizontal'
        edge_a, edge_b = 'right edge', 'left edge'
        unit = f'rows {lo}-{hi}'
    else:
        box_a = (lo, a.shape[0] - overlap, hi - lo, overlap)
        box_b = (lo, 0, hi - lo, overlap)
        win = hann2d(a[-overlap:, lo:hi].shape)
        aspect, kind = 'auto', 'vertical'
        edge_a, edge_b = 'bottom edge', 'top edge'
        unit = f'cols {lo}-{hi}'

    s = display_step(a, b)

    def _rect(box):
        return mpatches.Rectangle((box[0] / s, box[1] / s), box[2] / s, box[3] / s,
                                  fill=False, edgecolor='yellow', lw=1.5)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(_disp(a, s), cmap='gray', aspect=aspect)
    axes[0].add_patch(_rect(box_a))
    axes[0].set_title(f'{label} (A) - box on {edge_a}', fontsize=9)
    axes[0].axis('off')
    axes[1].imshow(_disp(b, s), cmap='gray', aspect=aspect)
    axes[1].add_patch(_rect(box_b))
    axes[1].set_title(f'{label} (B) - box on {edge_b}', fontsize=9)
    axes[1].axis('off')
    axes[2].imshow(win, cmap='viridis', aspect='auto')
    axes[2].set_title(f'Hann window ({win.shape[0]}x{win.shape[1]}) - applied before PCC',
                      fontsize=9)
    axes[2].axis('off')
    fig.suptitle(f'{label} - {kind} seam ({unit}): edge-focus box (yellow) + Hann window',
                 fontsize=10)
    fig.tight_layout()
    save_fig(fig, out_dir, f'{prefix}_box_hann.png')

    fig, axm = plt.subplots(1, 4, figsize=(20, 5))
    for k, (img, box, tag) in enumerate([(a, box_a, 'A'), (b, box_b, 'B')]):
        axm[2*k].imshow(_disp(img, s), cmap='gray', aspect=aspect)
        axm[2*k].add_patch(_rect(box))
        axm[2*k].set_title(f'{label} ({tag})', fontsize=9)
        axm[2*k].axis('off')
        axm[2*k+1].imshow(img[::s, ::s] > thresh, cmap='gray', aspect=aspect)
        axm[2*k+1].add_patch(_rect(box))
        axm[2*k+1].set_title(f'({tag}) tissue mask (> {thresh:.0f})', fontsize=9)
        axm[2*k+1].axis('off')
    fig.suptitle(f'{label} - {kind} seam: images vs tissue masks (sanity check)', fontsize=10)
    fig.tight_layout()
    save_fig(fig, out_dir, f'{prefix}_masks.png')


# -- Metrics: MSE + Mutual Information over a zoomed, masked overlap box --------
# Layered on top of the pipeline WITHOUT changing any registration. Each seam is
# scored twice:
#   * NAIVE      -- assumed overlap, dy=dx=0
#   * REGISTERED -- the measured (dy, dx)
# Two choices make the score trustworthy: evaluate inside the SAME edge-focus box
# used for the Hann window (zoomed to the tissue band, so empty background can't
# dilute it), and keep only foreground pixels (raw > thresh in EITHER image).
#
# Scoring is on RAW intensities on purpose. Registration optimised the HIGH-PASSED
# signal, so raw agreement is an INDEPENDENT check -- improvement there is genuine
# corroboration, not the circular result of grading on the optimiser's own
# objective. Raw MSE also carries the per-tile exposure gap, which makes raw MI the
# more trustworthy witness; both moving the right way (MSE down, MI up) is the
# validation you want.

def mutual_information(a_vals, b_vals, bins=64):
    """MI (nats) between two 1-D arrays of paired intensities."""
    a_vals = np.asarray(a_vals)
    b_vals = np.asarray(b_vals)
    if a_vals.size < 2:
        return float('nan')

    def to_bins(x):
        lo, hi = float(x.min()), float(x.max())
        if hi <= lo:
            return np.zeros(x.size, dtype=int)
        return np.clip(((x - lo) / (hi - lo) * bins).astype(int), 0, bins - 1)

    joint, _, _ = np.histogram2d(to_bins(a_vals), to_bins(b_vals),
                                 bins=bins, range=[[0, bins], [0, bins]])
    pxy = joint / (joint.sum() + 1e-12)
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    return float(np.sum(pxy[nz] * np.log(pxy[nz] / ((px * py)[nz] + 1e-12))))


def _overlap_box_h(a, b, overlap, r0, r1, dy, dx):
    """Aligned overlap index rectangles for a HORIZONTAL seam (A=left, B=right),
    restricted to the edge-focus box rows [r0:r1]. Returns (naiveA, naiveB, regA,
    regB), each an (row0, row1, col0, col1) tuple. Naive uses dy=dx=0."""
    h_a, w_a = a.shape
    h_b, w_b = b.shape
    idy, idx = int(round(dy)), int(round(dx))
    naive_A = (r0, r1, w_a - overlap, w_a)
    naive_B = (r0, r1, 0, overlap)
    b_x0 = max(0, w_a - overlap + idx)                 # B left edge on the A frame
    col_w = min(w_a - b_x0, w_b)                        # columns where both exist
    ca0, ca1 = b_x0, b_x0 + col_w
    cb0, cb1 = 0, col_w
    ov_a0, ov_a1 = max(0, idy), min(h_a, h_b + idy)    # rows where both exist (A=B+idy)
    ra0, ra1 = max(ov_a0, r0), min(ov_a1, r1)          # ... intersected with the box
    rb0, rb1 = ra0 - idy, ra1 - idy
    return naive_A, naive_B, (ra0, ra1, ca0, ca1), (rb0, rb1, cb0, cb1)


def _overlap_box_v(a, b, overlap, c0, c1, dy, dx):
    """Aligned overlap rectangles for a VERTICAL seam (A=top, B=bottom), restricted
    to the edge-focus box columns [c0:c1]. Mirror image of _overlap_box_h."""
    h_a, w_a = a.shape
    h_b, w_b = b.shape
    idy, idx = int(round(dy)), int(round(dx))
    naive_A = (h_a - overlap, h_a, c0, c1)
    naive_B = (0, overlap, c0, c1)
    b_y0 = max(0, h_a - overlap + idy)                 # B top edge on the A frame
    row_h = min(h_a - b_y0, h_b)                        # rows where both exist
    ra0, ra1 = b_y0, b_y0 + row_h
    rb0, rb1 = 0, row_h
    ov_a0, ov_a1 = max(0, idx), min(w_a, w_b + idx)    # cols where both exist (A=B+idx)
    ca0, ca1 = max(ov_a0, c0), min(ov_a1, c1)          # ... intersected with the box
    cb0, cb1 = ca0 - idx, ca1 - idx
    return naive_A, naive_B, (ra0, ra1, ca0, ca1), (rb0, rb1, cb0, cb1)


def _metric_panel(ax, a, b, b_y0, b_x0, boxA, boxB, thresh, title, step=1):
    """R/G/Y overlay (A=red, B=green) with the MSE box drawn in cyan and the scored
    background blacked out, so the panel shows exactly which pixels count. Every
    coordinate is divided by `step` to land on the decimated canvas; the foreground
    mask is rebuilt from the decimated crops (the crops are the overlap box, not the
    whole image, so cropping at full resolution first stays cheap)."""
    canvas = _overlay_canvas(a, b, b_y0, b_x0, step)
    sb_y0, sb_x0 = int(round(b_y0 / step)), int(round(b_x0 / step))
    oy, ox = max(0, -sb_y0), max(0, -sb_x0)            # _overlay_canvas offset for A
    r0, r1, c0, c1 = boxA
    rb0, rb1, cb0, cb1 = boxB
    ra = a[r0:r1, c0:c1][::step, ::step]
    rb = b[rb0:rb1, cb0:cb1][::step, ::step]
    sr0, sc0 = r0 // step, c0 // step
    hh = min(ra.shape[0], rb.shape[0])
    ww = min(ra.shape[1], rb.shape[1])
    if hh > 0 and ww > 0:
        fg = (ra[:hh, :ww] > thresh) | (rb[:hh, :ww] > thresh)
        sub = canvas[sr0 + oy:sr0 + oy + hh, sc0 + ox:sc0 + ox + ww, :]
        # The canvas edge can clip the box by a pixel after rounding; trim to fit.
        fg = fg[:sub.shape[0], :sub.shape[1]]
        sub[~fg] = 0.0                                 # black out scored background
    ax.imshow(canvas, aspect='auto')
    ax.add_patch(mpatches.Rectangle((sc0 + ox, sr0 + oy),
                                    (c1 - c0) / step, (r1 - r0) / step,
                                    fill=False, edgecolor='cyan', lw=2))
    ax.set_title(title, fontsize=9)
    ax.axis('off')


def score_seam(a, b, axis, overlap, box_lo, box_hi, dy, dx, thresh,
               label, out_dir, prefix):
    """Score one seam naive vs registered inside the zoomed, masked overlap box, and
    draw the panel showing which pixels counted. Returns a dict of the two metrics
    under both placements plus the foreground pixel pools (for the whole-image
    aggregate, which pools pixels rather than averaging per-seam numbers)."""
    if axis == 'h':
        nA, nB, rA, rB = _overlap_box_h(a, b, overlap, box_lo, box_hi, dy, dx)
        b_y0_n, b_x0_n = 0, a.shape[1] - overlap
        b_y0_r, b_x0_r = int(round(dy)), a.shape[1] - overlap + int(round(dx))
    else:
        nA, nB, rA, rB = _overlap_box_v(a, b, overlap, box_lo, box_hi, dy, dx)
        b_y0_n, b_x0_n = a.shape[0] - overlap, 0
        b_y0_r, b_x0_r = a.shape[0] - overlap + int(round(dy)), int(round(dx))

    def crop(arr, box):
        r0, r1, c0, c1 = box
        return arr[r0:r1, c0:c1]

    res = dict(label=label, axis=axis, dy=dy, dx=dx, overlap=overlap)
    for tag, boxA, boxB in [('n', nA, nB), ('r', rA, rB)]:
        ra, rb = crop(a, boxA), crop(b, boxB)
        hh = min(ra.shape[0], rb.shape[0])
        ww = min(ra.shape[1], rb.shape[1])
        mask = ((ra[:hh, :ww] > thresh) | (rb[:hh, :ww] > thresh)) if (hh > 0 and ww > 0) \
            else np.zeros((0, 0), dtype=bool)
        res[f'nfg_{tag}'] = int(mask.sum())
        if mask.size and mask.sum() >= 2:
            # float32, not float64: the tiles are uint16, which float32 holds
            # EXACTLY (24-bit mantissa > 16-bit values), so this is lossless and
            # halves the pooled arrays we carry until the aggregate. The reduction
            # still accumulates in float64 so the mean itself stays precise.
            pa = ra[:hh, :ww][mask].astype('float32')
            pb = rb[:hh, :ww][mask].astype('float32')
            mse = float(np.mean((pa - pb) ** 2, dtype='float64'))
            mi = mutual_information(pa, pb)
        else:
            pa = pb = np.array([])
            mse = mi = float('nan')
        res[f'mse_{tag}'] = mse
        res[f'mi_{tag}'] = mi
        res[f'pool_{tag}'] = (pa, pb)

    mn, mr = res['mse_n'], res['mse_r']
    in_, ir = res['mi_n'], res['mi_r']
    print(f'  [metrics] {label}   fg naive/reg = {res["nfg_n"]:,}/{res["nfg_r"]:,}')
    print(f'      raw    MSE {mn:10.4f} -> {mr:10.4f} ({_pct(mn, mr):+6.1f}%)   '
          f'MI {in_:6.4f} -> {ir:6.4f} ({_pct(in_, ir):+6.1f}%)')

    s = display_step(a, b)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5) if axis == 'h' else (12, 8))
    _metric_panel(axes[0], a, b, b_y0_n, b_x0_n, nA, nB, thresh,
                  f'NAIVE  (fg={res["nfg_n"]:,})', s)
    _metric_panel(axes[1], a, b, b_y0_r, b_x0_r, rA, rB, thresh,
                  f'REGISTERED  (fg={res["nfg_r"]:,})', s)
    _legend(fig)
    fig.suptitle(f'{label} - MSE/MI box (cyan): zoomed to tissue, background masked',
                 fontsize=10)
    fig.tight_layout()
    save_fig(fig, out_dir, f'{prefix}_metrics.png')
    return res


def _pct(n, r):
    """Percent change n -> r (NaN-safe; used for both MSE and MI)."""
    return 100.0 * (r - n) / n if (n and not np.isnan(n)) else float('nan')


# -- Aggregate reporting -------------------------------------------

def aggregate(seam_metrics):
    """Whole-image MSE / MI: pool EVERY seam's foreground pixel pairs into one
    population and score that once, rather than averaging per-seam numbers (which
    would weight a tiny seam the same as a huge one)."""
    def pool(tag, k):
        parts = [m[f'pool_{tag}'][k] for m in seam_metrics if m[f'pool_{tag}'][k].size]
        return np.concatenate(parts) if parts else np.array([])

    an, bn = pool('n', 0), pool('n', 1)
    ar, br = pool('r', 0), pool('r', 1)
    return dict(
        mse_n=float(np.mean((an - bn) ** 2, dtype='float64')) if an.size else float('nan'),
        mse_r=float(np.mean((ar - br) ** 2, dtype='float64')) if ar.size else float('nan'),
        mi_n=mutual_information(an, bn) if an.size else float('nan'),
        mi_r=mutual_information(ar, br) if ar.size else float('nan'),
    )


def report_metrics(seam_metrics, agg, z, out_dir):
    """Print the per-seam table, write metrics.csv, and draw the bar charts +
    whole-image verdict."""
    print('=' * 92)
    print('  RAW INTENSITY METRICS  |  MSE (lower=better)  |  MI (higher=better)')
    print('=' * 92)
    print(f"  {'Seam':<12} {'MSE naive':>11} {'MSE reg':>11} {'MSE %':>8}   "
          f"{'MI naive':>9} {'MI reg':>9} {'MI %':>8}")
    print('  ' + '-' * 88)
    for m in seam_metrics:
        print(f"  {m['label']:<12} {m['mse_n']:>11.4f} {m['mse_r']:>11.4f} "
              f"{_pct(m['mse_n'], m['mse_r']):>+7.1f}%   "
              f"{m['mi_n']:>9.4f} {m['mi_r']:>9.4f} {_pct(m['mi_n'], m['mi_r']):>+7.1f}%")
    print('  ' + '-' * 88)
    print(f"  {'ALL (image)':<12} {agg['mse_n']:>11.4f} {agg['mse_r']:>11.4f} "
          f"{_pct(agg['mse_n'], agg['mse_r']):>+7.1f}%   "
          f"{agg['mi_n']:>9.4f} {agg['mi_r']:>9.4f} "
          f"{_pct(agg['mi_n'], agg['mi_r']):>+7.1f}%")
    print('\n  MSE: negative % = improvement   |   MI: positive % = improvement')

    with open(Path(out_dir) / 'metrics.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['seam', 'axis', 'dy', 'dx', 'overlap', 'nfg_naive', 'nfg_reg',
                    'mse_naive', 'mse_reg', 'mse_pct', 'mi_naive', 'mi_reg', 'mi_pct'])
        for m in seam_metrics:
            w.writerow([m['label'], m['axis'], f"{m['dy']:.3f}", f"{m['dx']:.3f}",
                        m['overlap'], m['nfg_n'], m['nfg_r'],
                        f"{m['mse_n']:.6f}", f"{m['mse_r']:.6f}",
                        f"{_pct(m['mse_n'], m['mse_r']):.3f}",
                        f"{m['mi_n']:.6f}", f"{m['mi_r']:.6f}",
                        f"{_pct(m['mi_n'], m['mi_r']):.3f}"])
        w.writerow(['ALL (image)', '', '', '', '', '', '',
                    f"{agg['mse_n']:.6f}", f"{agg['mse_r']:.6f}",
                    f"{_pct(agg['mse_n'], agg['mse_r']):.3f}",
                    f"{agg['mi_n']:.6f}", f"{agg['mi_r']:.6f}",
                    f"{_pct(agg['mi_n'], agg['mi_r']):.3f}"])

    labels = [m['label'] for m in seam_metrics] + ['ALL\n(image)']
    x = np.arange(len(labels))
    wid = 0.38
    mse_n = [m['mse_n'] for m in seam_metrics] + [agg['mse_n']]
    mse_r = [m['mse_r'] for m in seam_metrics] + [agg['mse_r']]
    mi_n = [m['mi_n'] for m in seam_metrics] + [agg['mi_n']]
    mi_r = [m['mi_r'] for m in seam_metrics] + [agg['mi_r']]
    fig, (axm, axi) = plt.subplots(1, 2, figsize=(15, 4.5))
    for ax, vals_n, vals_r, ylab, title in [
            (axm, mse_n, mse_r, 'MSE [raw]', 'MSE per seam + whole image  (lower = better)'),
            (axi, mi_n, mi_r, 'MI [raw]', 'MI per seam + whole image  (higher = better)')]:
        ax.bar(x - wid/2, vals_n, wid, label='naive', color='#c44e52')
        ax.bar(x + wid/2, vals_r, wid, label='registered', color='#4c72b0')
        ax.axvline(len(labels) - 1.5, color='gray', ls='--', lw=1, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle(f'Naive vs registered - raw MSE & MI (z-slice {z})', fontsize=12)
    fig.tight_layout()
    save_fig(fig, out_dir, '50_metrics_bars.png')

    fig, (axm, axi) = plt.subplots(1, 2, figsize=(11, 4.5))
    axm.bar(['naive', 'registered'], [agg['mse_n'], agg['mse_r']],
            color=['#c44e52', '#4c72b0'])
    axm.set_title(f"Whole-image MSE (raw)  ({_pct(agg['mse_n'], agg['mse_r']):+.1f}%, "
                  f"lower=better)", fontsize=10)
    axm.set_ylabel('MSE')
    axi.bar(['naive', 'registered'], [agg['mi_n'], agg['mi_r']],
            color=['#c44e52', '#4c72b0'])
    axi.set_title(f"Whole-image MI (raw)  ({_pct(agg['mi_n'], agg['mi_r']):+.1f}%, "
                  f"higher=better)", fontsize=10)
    axi.set_ylabel('MI (nats)')
    fig.suptitle(f'Whole-image validation verdict - all seams pooled (z-slice {z})',
                 fontsize=11)
    fig.tight_layout()
    save_fig(fig, out_dir, '51_verdict.png')


# -- QC driver for one z-slice -------------------------------------

def qc_slice(z, grid_tiles, thresh, out_dir, p):
    """Run the full notebook workflow on one loaded tile grid: register, draw every
    diagnostic, score every seam, and save the naive-vs-registered mosaics. Returns
    the whole-image aggregate so the caller can summarise across slices."""
    n_rows = len(grid_tiles)

    # -- Raw input preview ------------------------------------------
    ncol = max(len(row) for row in grid_tiles)
    fig, axes = plt.subplots(n_rows, ncol, figsize=(3 * ncol, 3 * n_rows), squeeze=False)
    for r, row in enumerate(grid_tiles):
        for c in range(ncol):
            axes[r][c].axis('off')
            if c < len(row):
                axes[r][c].imshow(_disp(row[c], display_step(row[c])), cmap='gray')
                axes[r][c].set_title(f'row {r}, col {c}', fontsize=8)
    fig.suptitle(f'Raw input tiles, z{z:04d} (top -> bottom, left -> right)', fontsize=10)
    fig.tight_layout()
    save_fig(fig, out_dir, '01_input_tiles.png')

    # -- Stage 1: horizontal registration, each grid row -> one row image --------
    row_shifts, rows_reg, rows_naive = [], [], []
    for r, tiles in enumerate(grid_tiles):
        print(f"\n{'='*70}\n  ROW {r}  ({len(tiles)} tiles)\n{'='*70}")
        if len(tiles) == 1:
            shifts = []
            print('  single tile - nothing to register')
        else:
            shifts = register_row(tiles, p.overlap_frac_h, thresh, p.edge_pad_px,
                                  p.upsample, p.highpass_sigma, p.max_cross_px_h,
                                  p.join_band_frac, p.min_signal_frac,
                                  p.min_peak_ratio, p.min_peak_z, label=f'row {r}')
        row_shifts.append(shifts)
        rows_reg.append(composite_line(tiles, shifts, 'h', p.overlap_frac_h, register=True))
        rows_naive.append(composite_line(tiles, shifts, 'h', p.overlap_frac_h, register=False))
        print(f'  row {r}: registered shape={rows_reg[-1].shape}   '
              f'naive shape={rows_naive[-1].shape}')

    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows), squeeze=False)
    for r, img in enumerate(rows_reg):
        axes[r][0].imshow(_disp(img, display_step(img)), cmap='gray', aspect='auto')
        axes[r][0].set_title(f'Row {r} stitched (registered)  {img.shape}', fontsize=9)
        axes[r][0].axis('off')
    fig.suptitle('Stage 1 - horizontally stitched rows (registered)', fontsize=10)
    fig.tight_layout()
    save_fig(fig, out_dir, '20_rows_registered.png')

    # -- Stage 1 diagnostics + metrics, one horizontal seam at a time -----------
    seam_metrics = []
    for r, tiles in enumerate(grid_tiles):
        if len(tiles) < 2:
            continue
        print(f"\n{'-'*70}\n  ROW {r} horizontal seam diagnostics + metrics\n{'-'*70}")
        for i in range(1, len(tiles)):
            a, b = tiles[i-1], tiles[i]
            dy, dx, overlap = row_shifts[r][i-1]
            # Same strips and the same edge_focus_box call register_row made, so the
            # box drawn and scored here IS the box the engine used.
            sa, sb = a[:, -overlap:], b[:, :overlap]
            r0, r1 = edge_focus_box(sa, sb, thresh, p.edge_pad_px, axis=0)
            label = f'R{r} H{i-1}-{i}'
            prefix = f'10_row{r}_pair{i-1}-{i}'
            plot_seam_box(a, b, overlap, r0, r1, 'h', label, out_dir, prefix, thresh)
            compare_overlay_h(a, b, overlap, dy, dx,
                              f'{label} - horizontal pair (tile {i-1} = A/left, '
                              f'tile {i} = B/right)', out_dir, f'{prefix}_overlay.png')
            seam_metrics.append(score_seam(a, b, 'h', overlap, r0, r1, dy, dx, thresh,
                                           label, out_dir, prefix))

    # -- Stage 2: vertical registration, row images -> full mosaic --------------
    if n_rows == 1:
        print('\nOnly one row - the row image IS the full mosaic.')
        v_shifts = []
        mosaic_reg, mosaic_naive = rows_reg[0], rows_naive[0]
    else:
        print('\nRegistering rows top -> bottom (edge-focus box on columns + constrained PCC):')
        v_shifts = register_column(rows_reg, p.overlap_frac_v, thresh, p.edge_pad_px,
                                   p.upsample, p.highpass_sigma, p.max_cross_px_v,
                                   p.join_band_frac, p.min_signal_frac, label='rows')
        mosaic_reg = composite_line(rows_reg, v_shifts, 'v', p.overlap_frac_v, register=True)
        mosaic_naive = composite_line(rows_naive, v_shifts, 'v', p.overlap_frac_v,
                                      register=False)
    print(f'\nRegistered mosaic shape = {mosaic_reg.shape}')
    print(f'Naive mosaic shape      = {mosaic_naive.shape}')

    # -- Stage 2 diagnostics + metrics, one vertical seam at a time -------------
    if n_rows >= 2:
        print(f"\n{'-'*70}\n  Vertical seam diagnostics + metrics (row images)\n{'-'*70}")
        for i in range(1, n_rows):
            a, b = rows_reg[i-1], rows_reg[i]
            dy, dx, overlap = v_shifts[i-1]
            sa, sb = a[-overlap:, :], b[:overlap, :]
            c0, c1 = edge_focus_box(sa, sb, thresh, p.edge_pad_px, axis=1)
            label = f'V {i-1}-{i}'
            prefix = f'30_vseam{i-1}-{i}'
            plot_seam_box(a, b, overlap, c0, c1, 'v', label, out_dir, prefix, thresh)
            compare_overlay_v(a, b, overlap, dy, dx,
                              f'{label} - vertical pair (row {i-1} = A/top, '
                              f'row {i} = B/bottom)', out_dir, f'{prefix}_overlay.png')
            seam_metrics.append(score_seam(a, b, 'v', overlap, c0, c1, dy, dx, thresh,
                                           label, out_dir, prefix))
    else:
        print('Single row - no vertical seams to score.')

    # -- Final mosaic, naive vs registered --------------------------
    s = display_step(mosaic_reg, mosaic_naive)
    fig, axes = plt.subplots(1, 2, figsize=(
        min(24, max(mosaic_reg.shape[1], mosaic_naive.shape[1]) // 80 + 2), 12))
    axes[0].imshow(_disp(mosaic_naive, s), cmap='gray', aspect='auto')
    axes[0].set_title(f'NAIVE full mosaic  {mosaic_naive.shape}  '
                      f'(assumed overlap, dy=dx=0)', fontsize=10)
    axes[0].axis('off')
    axes[1].imshow(_disp(mosaic_reg, s), cmap='gray', aspect='auto')
    axes[1].set_title(f'REGISTERED full mosaic  {mosaic_reg.shape}  '
                      f'(edge-focus + Hann PCC + constrained peak search)', fontsize=10)
    axes[1].axis('off')
    fig.suptitle(f'Full slice registration [CONSTRAINED] - {n_rows} rows x '
                 f'{max(len(r) for r in grid_tiles)} cols (z-slice {z})', fontsize=11)
    fig.tight_layout()
    save_fig(fig, out_dir, '40_mosaic_naive_vs_registered.png')

    if not seam_metrics:
        print('\nNo seams to score (single tile) - skipping the metrics report.')
        return None, mosaic_reg, mosaic_naive, row_shifts, v_shifts

    agg = aggregate(seam_metrics)
    report_metrics(seam_metrics, agg, z, out_dir)
    return agg, mosaic_reg, mosaic_naive, row_shifts, v_shifts


# -- Slice selection -----------------------------------------------

def percentile_slices(z_slices, percentiles):
    """The z at each requested percentile of the DETECTED stack, by position in the
    sorted list (not by z value), so an irregularly numbered stack still yields
    evenly spaced probes. Returns [(percentile, z), ...]."""
    n = len(z_slices)
    return [(q, z_slices[int(round((q / 100.0) * (n - 1)))]) for q in percentiles]


def ordinal(q):
    """'25' -> '25th', '1' -> '1st' -- for the output folder names."""
    q = int(q)
    if 10 <= q % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(q % 10, 'th')
    return f'{q}{suffix}'


def parse_args():
    ap = argparse.ArgumentParser(
        description='QC dry run for registration.py: full diagnostics + MSE/MI seam '
                    'metrics on the 25th, 50th and 75th percentile z-slices, so you '
                    'can check the pipeline before running the whole volume.')
    ap.add_argument('input_dir',
                    help='Folder of tiles named <stem>[<r> x <c>]_C<ch>_z<z>.ome.tif')
    ap.add_argument('--output-dir', default=None,
                    help='Where the "<n>th_percentile_QC" folders go '
                         '(default: the input folder)')
    ap.add_argument('--reference-channel', type=int, default=0,
                    help='Channel to QC -- the same one registration.py registers '
                         '(default: 0)')
    ap.add_argument('--percentiles', default='25,50,75',
                    help='Comma-separated z-stack percentiles to QC (default: 25,50,75)')
    # -- Algorithm knobs (defaults match registration.py) ------------
    ap.add_argument('--overlap-frac-h', type=float, default=0.20,
                    help='Assumed horizontal overlap, fraction of tile WIDTH (default: 0.20)')
    ap.add_argument('--overlap-frac-v', type=float, default=0.20,
                    help='Assumed vertical overlap, fraction of row HEIGHT (default: 0.20)')
    ap.add_argument('--otsu-multiplier', type=float, default=0.5,
                    help='Tissue threshold = OTSU_MULTIPLIER x Otsu(all tiles) (default: 0.5)')
    ap.add_argument('--edge-pad-px', type=int, default=80,
                    help='Rows/cols kept either side of the tissue edge (default: 80)')
    ap.add_argument('--upsample', type=int, default=20,
                    help='(Unused; constrained search uses a parabolic sub-pixel fit)')
    ap.add_argument('--highpass-sigma', type=float, default=12,
                    help='Gaussian sigma for the high-pass prefilter (default: 12)')
    ap.add_argument('--max-cross-px-h', type=int, default=20,
                    help='Max plausible cross-axis shift for HORIZONTAL tile seams, '
                         'px (|dy|; default: 20)')
    ap.add_argument('--max-cross-px-v', type=int, default=100,
                    help='Max plausible cross-axis shift for VERTICAL row seams, px '
                         '(|dx|; whole rows can drift left<->right a lot, default: 100)')
    ap.add_argument('--join-band-frac', type=float, default=1.0,
                    help='Join-axis search half-width, fraction of overlap (default: 1.0)')
    ap.add_argument('--min-signal-frac', type=float, default=0.0001,
                    help='Minimum fraction of each overlap strip that must be signal to '
                         'register with PCC; below it the seam falls back to the nominal '
                         'overlap, dy=dx=0 (default: 0.0001)')
    ap.add_argument('--min-peak-ratio', type=float, default=1.10,
                    help='HORIZONTAL stage only: minimum peak/2nd-peak ratio for the '
                         'constrained correlation peak to count as dominant. Below it the '
                         'seam is treated as textureless (smooth glow / noise) and falls '
                         'back to the nominal overlap, dy=dx=0 (default: 1.10)')
    ap.add_argument('--min-peak-z', type=float, default=0.0,
                    help='HORIZONTAL stage only: minimum peak-to-sidelobe z-score for the '
                         'peak-dominance gate (secondary; 0.0 = off by default)')
    return ap.parse_args()


def main():
    p = parse_args()
    input_dir = Path(p.input_dir)
    out_root = Path(p.output_dir) if p.output_dir else input_dir

    stem, n_rows, n_cols, channels, z_slices, index = scan_input_folder(input_dir)
    in_dtype = tifffile.imread(str(next(iter(index.values()))), is_ome=False).dtype
    if p.reference_channel not in channels:
        raise SystemExit(f'Reference channel {p.reference_channel} not found; '
                         f'available channels: {channels}')
    percentiles = [float(q) for q in p.percentiles.split(',')]
    picks = percentile_slices(z_slices, percentiles)

    print(f'Input folder : {input_dir}')
    print(f'Stem         : {stem}')
    print(f'Grid         : {n_rows} rows x {n_cols} cols')
    print(f'Channels     : {channels}  (QC on C{p.reference_channel:02d})')
    print(f'Z-slices     : {len(z_slices)} detected, z{z_slices[0]:04d}..z{z_slices[-1]:04d}')
    print(f'QC slices    : ' + ', '.join(f'{ordinal(q)} -> z{z:04d}' for q, z in picks))
    print(f'Output root  : {out_root}')

    summary = []
    for q, z in picks:
        out_dir = out_root / f'{ordinal(q)}_percentile_QC'
        out_dir.mkdir(parents=True, exist_ok=True)
        tee = _Tee(out_dir / 'log.txt')
        saved_stdout, sys.stdout = sys.stdout, tee
        try:
            print(f"{'#'*70}\n#  {ordinal(q)} PERCENTILE OF THE Z-STACK  ->  Z-SLICE "
                  f"{z:04d}\n{'#'*70}")
            grid_tiles = load_grid(index, p.reference_channel, z, n_rows, n_cols)
            sample = np.concatenate([t[::8, ::8].ravel() for row in grid_tiles for t in row])
            thresh = p.otsu_multiplier * threshold_otsu(sample)
            print(f'Tissue threshold (Otsu x {p.otsu_multiplier}) = {thresh:.1f}')
            for r, row in enumerate(grid_tiles):
                for c, t in enumerate(row):
                    print(f'  row {r} col {c}: shape={t.shape} dtype={t.dtype} '
                          f'max={t.max():.1f}')

            agg, mosaic_reg, mosaic_naive, _, _ = qc_slice(z, grid_tiles, thresh, out_dir, p)

            # Save both mosaics in the INPUT dtype -- the registered one is exactly
            # what registration.py would write for this slice.
            tifffile.imwrite(str(out_dir / 'mosaic_registered.tif'),
                             to_input_dtype(mosaic_reg, in_dtype))
            tifffile.imwrite(str(out_dir / 'mosaic_naive.tif'),
                             to_input_dtype(mosaic_naive, in_dtype))
            print(f'\nMosaics + figures written to: {out_dir}')
        finally:
            sys.stdout = saved_stdout
            tee.close()
        summary.append((q, z, agg, out_dir))

    print(f"\n{'='*92}\n  QC SUMMARY - whole-image (all seams pooled), naive -> registered"
          f"\n{'='*92}")
    print(f"  {'Slice':<24} {'MSE naive':>12} {'MSE reg':>12} {'MSE %':>8}   "
          f"{'MI naive':>9} {'MI reg':>9} {'MI %':>8}")
    print('  ' + '-' * 88)
    for q, z, agg, _ in summary:
        name = f'{ordinal(q)} pct (z{z:04d})'
        if agg is None:
            print(f'  {name:<24} {"- no seams to score -":>60}')
            continue
        print(f"  {name:<24} {agg['mse_n']:>12.4f} {agg['mse_r']:>12.4f} "
              f"{_pct(agg['mse_n'], agg['mse_r']):>+7.1f}%   "
              f"{agg['mi_n']:>9.4f} {agg['mi_r']:>9.4f} "
              f"{_pct(agg['mi_n'], agg['mi_r']):>+7.1f}%")
    print('  ' + '-' * 88)
    print(f'  MSE down and MI up on all {len(summary)} slices = the pipeline is working; '
          'run registration.py on the volume.')
    print('  Raw MSE carries the per-tile exposure gap, so MI is the more trustworthy '
          'witness of the two.')
    print('\nPer-slice QC folders:')
    for _, _, _, d in summary:
        print(f'  {d}')


if __name__ == '__main__':
    main()
