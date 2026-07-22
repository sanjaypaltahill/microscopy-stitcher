"""registration.py

Constrained-PCC full-slice registration, extracted from
full_slice_registration_constrained.ipynb. Runs the EXACT same preprocessing and
constrained phase-cross-correlation algorithm, but with none of the analysis
scaffolding: no MSE/MI, no naive stitching, no sanity-check or intermediate plots.

Point it at an input folder of OME-TIFF tiles named like
    <stem>[<row> x <col>]_C<ch>_z<z>.ome.tif
It auto-detects the stem, grid size, channels and z-slices. For every z-slice it
registers the REFERENCE channel (default C00), remembers the per-tile shifts, and
applies those SAME shifts to the other channels' matching z-slice -- so all
channels stay pixel-aligned. Each channel's registered mosaics are written to its
own "Channel <ch> stitched" folder.

Run with the project python, e.g.:
    /opt/miniconda3/bin/python registration.py "/Users/spaltahill/test_images"
    /opt/miniconda3/bin/python registration.py <input_dir> --reference-channel 0 \
        --output-dir <out> --overlap-frac-h 0.20
"""

# -- Imports -------------------------------------------------------
import argparse
import re
import warnings
from collections import defaultdict
import numpy as np
import tifffile
from pathlib import Path
from skimage.registration import phase_cross_correlation
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter


# -- Core helpers (from basic_horizontal) --------------------------

def load_tile(path):
    img = tifffile.imread(str(path)).astype('float32')
    while img.ndim > 2:
        img = img[0]
    return img


def norm_disp(img, lo_pct=0.5, hi_pct=99.5):
    """Scale to [0,1] for display with robust percentile limits, so a single
    bright speck can't squash the rest of an image to near-black."""
    lo, hi = np.percentile(img, [lo_pct, hi_pct])
    return np.clip((img - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def highpass(img, sigma=12.0):
    """Subtract a large-sigma Gaussian to strip the smooth brightness envelope
    (residual vignette + tissue low-frequency gradient) so the true-texture
    correlation peak dominates. High-pass, NOT band-pass."""
    img = img.astype('float32')
    return img - gaussian_filter(img, sigma)


def hann2d(shape):
    """Separable 2D Hann window for a strip of the given shape."""
    return (np.hanning(shape[0])[:, None] * np.hanning(shape[1])[None, :]).astype('float32')


def _pcc_prep(a, b, highpass_sigma, window):
    """Match shapes, high-pass, and Hann-window a pair of strips (shared by the
    plain and constrained PCC so both see identical input)."""
    if a.shape != b.shape:
        r = min(a.shape[0], b.shape[0]); c = min(a.shape[1], b.shape[1])
        a, b = a[:r, :c], b[:r, :c]
    if highpass_sigma:
        a, b = highpass(a, highpass_sigma), highpass(b, highpass_sigma)
    if window:
        win = hann2d(a.shape)
        a, b = a * win, b * win
    return a, b


def _parabolic_subpx(corr, py, px):
    """Sub-pixel offset of a correlation peak by a parabolic fit along each axis
    (replaces skimage's upsampled-DFT refinement for the constrained search)."""
    h, w = corr.shape
    def off(i, n, val):
        m1, p1, z = val((i - 1) % n), val((i + 1) % n), val(i)
        den = m1 - 2 * z + p1
        return 0.5 * (m1 - p1) / den if abs(den) > 1e-9 else 0.0
    dy = off(py, h, lambda k: corr[k, px])
    dx = off(px, w, lambda k: corr[py, k])
    return dy, dx


def register_pcc_constrained(strip_a, strip_b, highpass_sigma, window,
                             max_cross, join_center, join_half, axis):
    """Phase correlation that takes the best peak WITHIN A PLAUSIBLE WINDOW instead
    of the global peak. Same normalized cross-power spectrum as `register_pcc`
    (verified to reproduce skimage's shift on well-conditioned pairs), but the
    peak search is masked to physically sensible shifts:

      * axis='h': |dy| <= max_cross (a horizontal pair barely moves vertically) and
        dx in [join_center +- join_half] (near the assumed overlap).
      * axis='v': |dx| <= max_cross and dy in the join window.

    A spurious global peak (e.g. PCC sliding along a diagonal edge to dy=+65) falls
    OUTSIDE the window, so the true peak inside it wins instead. Sub-pixel via a
    parabolic fit. Returns (dy, dx) in the same convention as register_pcc."""
    a, b = _pcc_prep(strip_a, strip_b, highpass_sigma, window)
    A = np.fft.fft2(a); B = np.fft.fft2(b)
    R = A * np.conj(B); R /= np.abs(R) + 1e-12
    corr = np.fft.ifft2(R).real
    h, w = corr.shape
    dY = np.where(np.arange(h) > h // 2, np.arange(h) - h, np.arange(h))
    dX = np.where(np.arange(w) > w // 2, np.arange(w) - w, np.arange(w))
    if axis == 'h':
        oky = np.abs(dY) <= max_cross
        okx = (dX >= join_center - join_half) & (dX <= join_center + join_half)
    else:
        oky = (dY >= join_center - join_half) & (dY <= join_center + join_half)
        okx = np.abs(dX) <= max_cross
    masked = np.where(np.outer(oky, okx), corr, -np.inf)
    py, px = np.unravel_index(np.argmax(masked), masked.shape)
    dyf, dxf = _parabolic_subpx(corr, py, px)
    dy = (py - h if py > h // 2 else py) + dyf
    dx = (px - w if px > w // 2 else px) + dxf
    return float(dy), float(dx)


def edge_focus_box(strip_a, strip_b, thresh, pad_px, axis=0, min_frac=0.02, min_lines=32):
    """Indices (lo, hi) ALONG `axis` cropped to the tissue extent; the other
    dimension is left untouched. This is the "focus thin" box, generalised so the
    horizontal and vertical stages are analogous:

      * axis=0 crops ROWS  -> horizontal-seam strips (H x overlap): box hugs the
        tissue's vertical extent, full overlap width kept.
      * axis=1 crops COLS  -> vertical-seam strips (overlap x W): box hugs the
        tissue's horizontal extent, full overlap height kept.

    A line along `axis` counts as tissue when > min_frac of it is foreground
    (> thresh) in EITHER strip; keep [first..last] tissue lines padded by pad_px,
    clamped to the extent. Falls back to the full extent when there is too little
    tissue to trust (< min_lines lines)."""
    n = min(strip_a.shape[axis], strip_b.shape[axis])
    if axis == 0:
        a, b = strip_a[:n], strip_b[:n]
    else:
        a, b = strip_a[:, :n], strip_b[:, :n]
    mask = (a > thresh) | (b > thresh)
    cover = mask.mean(axis=1 - axis)            # foreground fraction per line along axis
    lines = np.where(cover > min_frac)[0]
    if lines.size == 0:
        return 0, n
    lo = max(0, int(lines[0]) - pad_px)
    hi = min(n, int(lines[-1]) + 1 + pad_px)
    if hi - lo < min_lines:
        return 0, n
    return lo, hi


def enough_signal(strip_a, strip_b, thresh, min_signal_frac):
    """Whether the two overlap strips carry enough tissue to trust PCC.

    Builds the same foreground mask edge_focus_box uses (pixels > thresh) and
    measures the fraction of EACH strip that is signal. Returns
    (ok, frac_a, frac_b): ok is False when EITHER strip has less than
    min_signal_frac of its pixels above threshold -- too little actual signal in
    the overlap to register reliably, so the caller skips PCC and falls back to
    the nominal assumed overlap (dy=dx=0)."""
    frac_a = float((strip_a > thresh).mean())
    frac_b = float((strip_b > thresh).mean())
    return (frac_a >= min_signal_frac and frac_b >= min_signal_frac), frac_a, frac_b


def sinusoid_ramp(n, feather):
    """Length-n weight: 1 in the interior, a raised-cosine ease from ~0 to 1 over
    `feather` px at each end -> a smooth sinusoidal cross-fade across overlaps."""
    w = np.ones(n, dtype='float32')
    f = int(min(feather, n // 2))
    if f > 0:
        t = (np.arange(f) + 1) / (f + 1)
        edge = (0.5 * (1 - np.cos(np.pi * t))).astype('float32')
        w[:f] = edge
        w[-f:] = edge[::-1]
    return w


def feather_weight(shape, axis, feather):
    """Sinusoidal cross-fade weight map ramping along `axis` (1 = left/right for a
    horizontal join, 0 = top/bottom for a vertical stack)."""
    ramp = sinusoid_ramp(shape[axis], feather)
    if axis == 1:
        return np.broadcast_to(ramp[None, :], shape).copy()
    return np.broadcast_to(ramp[:, None], shape).copy()


def composite(placements, feather_axis, feather_px):
    """Blend placed images onto one canvas with sinusoidal feathered weights so
    overlaps cross-fade. placements: list of (img, y0, x0) in a shared frame
    (offsets may be negative). Background pixels (value == 0) carry zero weight so
    they never bleed into a seam. Returns the canvas."""
    y_min = min(y0 for _, y0, _ in placements)
    x_min = min(x0 for _, _, x0 in placements)
    H = max(y0 - y_min + img.shape[0] for img, y0, _ in placements)
    W = max(x0 - x_min + img.shape[1] for img, _, x0 in placements)
    acc = np.zeros((H, W), 'float32'); wsum = np.zeros((H, W), 'float32')
    for img, y0, x0 in placements:
        yy, xx = y0 - y_min, x0 - x_min
        w = feather_weight(img.shape, feather_axis, feather_px)
        w[img == 0] = 0.0
        acc[yy:yy+img.shape[0], xx:xx+img.shape[1]] += w * img
        wsum[yy:yy+img.shape[0], xx:xx+img.shape[1]] += w
    return acc / np.maximum(wsum, 1e-6)


# -- Registration + line compositing -------------------------------

def register_row(tiles, overlap_frac, thresh, pad_px, upsample, highpass_sigma,
                 max_cross, join_band_frac, min_signal_frac, label=''):
    """Register a row of tiles left->right with the basic_horizontal engine, but
    with CONSTRAINED-SEARCH PCC: the peak is taken within |dy| <= max_cross and dx
    within +-join_band_frac x overlap, so a spurious diagonal-slide peak can't win.
    (`upsample` is unused here; the constrained search uses a parabolic sub-pixel
    fit.) When less than `min_signal_frac` of EITHER overlap strip is signal
    (foreground > thresh) there is too little tissue to trust PCC, so that seam
    skips registration and falls back to the nominal overlap (dy=dx=0). Returns
    [(dy, dx, overlap), ...]."""
    shifts = []
    for i in range(1, len(tiles)):
        a, b = tiles[i-1], tiles[i]
        overlap = max(1, min(int(round(overlap_frac * a.shape[1])), a.shape[1], b.shape[1]))
        sa, sb = a[:, -overlap:], b[:, :overlap]          # A right edge, B left edge
        ok, fa, fb = enough_signal(sa, sb, thresh, min_signal_frac)
        if not ok:
            shifts.append((0.0, 0.0, overlap))
            print(f'  {label}: pair {i-1}-{i}  overlap={overlap}px  signal a={fa:.1%} '
                  f'b={fb:.1%} < {min_signal_frac:.1%}  -> too little signal, skipping '
                  f'PCC, using nominal overlap (dy=dx=0)')
            continue
        r0, r1 = edge_focus_box(sa, sb, thresh, pad_px, axis=0)
        dy, dx = register_pcc_constrained(sa[r0:r1], sb[r0:r1], highpass_sigma, True,
                                          max_cross, 0, int(round(join_band_frac * overlap)), 'h')
        shifts.append((dy, dx, overlap))
        hug = ('  [hugs top]' if r0 == 0 else '') + ('  [hugs bottom]' if r1 == sa.shape[0] else '')
        print(f'  {label}: pair {i-1}-{i}  overlap={overlap}px  box rows {r0}-{r1} '
              f'({r1-r0}px tall){hug}  dy={dy:+.2f} dx={dx:+.2f}')
    return shifts


def register_column(rows, overlap_frac, thresh, pad_px, upsample, highpass_sigma,
                    max_cross, join_band_frac, min_signal_frac, label=''):
    """Register row images top->bottom, ANALOGOUS to register_row but rotated 90deg
    (edge-focus box on COLUMNS, axis=1), with the same CONSTRAINED-SEARCH PCC: the
    peak is taken within |dx| <= max_cross and dy within +-join_band_frac x overlap.
    (`upsample` is unused; parabolic sub-pixel.) When less than `min_signal_frac` of
    EITHER overlap strip is signal (foreground > thresh) there is too little tissue
    to trust PCC, so that seam skips registration and falls back to the nominal
    overlap (dy=dx=0). Returns [(dy, dx, overlap), ...]."""
    shifts = []
    for i in range(1, len(rows)):
        a, b = rows[i-1], rows[i]
        overlap = max(1, min(int(round(overlap_frac * a.shape[0])), a.shape[0], b.shape[0]))
        sa, sb = a[-overlap:, :], b[:overlap, :]          # A bottom edge, B top edge
        ok, fa, fb = enough_signal(sa, sb, thresh, min_signal_frac)
        if not ok:
            shifts.append((0.0, 0.0, overlap))
            print(f'  {label}: rows {i-1}-{i}  overlap={overlap}px  signal a={fa:.1%} '
                  f'b={fb:.1%} < {min_signal_frac:.1%}  -> too little signal, skipping '
                  f'PCC, using nominal overlap (dy=dx=0)')
            continue
        c0, c1 = edge_focus_box(sa, sb, thresh, pad_px, axis=1)
        dy, dx = register_pcc_constrained(sa[:, c0:c1], sb[:, c0:c1], highpass_sigma, True,
                                          max_cross, 0, int(round(join_band_frac * overlap)), 'v')
        shifts.append((dy, dx, overlap))
        w = min(sa.shape[1], sb.shape[1])
        hug = ('  [hugs left]' if c0 == 0 else '') + ('  [hugs right]' if c1 == w else '')
        print(f'  {label}: rows {i-1}-{i}  overlap={overlap}px  box cols {c0}-{c1} '
              f'({c1-c0}px wide){hug}  dy={dy:+.2f} dx={dx:+.2f}')
    return shifts


def composite_line(images, shifts, axis, overlap_frac, register=True):
    """Place `images` along `axis` ('h' or 'v') and blend with a sinusoidal
    cross-fade. register=True applies the measured (dy, dx); register=False forces
    dy=dx=0 (the naive assumed-overlap baseline). Overlaps recomputed per seam so
    the naive canvas is self-consistent."""
    placements = [(images[0], 0, 0)]
    y_cur = x_cur = 0
    for k in range(1, len(images)):
        dy, dx, _ = shifts[k-1]
        prev = images[k-1]
        sdy = int(round(dy)) if register else 0
        sdx = int(round(dx)) if register else 0
        if axis == 'h':
            overlap = max(1, min(int(round(overlap_frac * prev.shape[1])),
                                 prev.shape[1], images[k].shape[1]))
            x_cur += prev.shape[1] - overlap + sdx
            y_cur += sdy
        else:
            overlap = max(1, min(int(round(overlap_frac * prev.shape[0])),
                                 prev.shape[0], images[k].shape[0]))
            y_cur += prev.shape[0] - overlap + sdy
            x_cur += sdx
        placements.append((images[k], y_cur, x_cur))
    feather_axis = 1 if axis == 'h' else 0
    fdim = images[0].shape[1] if axis == 'h' else images[0].shape[0]
    feather = max(1, int(round(overlap_frac * fdim)))
    return composite(placements, feather_axis, feather)


# -- Input-folder auto-detection -----------------------------------
# Tiles are named  <stem>[<row> x <col>]_C<ch>_z<z>.ome.tif  (the OME-TIFF layout
# from the microscope). We parse every such file in the input folder to recover the
# stem, grid size, channel list and z-slice list -- no hardcoding.
TILE_RE = re.compile(
    r'^(?P<stem>.+?)\[(?P<row>\d+) x (?P<col>\d+)\]_C(?P<ch>\d+)_z(?P<z>\d+)\.ome\.tif$')


def scan_input_folder(input_dir):
    """Parse tile filenames in `input_dir`. Returns (stem, n_rows, n_cols,
    channels, z_slices, index) where index[(ch, z, row, col)] = Path."""
    input_dir = Path(input_dir)
    stems, rows, cols, chans, zs = set(), set(), set(), set(), set()
    index = {}
    for f in sorted(input_dir.iterdir()):
        m = TILE_RE.match(f.name)
        if not m:
            continue
        stems.add(m['stem'])
        r, c, ch, z = int(m['row']), int(m['col']), int(m['ch']), int(m['z'])
        rows.add(r); cols.add(c); chans.add(ch); zs.add(z)
        index[(ch, z, r, c)] = f
    if not index:
        raise SystemExit(f'No tiles matching "<stem>[<r> x <c>]_C<ch>_z<z>.ome.tif" '
                         f'found in {input_dir}')
    if len(stems) > 1:
        raise SystemExit(f'Multiple stems found in {input_dir}: {sorted(stems)} '
                         '- point at a folder holding a single acquisition.')
    stem = stems.pop()
    return (stem, max(rows) + 1, max(cols) + 1,
            sorted(chans), sorted(zs), index)


def load_grid(index, ch, z, n_rows, n_cols):
    """Load the (n_rows x n_cols) tile grid for one channel + z-slice as a list of
    rows (TOP->BOTTOM), each a list of tiles (LEFT->RIGHT)."""
    grid = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            path = index.get((ch, z, r, c))
            if path is None:
                raise SystemExit(f'Missing tile: channel {ch}, z{z:04d}, '
                                 f'row {r}, col {c}')
            row.append(load_tile(path))
        grid.append(row)
    return grid


# -- Registration driver -------------------------------------------

def compute_shifts(ref_grid, thresh, p):
    """Register the REFERENCE grid and return the shifts to reuse across channels:
      row_shifts : list (one per row) of [(dy, dx, overlap), ...] horizontal seams
      v_shifts   : [(dy, dx, overlap), ...] vertical seams between the row images
    Identical to the notebook's two-stage registration on the reference channel."""
    n_rows = len(ref_grid)
    row_shifts = []
    rows_reg = []
    for r, tiles in enumerate(ref_grid):
        print(f"\n{'='*70}\n  ROW {r}  ({len(tiles)} tiles)\n{'='*70}")
        if len(tiles) == 1:
            shifts = []
            print('  single tile - nothing to register')
        else:
            shifts = register_row(tiles, p.overlap_frac_h, thresh, p.edge_pad_px,
                                  p.upsample, p.highpass_sigma, p.max_cross_px,
                                  p.join_band_frac, p.min_signal_frac, label=f'row {r}')
        row_shifts.append(shifts)
        rows_reg.append(composite_line(tiles, shifts, 'h', p.overlap_frac_h, register=True))

    if n_rows == 1:
        print('\nOnly one row - no vertical registration.')
        v_shifts = []
    else:
        print('\nRegistering rows top -> bottom (edge-focus box on columns + constrained PCC):')
        v_shifts = register_column(rows_reg, p.overlap_frac_v, thresh, p.edge_pad_px,
                                   p.upsample, p.highpass_sigma, p.max_cross_px,
                                   p.join_band_frac, p.min_signal_frac, label='rows')
    return row_shifts, v_shifts


def apply_shifts(grid, row_shifts, v_shifts, p):
    """Stitch `grid` using PRE-COMPUTED shifts (no registration). Placement is
    deterministic in composite_line, so feeding the reference channel's shifts here
    reproduces its exact mosaic for any channel whose tiles share the same shape."""
    rows_reg = [composite_line(tiles, shifts, 'h', p.overlap_frac_h, register=True)
                for tiles, shifts in zip(grid, row_shifts)]
    if len(rows_reg) == 1:
        return rows_reg[0]
    return composite_line(rows_reg, v_shifts, 'v', p.overlap_frac_v, register=True)


def to_input_dtype(mosaic, dtype):
    """Cast the float32 blended mosaic back to the input tiles' dtype. Integer
    types are rounded and clipped to their valid range; float types pass through."""
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.rint(mosaic).clip(info.min, info.max).astype(dtype)
    return mosaic.astype(dtype)


def parse_args():
    ap = argparse.ArgumentParser(
        description='Constrained-PCC registration of an OME-TIFF tile folder. '
                    'Registers the reference channel per z-slice and applies the '
                    'same shifts to every other channel.')
    ap.add_argument('input_dir',
                    help='Folder of tiles named <stem>[<r> x <c>]_C<ch>_z<z>.ome.tif')
    ap.add_argument('--output-dir', default=None,
                    help='Where the "Channel <ch> stitched" folders go '
                         '(default: the input folder)')
    ap.add_argument('--reference-channel', type=int, default=0,
                    help='Channel used to compute the shifts (default: 0)')
    # -- Algorithm knobs (defaults match the notebook) --------------
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
    ap.add_argument('--max-cross-px', type=int, default=20,
                    help='Max plausible cross-axis shift, px (default: 20)')
    ap.add_argument('--join-band-frac', type=float, default=1.0,
                    help='Join-axis search half-width, fraction of overlap (default: 1.0)')
    ap.add_argument('--min-signal-frac', type=float, default=0.05,
                    help='Minimum fraction of each overlap strip that must be signal '
                         '(foreground > tissue threshold) to register with PCC. If '
                         'either strip has less, skip PCC and default to the nominal '
                         'assumed overlap, dy=dx=0 (default: 0.05)')
    ap.add_argument('--z-slices', default=None,
                    help='Comma-separated z-slices to process (default: all detected)')
    return ap.parse_args()


def main():
    p = parse_args()
    input_dir = Path(p.input_dir)
    out_root = Path(p.output_dir) if p.output_dir else input_dir

    stem, n_rows, n_cols, channels, z_slices, index = scan_input_folder(input_dir)
    in_dtype = tifffile.imread(str(next(iter(index.values())))).dtype  # match on save
    if p.z_slices:
        wanted = {int(z) for z in p.z_slices.split(',')}
        z_slices = [z for z in z_slices if z in wanted]
    if p.reference_channel not in channels:
        raise SystemExit(f'Reference channel {p.reference_channel} not found; '
                         f'available channels: {channels}')

    print(f'Input folder : {input_dir}')
    print(f'Stem         : {stem}')
    print(f'Grid         : {n_rows} rows x {n_cols} cols')
    print(f'Channels     : {channels}  (reference = C{p.reference_channel:02d})')
    print(f'Z-slices     : {z_slices}')
    print(f'Output root  : {out_root}')

    # One output folder per channel.
    out_dirs = {ch: out_root / f'Channel {ch} stitched' for ch in channels}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    for z in z_slices:
        print(f"\n{'#'*70}\n#  Z-SLICE {z:04d}\n{'#'*70}")

        # 1) Register the reference channel -> per-tile shifts.
        ref_grid = load_grid(index, p.reference_channel, z, n_rows, n_cols)
        sample = np.concatenate([t[::8, ::8].ravel() for row in ref_grid for t in row])
        thresh = p.otsu_multiplier * threshold_otsu(sample)
        print(f'Tissue threshold (Otsu x {p.otsu_multiplier}) = {thresh:.1f}')
        row_shifts, v_shifts = compute_shifts(ref_grid, thresh, p)

        # 2) Apply those SAME shifts to every channel (incl. the reference) and save.
        for ch in channels:
            grid = ref_grid if ch == p.reference_channel else load_grid(
                index, ch, z, n_rows, n_cols)
            mosaic = apply_shifts(grid, row_shifts, v_shifts, p)
            out_path = out_dirs[ch] / f'{stem}_z{z:04d}_C{ch:02d}_registered.tif'
            # Save the RAW registered mosaic: real intensities, only shifted and
            # feather-blended (no norm_disp / contrast stretch). Cast back to the
            # INPUT dtype so the output matches the tiles; integer types are rounded
            # and clipped to the valid range (blending can't exceed it, but guard).
            tifffile.imwrite(str(out_path), to_input_dtype(mosaic, in_dtype))
            print(f'  C{ch:02d} z{z:04d}: mosaic {mosaic.shape} -> {out_path}')

    print(f'\nDone. Registered mosaics written under: {out_root}')


if __name__ == '__main__':
    main()
