# microscopy-stitcher

Registration and stitching of light-sheet microscopy tiles into seamless
whole-slice mosaics.

A slide is imaged as an overlapping grid of tiles (rows × columns). The nominal
overlap from the microscope stage is only approximately right — stage drift,
backlash, and sample settling leave each seam misregistered by tens of pixels.
This project measures the true per-seam shift and stitches the tiles into a
single mosaic, keeping every channel pixel-aligned.

The production tool is [`registration.py`](registration.py): point it at a
folder of OME-TIFF tiles and it writes one stitched mosaic per channel per
z-slice. The Jupyter notebooks are the annotated, figure-by-figure companions
that show *why* the algorithm is built the way it is.

---

## How the pipeline works

Stitching is done in **two stages**, and each seam is registered independently
with phase cross-correlation (PCC) on just the overlap strips.

```
   tiles in a grid            Stage 1: register + composite          Stage 2: register +
   (rows × cols)              each row left → right                  composite rows top → bottom
   ┌───┬───┬───┐              ┌───────────┐                          ┌───────────┐
   │0,0│0,1│0,2│   ──────►    │  row 0    │   ──────►                │           │
   ├───┼───┼───┤              ├───────────┤            stack rows    │ full      │
   │1,0│1,1│1,2│   ──────►    │  row 1    │   ──────►  vertically    │ mosaic    │
   ├───┼───┼───┤              ├───────────┤                          │           │
   │2,0│2,1│2,2│   ──────►    │  row 2    │   ──────►                └───────────┘
   └───┴───┴───┘              └───────────┘
```

- **Stage 1 — horizontal.** Within each grid row, every adjacent tile pair is
  registered left → right, then the row is composited into one wide row image.
- **Stage 2 — vertical.** The row images are registered top → bottom and
  composited into the final mosaic. Stage 2 is Stage 1 rotated 90° — the same
  engine, with the crop and search axes swapped.

### Registering a single seam

For each seam, only the overlapping strips of the two neighbours are compared
(the right edge of A vs. the left edge of B for a horizontal seam). Four ideas
make the shift estimate robust on real tissue:

1. **High-pass prefilter.** `img − gaussian_filter(img, σ=12)` removes the
   smooth brightness envelope (residual vignette + low-frequency tissue
   gradient) so the correlation peak is driven by real texture, not shading.
   This is a high-pass, *not* a band-pass — phase-whitening an already
   band-passed strip amplifies noise and collapses the shift to zero.

2. **Hann window.** A separable 2-D raised-cosine window tapers each strip to
   zero at its borders. Without it the hard strip edges (and bright near-
   horizontal light-sheet streaks) leak a cross of energy through the FFT and
   can capture the peak on a spurious cross-axis shift.

3. **Edge-focus box.** The overlap strip spans the full tile dimension
   perpendicular to the seam, most of which is empty background. PCC can slide
   over that empty run-up and lock onto a false shift. So the strip is cropped
   to hug the **tissue edge** — its rows for a horizontal seam, its columns for
   a vertical one — before correlation. The tissue boundary pins the cross-axis
   shift; the adjacent texture pins the join-axis shift.

4. **Constrained peak search.** Instead of taking phase correlation's *global*
   peak, the algorithm takes the best peak **inside a physically plausible
   window**: the cross-axis shift is bounded (`|dy| ≤ MAX_CROSS_PX` for a
   horizontal seam — tiles in a row barely move vertically), and the join-axis
   shift is bounded to a band around the assumed overlap. A spurious global peak
   (e.g. PCC sliding along a diagonal edge to `dy = +65`) falls outside the
   window, so the true peak inside it wins. Sub-pixel precision comes from a
   parabolic fit around the chosen peak.

### The signal gate

Featureless seams have no reliable peak to find. Before running PCC, each
overlap strip is checked for tissue content: if **either** strip has less than
`MIN_SIGNAL_FRAC` (default 5%) of its pixels above the tissue threshold, PCC is
skipped and the seam falls back to the nominal assumed overlap (`dy = dx = 0`).
This stops empty or near-empty tiles from injecting garbage shifts that would
otherwise staircase the whole mosaic.

The tissue threshold itself is `OTSU_MULTIPLIER × Otsu(subsample of all tiles)`
— one global threshold for the whole slice.

### Blending

Placed tiles are composited with a **sinusoidal (raised-cosine) cross-fade**
over each overlap, so seams fade smoothly rather than showing a hard line.
Background pixels (value 0) are given zero blend weight so they never bleed into
a seam.

### Multi-channel alignment

Shifts are computed **once** on a reference channel (default C00) and the *same*
shifts are applied to every other channel's matching z-slice. Placement is fully
deterministic, so reusing the reference channel's shifts reproduces its exact
geometry for every channel — all channels stay pixel-aligned. This repeats
independently for each z-slice.

### Why raw MSE / MI for validation

The metrics notebook scores each seam with mean-squared error (lower is better)
and mutual information (higher is better), computed on **raw** intensities inside
the tissue box. Registration optimises the *high-passed* signal, so raw
agreement is an **independent** check — improvement there is genuine
corroboration, not the circular result of grading on the optimiser's own
objective. Because raw MSE also carries the per-tile exposure gap, **raw MI is
the more trustworthy witness**; both moving the right way (MSE down, MI up) is
the validation to look for.

---

## Installation

The code depends on the scientific Python stack. It was developed and is pinned
against:

| Package    | Version  |
|------------|----------|
| Python     | 3.12     |
| numpy      | 1.26.4   |
| scipy      | 1.12.0   |
| scikit-image | 0.25.2 |
| tifffile   | 2025.6.11 |

```bash
pip install "numpy==1.26.4" scipy "scikit-image==0.25.2" tifffile matplotlib
```

> **Local environment note.** On the development machine the only interpreter
> with these packages installed is `/opt/miniconda3/bin/python`, which is also
> what the `python3` Jupyter kernel points to. The system `python3` and Homebrew
> Pythons do **not** have the stack. Substitute your own environment's
> interpreter in the commands below.

`matplotlib` is only needed for the notebooks; `registration.py` itself needs
only numpy, scipy, scikit-image, and tifffile.

---

## Input format

Tiles must be OME-TIFFs named with the microscope's layout convention:

```
<stem>[<row> x <col>]_C<channel>_z<zslice>.ome.tif
```

for example:

```
260713_UltraII_5300149-5R_..._ol20[00 x 01]_C00_z0082.ome.tif
                                  └row┘ └col┘  └ch┘ └─z──┘
```

`registration.py` parses every matching file in the input folder to recover the
stem, grid size, channel list, and z-slice list automatically — nothing is
hardcoded. All tiles in a folder must share a single stem (one acquisition per
folder).

---

## `registration.py` — the production tool

This is the final product: the exact preprocessing and constrained-PCC algorithm
from the notebooks, with all the analysis scaffolding (metrics, naive baselines,
diagnostic plots) stripped out. It runs headless and writes stitched TIFFs.

### Basic usage

```bash
/opt/miniconda3/bin/python registration.py "/path/to/tile_folder"
```

It will:

1. Scan the folder and report the detected stem, grid, channels, and z-slices.
2. For **each z-slice**, register the reference channel (C00) — Stage 1
   horizontal, then Stage 2 vertical — and remember the per-seam shifts.
3. Apply those same shifts to **every** channel's matching z-slice.
4. Write each channel's mosaics into its own `Channel <ch> stitched/` folder,
   named `<stem>_z<zslice>_C<ch>_registered.tif`.

Output mosaics carry **real intensities** (only shifted and feather-blended, no
contrast stretch) and are cast back to the input tiles' dtype.

### Example with options

```bash
/opt/miniconda3/bin/python registration.py "/path/to/tiles" \
    --reference-channel 0 \
    --output-dir "/path/to/output" \
    --overlap-frac-h 0.20 \
    --z-slices 82,100
```

### Command-line options

| Option | Default | Meaning |
|--------|---------|---------|
| `input_dir` | *(required)* | Folder of `<stem>[<r> x <c>]_C<ch>_z<z>.ome.tif` tiles |
| `--output-dir` | input folder | Where the `Channel <ch> stitched/` folders are written |
| `--reference-channel` | `0` | Channel used to compute the shifts |
| `--overlap-frac-h` | `0.20` | Assumed horizontal overlap, fraction of tile **width** |
| `--overlap-frac-v` | `0.20` | Assumed vertical overlap, fraction of row **height** |
| `--otsu-multiplier` | `0.5` | Tissue threshold = multiplier × Otsu(all tiles) |
| `--edge-pad-px` | `80` | Rows/cols kept either side of the tissue edge in the focus box |
| `--highpass-sigma` | `12` | Gaussian σ for the high-pass prefilter |
| `--max-cross-px` | `20` | Max plausible cross-axis shift (px) for the constrained search |
| `--join-band-frac` | `1.0` | Join-axis search half-width, as a fraction of the overlap |
| `--min-signal-frac` | `0.05` | Min foreground fraction of **each** overlap strip to run PCC; below this the seam falls back to the nominal overlap |
| `--z-slices` | all detected | Comma-separated subset of z-slices to process, e.g. `82,100` |
| `--upsample` | `20` | Unused in this variant (the constrained search uses a parabolic sub-pixel fit); kept for CLI compatibility |

### When to adjust the knobs

- **Wrong overlap** — if the microscope's tile overlap isn't ~20%, set
  `--overlap-frac-h/-v` to match; the search band is anchored to this value.
- **Seams still drifting on the cross-axis** — lower `--max-cross-px` to
  constrain the search more tightly, or raise `--edge-pad-px` if the tissue edge
  is being cropped too aggressively.
- **Empty tiles injecting bad shifts** — raise `--min-signal-frac`; lower it if
  legitimate sparse seams are being skipped.
- **Faint or low-contrast tissue** — lower `--otsu-multiplier` so more of the
  strip counts as foreground.

---

## The notebooks

The notebooks are the explanatory companions to `registration.py`. They share
the same core helpers (`load_tile`, `highpass`, the Hann window, `edge_focus_box`,
the constrained PCC, and the feathered compositor) and each renders
**naive vs. registered** side by side so the improvement is visible at every
step. All produce figures — run them with a Jupyter kernel on the pinned
environment.

Because they open specific tiles for illustration, the `TILE_PATH(S)` / `GRID`
and `Z_SLICE` variables at the top of each are hardcoded to the development
machine's test images — **edit those to your own paths before running.**

### [`preprocessing.ipynb`](preprocessing.ipynb) — the prefilter, visualised

The smallest notebook. Pick **one** tile (`TILE_PATH`) and it shows, side by
side, the raw tile → after the Gaussian high-pass → after the Hann window. This
is exactly the two-step preprocessing every overlap strip goes through before
PCC. Start here to build intuition for *why* the prefilter matters.

### [`basic_horizontal.ipynb`](basic_horizontal.ipynb) — one seam, end to end

Registers a **single horizontal pair** (A = left, B = right) and walks through
the whole per-seam method:

- loads both tiles and shows each next to its tissue mask,
- builds the **edge-focus box** (yellow) on the overlap strips,
- runs the high-passed, Hann-windowed PCC (with the signal-gate fallback),
- shows the red/green/yellow overlap overlay **before vs. after** the shift, and
- stitches the pair both ways with the sinusoidal blend.

Set `TILE_PATHS` to two horizontally adjacent tiles. This is the reference
implementation of a single seam — the atom the full pipeline is built from.

### [`full_slice_registration_constrained.ipynb`](full_slice_registration_constrained.ipynb) — the full mosaic

Stitches a **whole z-slice** — an arbitrary N-row × M-col grid — using the
two-stage pipeline, and is the notebook `registration.py` was extracted from.

Configure the input at the top:

- `BASE` + `_stem` + `Z_SLICE` build the tile paths (several test cases are
  provided, commented out),
- `GRID` is a list of rows, each a list of tile paths **left → right**, rows
  listed **top → bottom** (rows need not be equal length),
- the algorithm knobs (`OVERLAP_FRAC_H/V`, `OTSU_MULTIPLIER`, `EDGE_PAD_PX`,
  `HIGHPASS_SIGMA`, `MIN_SIGNAL_FRAC`, `MAX_CROSS_PX`, `JOIN_BAND_FRAC`) mirror
  the `registration.py` command-line options one-for-one.

It renders the raw input grid, the edge-focus box + Hann window at every
horizontal pair, the naive-vs-registered overlay at each step, Stage 1 row
images, Stage 2 vertical joins, and the final naive-vs-registered mosaic, then
saves the registered mosaic. This variant takes the constrained-window peak
(not the global peak), so each seam still gets its own measured shift.

### [`full_slice_with_metrics.ipynb`](full_slice_with_metrics.ipynb) — the full mosaic, scored

Identical stitching to the constrained notebook, **plus an evaluation layer**.
Every seam is scored naive vs. registered with MSE and mutual information,
computed on **raw** intensities inside the same tissue box used for the Hann
window (background masked out, so empty pixels can't dilute the score). It adds:

- a cyan MSE box drawn on each seam (exactly the pixels that count),
- Stage 1 and Stage 2 per-seam metric sections, and
- a final summary — a per-seam table with a whole-image aggregate, plus bar
  charts of every seam's naive → registered movement.

Use this notebook to **validate** a stitch: confirm MSE goes down and MI goes up
across the seams. Use the constrained notebook (or `registration.py`) when you
just want the mosaic.

---

## Repository layout

```
registration.py                              # production stitcher (CLI)
preprocessing.ipynb                          # high-pass + Hann window, visualised
basic_horizontal.ipynb                       # one horizontal seam, end to end
full_slice_registration_constrained.ipynb    # full-slice two-stage mosaic
full_slice_with_metrics.ipynb                # full-slice mosaic + MSE/MI validation
```
