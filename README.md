# Brain cell-counting pipeline

Takes raw light-sheet tiles of a mouse brain and gives you a table of
**cFos+ cell counts per brain region**. Runs on Sherlock.

Sanjay Palta-Hill & Patrick Steadman

Two halves:

1. **Stitching** (`registrationQC.py`, `registration.py`, `padding.py`) — our own
   code, turns raw tiles into aligned, equally-sized slices.
2. **Atlas pipeline** (`pipeline/`) — MIRACL's CLARITY tools, run end to end.
   Registration, segmentation and counting are all MIRACL's; nothing in
   `pipeline/` re-implements a MIRACL step.

---

## How the two channels are used

| Channel | Contents | What happens to it |
| --- | --- | --- |
| `Channel_0` | autofluorescence | **CLARITY-Allen registration** — this is the channel the Allen atlas is registered to (it shows anatomy; cFos does not) |
| `Channel_1` | cFos signal | **CLARITY segmentation** — cells are detected here, then summarized through the labels that registration produced |

Both are set in `pipeline/config.sh` (`AUTO_CH_DIR` / `SIGNAL_CH_DIR`). Nothing
in the pipeline hardcodes a channel number.

---

## Setup (do once)

**1.** Copy the code to Sherlock:

```bash
scp -r pipeline sanjay01@sherlock.stanford.edu:~/pipeline
```

```bash
scp registrationQC.py registration.py padding.py sanjay01@sherlock.stanford.edu:~/
```

**2.** Open `~/pipeline/config.sh` and set the top block for your sample:

- `SAMPLE` – the sample's folder name
- `VX` / `VZ` – voxel size in microns (in-plane / Z-step). For this scope: `1.6` and `40`.
- `ORIENT` – orientation code of the autofluo volume. **`SAL` for this scope.**
- `SIDE` – `rh` or `lh` (which hemisphere was imaged)

Everything else in that file is already set for this study but is there to be
changed: channel folders, segmentation macro, label resolution, downsample
ratios, Fiji memory. The `DERIVED` block at the bottom updates itself — don't
edit it. Any setting can also be overridden for a single job without editing the
file: `sbatch --export=ALL,SAMPLE=other-sample 01_tiff_to_nii.sbatch`.

---

## Run (in order)

Do steps 1–3 on a compute node. Get one with:

```bash
sh_dev -c 8 -m 32GB -t 120
```

**1. Check the stitch** (look at the output before continuing):
```bash
python registrationQC.py <raw_tiles_folder> --reference-channel 0
```

**2. Stitch the tiles:**
```bash
python registration.py <raw_tiles_folder> --output-dir <sample_folder>
```

**3. Pad both channels** (makes every slice the same size):
```bash
python padding.py <sample_folder>/Channel_0_stitched <sample_folder>/Channel_0_stitched_padded
```

```bash
python padding.py <sample_folder>/Channel_1_stitched <sample_folder>/Channel_1_stitched_padded
```

The rest are Slurm jobs. Submit the whole chain at once — each job waits for the
ones it needs, and Slurm cancels the rest if one fails:

```bash
cd ~/pipeline && ./submit_all.sh
```

Or submit them one at a time, in this order:

**Stage 01 — make the atlas input**, autofluorescence (Ch 0) → NIfTI:
```bash
sbatch 01_tiff_to_nii.sbatch
```

**Stage 02 — register to the Allen atlas** on Ch 0 (~40 min):
```bash
sbatch 02_register.sbatch
```

**Stage 03 — segment the cFos cells** on Ch 1, with MIRACL's CLARITY segmentation:
```bash
sbatch 03_seg_clar.sbatch
```

**Stage 04 — voxelize the segmentation:**
```bash
sbatch 04_voxelize.sbatch
```

**Stage 05 — summarize per region:**
```bash
sbatch 05_feat_extract.sbatch
```

Stages 01–02 (registration, Ch 0) and 03–04 (segmentation, Ch 1) are
independent, so you can run them at the same time. Stage 05 needs both.

**Your result:** `segmentation_cfos/clarity_segmentation_features_ara_labels.csv`
— one row per Allen region, with `Count`, `Density` and cell-volume stats.

```bash
scp sanjay01@sherlock.stanford.edu:<sample_folder>/segmentation_cfos/clarity_segmentation_features_ara_labels.csv ~/Downloads/
```

### What `Count` means

`Count` is the number of detected cells falling in that region **on the voxelized
grid**. `miracl seg voxelize` downsamples by `VOX_DOWN` in Z as well as in-plane,
and our Z step is already 40 µm, so at the default `VOX_DOWN=5` roughly one Z
plane in five is kept. Treat `Count` as a consistent **relative** measure — sound
for comparing regions, and for comparing samples processed with identical
settings — rather than an absolute cell tally. Lower `VOX_DOWN` (e.g. `2`) for a
finer grid; stage 05 then needs noticeably more memory and time.

---

## Check it worked

**After stage 02 (registration):** the job checks this itself and fails loudly,
but you can confirm by hand:
```bash
ls <sample_folder>/reg_final/annotation_hemi_combined_10um_clar_vox.tif
```

**After stage 03 (segmentation)** — do this before trusting any numbers:
```bash
sbatch 06_qc_overlay.sbatch
```

```bash
scp 'sanjay01@sherlock.stanford.edu:<sample_folder>/segmentation_cfos/qc/*.png' ~/Downloads/
```
Open the two PNGs. Red = detected cells. They should sit on the bright dots and
mostly avoid empty background. Use `overlay_slab.png` to judge how many it
catches (it projects several slices so each cell shows once). The job also prints
intensity stats — see the `cfos` vs `cfos2` note below if `norm.tif` looks
saturated or empty.

**Scroll through the whole volume in ITK-SNAP:**
```bash
sbatch 07_view_volume.sbatch
```
Copy down the printed pair of files, open the signal NIfTI as the main image and
`voxelized_seg_cfos.nii.gz` as an overlay. For a full-resolution look instead,
open `segmentation_cfos/seg_cfos.tif` in Fiji next to the raw slices.

---

## Files

```
pipeline/
  config.sh                 everything tunable, per sample
  01_tiff_to_nii.sbatch     Ch 0 autofluo -> NIfTI          (miracl conv tiff_nii)
  02_register.sbatch        register Ch 0 to Allen atlas    (miracl reg clar_allen)
  03_seg_clar.sbatch        segment Ch 1 cFos cells         (MIRACL's Fiji macro, headless)
  04_voxelize.sbatch        voxelize the segmentation       (miracl seg voxelize)
  05_feat_extract.sbatch    per-region table -> .csv        (miracl seg feat_extract)
  06_qc_overlay.sbatch      QC images of the segmentation
  07_view_volume.sbatch     signal-channel NIfTI for ITK-SNAP
  qc_overlay.py             the QC images (run by stage 06); pictures only, no numbers
  submit_all.sh             submit every stage as one dependent Slurm chain
  smoketest_seg.sh          run stage 03 on ~10 slices first, to check Fiji works
```

Stitching scripts (`registrationQC.py`, `registration.py`, `padding.py`) live in
the repo root. Run any of them with `--help` for options.

---

## Tuning the segmentation

`SEG_TYPE` in `config.sh` picks which Fiji macro MIRACL runs:

- **`cfos`** (default) — cFos nuclei. Converts the input to 8-bit **without**
  rescaling, so very bright 16-bit data gets clipped.
- **`cfos2`** — same detector, but auto-scales the 16-bit range before
  converting. Switch to this if the stage 06 QC shows `norm.tif` nearly all white or
  all black, or if the overlays miss obvious cells.
- `virus`, `sparse`, `nuclear` — other stains.

The detector's own thresholds (`radpx`, `minobjsz`, `maxobjsz`) live inside the
macro, not on the command line. To change them, copy the macro out of the
container, edit it, and point `SEG_MACRO_OVERRIDE` at your copy — stage 03
bind-mounts it over MIRACL's:

```bash
singularity exec $SIF cat /code/miracl/seg/miracl_seg_neurons_clarity_3D_cfos.ijm > ~/my_cfos.ijm
```

---

## Orientation & hemisphere

`ORIENT` is the three-letter code describing how the autofluorescence volume is
laid out, and it is **not** a free parameter — a wrong code registers the atlas
to the wrong axes while still "succeeding". For this scope's acquisition it is
**`SAL`**. (An earlier version of this pipeline used `ALS`; that was wrong.)

`SIDE` is the hemisphere imaged (`rh`/`lh`). `HEMI` stays `combined` so left and
right share label IDs, which is what a single-hemisphere sample wants — and
MIRACL's registration script hardcodes `combined` in one internal path check, so
`split` is untested in this container build.

---

## If a step fails

- **Registration finishes in under 5 min with no warp file** – the Z-stack got
  crushed at stage 01. It's fixed already (stage 01 passes `-vx`/`-vz`/`-dz 0`), but
  if you see it, re-run stage 01 and check the printed size is one plane per input
  slice, not ~15.
- **Re-running registration** – stage 02 deletes `clar_allen_reg/` and `reg_final/`
  itself. If you run MIRACL by hand, delete them first or it reuses stale files
  and fails the same way.
- **Stage 03 dies with a `HeadlessException`** – a Fiji plugin in the macro wants
  a GUI. There is no X server in this container at all, so there is no display to
  fall back on; the intermediates left in `segmentation_cfos/` show how far it got.
- **Stage 03 dies with a Java heap error** – raise `JAVA_MEM` in `config.sh` (and
  `--mem` in the job to match).
- **Stage 03 "skips" instantly** – the macro quits if `seg_bin_cfos.tif` already
  exists. `rm -rf <sample_folder>/segmentation_cfos` to redo it.
- **Stage 05 runs out of memory** – you lowered `VOX_DOWN`. Either raise it back or
  move the job to the bigmem partition.
- **`Too many open files`** – you're on a login node; run on a compute node
  (`sh_dev`). The jobs already raise the limit themselves.

Known broken in this container build, and worked around rather than used:

- The `miracl` command itself imports ACE/MONAI for *every* subcommand and dies
  with `Too many open files` when the file-descriptor limit is low — always on a
  login node. The jobs raise the limit and, if the CLI still crashes, fall back to
  MIRACL's underlying script (same code, same flags, no ACE import).
- `miracl seg clar` launches Fiji without `--headless`, and this image ships no
  X server (no `Xvfb`, no `xvfb-run`), so the wrapper cannot work on a compute
  node. Its Fiji is also at `/opt/fiji/ImageJ-linux64`, not where the wrapper
  looks. Stage 03 therefore runs MIRACL's own macro
  (`miracl_seg_neurons_clarity_3D_<type>.ijm`) through that launcher with
  `--headless`, reproducing everything else the wrapper does.

- The registration QC mosaic (`CreateTiledMosaic`) segfaults, so stage 02 passes
  `-f 0` and calls MIRACL's registration shell script directly instead of the
  `miracl reg clar_allen` CLI, which hardcodes `-f 1`.
- The full-resolution per-slice label export (`reg_final/..._tiff_clar/`) never
  gets written. Nothing here needs it: `feat_extract` resamples the label volume
  onto the segmentation grid itself.
- `miracl flow seg` looks for `voxelized_seg_*.tif` while `seg voxelize` writes
  `.tiff`, so stages 03–05 call the three tools separately with explicit paths
  instead of using the flow wrapper.
