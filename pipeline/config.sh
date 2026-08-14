#!/bin/bash
# =============================================================================
#  Per-sample configuration for the MIRACL CLARITY registration + segmentation
#  pipeline.
# -----------------------------------------------------------------------------
#  Every NN_*.sbatch sources this file (from $PIPELINE_DIR, default
#  $HOME/pipeline). Edit the values in the "EDIT" blocks; everything under
#  "DERIVED" updates automatically. To process a new sample, change SAMPLE and
#  resubmit.
#
#  Every setting below uses ${VAR:-default}, so anything can also be overridden
#  for one job without editing this file:
#      sbatch --export=ALL,SAMPLE=other-sample 01_tiff_to_nii.sbatch
#
#  This pipeline runs MIRACL's own CLARITY tools end to end:
#      conv tiff_nii -> reg clar_allen -> seg clar -> seg voxelize -> seg feat_extract
#  Nothing here re-implements a MIRACL step. The only local script is
#  qc_overlay.py, which makes pictures and produces no numbers.
# =============================================================================

# ---- EDIT PER SAMPLE --------------------------------------------------------
SAMPLE="${SAMPLE:-5300149-5R}"                                     # sample ID = folder under MAPDIR
MAPDIR="${MAPDIR:-/scratch/users/sanjay01/mapDMG}"                 # parent dir holding all samples
SIF="${SIF:-/scratch/users/steadmp/mapDMG/miracl/miracl.sif}"   # MIRACL Singularity image

# ---- CHANNELS ---------------------------------------------------------------
# Which stitched+padded channel folder is which. On this scope:
#   Channel_0 = autofluorescence -> this is what gets REGISTERED to the Allen atlas
#   Channel_1 = cFos signal      -> this is what gets SEGMENTED
# Change these two names (not the roles) if a sample was acquired differently.
AUTO_CH_DIR="${AUTO_CH_DIR:-Channel_0_stitched_padded}"     # autofluorescence / reference channel
SIGNAL_CH_DIR="${SIGNAL_CH_DIR:-Channel_1_stitched_padded}"   # signal channel to segment (cFos)

AUTO_CH_NAME="${AUTO_CH_NAME:-autofluo}"   # -ch : label baked into the converted NIfTI's filename
SIGNAL_CH_NAME="${SIGNAL_CH_NAME:-cfos}"     # -ch : same, for the viewing volume built in stage 07

# ---- ACQUISITION ------------------------------------------------------------
# Acquisition voxel size, in microns: in-plane (X/Y) and Z step between slices.
# These MUST match how the sample was imaged — wrong values corrupt every step.
VX="${VX:-1.6}"          # X/Y resolution (um per pixel)
VZ="${VZ:-40}"           # Z step (um between slices)
DOWN="${DOWN:-5}"          # in-plane downsample ratio for the registration NIfTI

# ---- REGISTRATION (stage 02) ------------------------------------------------

## ASR is the correct orientation
ORIENT="${ORIENT:-ASR}"      # -o : orientation code of the autofluo volume (see README).
                  #      This is NOT a free parameter — a wrong code silently
                  #      mis-registers, and the run still exits 0.
                  #      ASR determined 2026-08-14 from the data, not guessed:
                  #        - axis0 = A-P  (215 vox x 50um = 10.75 mm)
                  #        - axis1 = D-V  (183 vox x 50um =  9.15 mm)
                  #        - axis2 = M-L  (109 sections x 40um = 4.35 mm, a
                  #          full hemisphere; no other brain axis is ~4.4 mm)
                  #      Signs came from mask-overlap scoring against the Allen
                  #      right hemisphere: A over P by 0.80/0.64, R over L by
                  #      0.80/0.57 (both decisive), S over I by 0.8051/0.7876
                  #      (narrow -- if the overlay looks dorsoventrally
                  #      mirrored, AIR is the alternative).
                  #      Prior values SAL and ALS were both wrong: SAL declared
                  #      axis0=D-V and axis1=A-P, i.e. the first two axes
                  #      transposed, so ANTs was warping a brain lying 90 deg
                  #      off the target.
SIDE="${SIDE:-rh}"         # -s : hemisphere imaged (rh/lh) — this sample is RIGHT hemisphere only
HEMI="${HEMI:-combined}"   # -m : combined = L/R share label IDs (what a single-hemisphere
                  #      sample wants). Use "split" ONLY for bilateral samples
                  #      where left and right must be distinguished. NOTE: MIRACL's
                  #      reg script hardcodes "combined" in one internal path check,
                  #      so "split" is untested in this container build.
LBL_VOX="${LBL_VOX:-10}"      # -v : Allen label resolution in um (10, 25 or 50)
OLF_BULB="${OLF_BULB:-0}"      # -b : olfactory bulb included in the image (0/1)
REG_MOSAIC="${REG_MOSAIC:-0}"    # -f : write the QC mosaic PNG. Keep 0 — CreateTiledMosaic
                  #      segfaults in this container build and aborts the run.

# ---- SEGMENTATION (stage 03) ------------------------------------------------
SEG_TYPE="${SEG_TYPE:-cfos}"   # -t : MIRACL CLARITY segmentation macro.
                  #        cfos    - cFos nuclei; converts to 8-bit WITHOUT rescaling
                  #        cfos2   - same detector, but auto-scales the 16-bit range
                  #                  first (use this if `norm.tif` looks blown out)
                  #        virus / sparse / nuclear - other stains
SEG_PREFIX="${SEG_PREFIX:-}"     # -p : filename substring to select files, e.g. "C01". Leave empty
                  #      when the channel folder already holds a single channel.
SEG_MACRO_OVERRIDE="${SEG_MACRO_OVERRIDE:-}"   # optional: path on Sherlock to your own edited copy of
                        # miracl_seg_neurons_clarity_3D_${SEG_TYPE}.ijm. It gets
                        # bind-mounted over MIRACL's copy so you can retune
                        # radpx / minobjsz / maxobjsz. Empty = use MIRACL's.
SEG_MACRO="/code/miracl/seg/miracl_seg_neurons_clarity_3D_${SEG_TYPE}.ijm"
JAVA_MEM="${JAVA_MEM:-200g}"   # Fiji JVM heap for stage 03 (--mem). Must fit in the job's --mem.
FIJI_BIN="${FIJI_BIN:-/opt/fiji/ImageJ-linux64}"   # Fiji launcher inside the container.
                  # MIRACL's `seg clar` wrapper looks for it under
                  # $MIRACL_HOME/../depends and on PATH as "Fiji"; in this image it
                  # lives here instead, and the wrapper launches it WITHOUT
                  # --headless. This container has no X server (no Xvfb, no
                  # xvfb-run), so stage 03 invokes MIRACL's macro through this
                  # launcher directly with --headless. Same macro, same outputs —
                  # only the launcher differs.

# ---- VOXELIZE / COUNTS (stages 04-05) ---------------------------------------
VOX_DOWN="${VOX_DOWN:-$DOWN}"  # -d for `seg voxelize`: downsample of the segmentation before
                  # counting. MIRACL recommends matching the registration
                  # downsample. Applies to Z as well as X/Y — see the README note
                  # on what the resulting Count column does and does not mean.

# ---- VIEWING (stage 07, optional) -------------------------------------------
VIEW_DOWN="${VIEW_DOWN:-2}"     # in-plane downsample for the signal-channel NIfTI you scroll
                  # through in ITK-SNAP. 1 = full res (huge), 4 = light.

# ---- DERIVED (do not edit) --------------------------------------------------
SAMPLE_DIR="$MAPDIR/$SAMPLE"
CH_AUTO="$SAMPLE_DIR/$AUTO_CH_DIR"          # registered (stage 01-02)
CH_SIGNAL="$SAMPLE_DIR/$SIGNAL_CH_DIR"      # segmented  (stage 03)

# `miracl conv tiff_nii` writes into a conv_final/ subfolder and names the file
# <outnii>_<DD>x_down_<chan>_chan.nii.gz with a zero-padded 2-digit ratio.
DOWN_PAD="$(printf '%02d' "$DOWN")"
AUTO_NII="$SAMPLE_DIR/conv_final/${SAMPLE}_${DOWN_PAD}x_down_${AUTO_CH_NAME}_chan.nii.gz"
VIEW_DOWN_PAD="$(printf '%02d' "$VIEW_DOWN")"
SIGNAL_NII="$SAMPLE_DIR/conv_final/${SAMPLE}_${VIEW_DOWN_PAD}x_down_${SIGNAL_CH_NAME}_chan.nii.gz"

REG_DIR="$SAMPLE_DIR/clar_allen_reg"
REG_FINAL="$SAMPLE_DIR/reg_final"
# Allen labels resampled into the oriented CLARITY space. This is the label
# volume `seg feat_extract` consumes; it is produced before the (broken) full-res
# per-slice label export, so stage 02 completes usefully even with -f 0.
REG_LABELS_VOX="$REG_FINAL/annotation_hemi_${HEMI}_${LBL_VOX}um_clar_vox.tif"

# `miracl seg clar` writes alongside the input channel folder, not under it.
SEG_DIR="$SAMPLE_DIR/segmentation_${SEG_TYPE}"
SEG_TIF="$SEG_DIR/seg_${SEG_TYPE}.tif"              # labelled cells
SEG_BIN_TIF="$SEG_DIR/seg_bin_${SEG_TYPE}.tif"      # binarized cells
# `seg voxelize` writes .tiff (two f's) even though MIRACL's own flow script and
# docs say .tif, and older builds prefix it with "bin". Rather than guess, stages
# 04/05/07 resolve whichever file actually got written.
VOX_SEG_TIF="$SEG_DIR/voxelized_seg_${SEG_TYPE}.tiff"   # the expected name
resolve_vox_seg() {
  local c
  for c in "$SEG_DIR/voxelized_seg_${SEG_TYPE}".tiff \
           "$SEG_DIR/voxelized_seg_${SEG_TYPE}".tif \
           "$SEG_DIR/voxelized_seg_bin_${SEG_TYPE}".tiff \
           "$SEG_DIR/voxelized_seg_bin_${SEG_TYPE}".tif; do
    [[ -f "$c" ]] && { printf '%s\n' "$c"; return 0; }
  done
  return 1
}
FEATURES_CSV="$SEG_DIR/clarity_segmentation_features_ara_labels.csv"
QC_DIR="$SEG_DIR/qc"

# The `miracl` entry point imports ACE/MONAI for EVERY subcommand, and dies with
# "Too many open files" whenever the fd limit is low — always on login nodes, and
# in jobs if the hard limit is tight. MIRACL's underlying scripts are the same
# code without that import chain and take identical flags, so each stage probes
# the CLI once and falls back to the script if it is broken.
MIRACL_TIFFNII_PY="/code/miracl/conv/miracl_conv_convertTIFFtoNII.py"
MIRACL_VOXELIZE_PY="/code/miracl/seg/miracl_seg_voxelize_parallel.py"
MIRACL_FEATEXTRACT_PY="/code/miracl/seg/miracl_seg_feat_extract.py"

# miracl_cli_ok <module>  -> true if `miracl <module> -h` runs without crashing
miracl_cli_ok() {
  local out
  out=$(singularity exec "$SIF" miracl "$1" -h 2>&1 || true)
  [[ "$out" != *Traceback* ]]
}

# miracl_cmd <module> <subcommand> <fallback_script>  -> prints the command to run
miracl_cmd() {
  if miracl_cli_ok "$1"; then
    printf '%s\n' miracl "$1" "$2"
  else
    echo "NOTE: the miracl CLI is unusable in this image; running $3 directly." >&2
    printf '%s\n' python "$3"
  fi
}
