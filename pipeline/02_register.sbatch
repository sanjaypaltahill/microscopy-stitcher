#!/bin/bash
#SBATCH --job-name=clar_reg
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "${PIPELINE_DIR:-$HOME/pipeline}/config.sh"
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

# -----------------------------------------------------------------------------
# STAGE 02 — CLARITY-Allen registration on the AUTOFLUORESCENCE channel
#            (and warp the Allen labels into this sample's space)
# -----------------------------------------------------------------------------
# Input is $AUTO_NII, built in stage 01 from $AUTO_CH_DIR. The signal channel is
# deliberately NOT used here.
#
# CRITICAL #1 (caching): MIRACL reuses any intermediate that already exists. A
#   re-run after fixing inputs will silently reuse a STALE clar.nii.gz and fail
#   the same way. Always start clean:
rm -rf "$REG_DIR" "$REG_FINAL"

# CRITICAL #2 (false success): the wrapper prints "Good job ... done" and exits 0
#   even when ANTs produced NO warp. We verify the transforms below and fail
#   loudly if they're missing (a real run takes ~30-40 min, not seconds).
# -----------------------------------------------------------------------------

# We call MIRACL's registration SHELL SCRIPT directly instead of `miracl reg
# clar_allen` for one reason the CLI can't give us: -f 0, to SKIP the QC mosaic.
# That PNG step (CreateTiledMosaic) reliably segfaults in this build and aborts
# the run; the CLI hardcodes -f 1, so we bypass it. Every other flag mirrors the
# CLI and comes from config.sh.
singularity exec "$SIF" \
  /code/miracl/reg/miracl_reg_clar-allen.sh \
  -i "$AUTO_NII" \
  -r "$SAMPLE_DIR" \
  -c "$CH_AUTO" \
  -o "$ORIENT" -s "$SIDE" -m "$HEMI" -v "$LBL_VOX" -b "$OLF_BULB" -f "$REG_MOSAIC" \
  -l None -p 0 -a None -w 0 -n -999999 -x -999999

echo "----- verify -----"
missing=0
for f in "$REG_DIR/allen_clar_ants0GenericAffine.mat" \
         "$REG_DIR/allen_clar_ants1Warp.nii.gz" \
         "$REG_LABELS_VOX"; do
  [[ -f "$f" ]] || { echo "MISSING: $f"; missing=1; }
done
if (( missing )); then
  echo "FAIL: registration did not produce usable output. Check the log for a"
  echo "      too-thin Z stack (stage 01) or a wrong ORIENT code."
  exit 1
fi
echo "OK: warps + $ORIENT-oriented Allen labels present."
singularity exec "$SIF" PrintHeader "$REG_LABELS_VOX" 2 | sed 's/^/label volume dims: /'
echo "Stage 05 summarizes the segmentation over this label volume."

# Note: the per-slice full-resolution label export
# ($REG_FINAL/annotation_hemi_${HEMI}_${LBL_VOX}um_tiff_clar/) is broken in this
# container build and stays empty. Nothing downstream needs it — feat_extract
# resamples $REG_LABELS_VOX to the segmentation grid itself.
