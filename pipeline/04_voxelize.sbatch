#!/bin/bash
#SBATCH --job-name=voxelize
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --partition=bigmem
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "${PIPELINE_DIR:-$HOME/pipeline}/config.sh"
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

# -----------------------------------------------------------------------------
# STAGE 04 — Voxelize the segmentation into a density map
# -----------------------------------------------------------------------------
# `miracl seg voxelize` turns each detected cell into a point and downsamples the
# volume by VOX_DOWN, producing a grid small enough for per-region summarization
# in stage 05.
#
# It is given seg_${TYPE}.tif but internally reads seg_bin_${TYPE}.tif — that is
# MIRACL's own convention, not a mistake here.
#
# READ THIS before interpreting the numbers: VOX_DOWN is applied to Z as well as
# X/Y, by nearest-neighbour subsampling. Our Z step is already 40 um, so a cell
# appears in only one plane and VOX_DOWN=5 keeps roughly one Z plane in five.
# The stage-05 Count column is therefore a consistent RELATIVE measure across
# regions and across samples processed identically — not an absolute cell tally.
# Lower VOX_DOWN (e.g. 2) for a finer grid at much higher cost in stage 05.
# -----------------------------------------------------------------------------

if [[ ! -f "$SEG_BIN_TIF" ]]; then
  echo "FAIL: $SEG_BIN_TIF missing — run stage 03 first."
  exit 1
fi

# voxelize skips itself if the output exists; remove it so config changes apply.
if old=$(resolve_vox_seg); then rm -f "$old" "${old%.*}.nii.gz"; fi

mapfile -t VOXELIZE < <(miracl_cmd seg voxelize "$MIRACL_VOXELIZE_PY")

singularity exec "$SIF" "${VOXELIZE[@]}" \
  -s "$SEG_TIF" \
  -v "$LBL_VOX" \
  -d "$VOX_DOWN" \
  -vx "$VX" -vz "$VZ"

echo "----- verify -----"
if ! out=$(resolve_vox_seg); then
  echo "FAIL: no voxelized_seg_* written in $SEG_DIR"
  ls -la "$SEG_DIR"
  exit 1
fi
ls -la "$out" "${out%.*}.nii.gz"
echo "OK: voxelized map ready for stage 05."
