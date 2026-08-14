#!/bin/bash
#SBATCH --job-name=view_volume
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=normal
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "${PIPELINE_DIR:-$HOME/pipeline}/config.sh"
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

# -----------------------------------------------------------------------------
# STAGE 07 — Build a signal-channel volume you can scroll through (optional)
# -----------------------------------------------------------------------------
# Same MIRACL converter as stage 01, pointed at the signal channel, so you get a
# 3-D NIfTI to open in ITK-SNAP next to stage 04's voxelized_seg_*.nii.gz. Both
# carry real voxel spacings, so ITK-SNAP lines them up in physical space even
# though the segmentation grid is coarser.
#
# For a per-cell look at full resolution, skip this and open
# $SEG_DIR/seg_${SEG_TYPE}.tif directly in Fiji alongside the raw slices.
# -----------------------------------------------------------------------------

mapfile -t CONV < <(miracl_cmd conv tiff_nii "$MIRACL_TIFFNII_PY")

singularity exec "$SIF" "${CONV[@]}" \
  -f "$CH_SIGNAL" \
  -w "$SAMPLE_DIR" \
  -o "$SAMPLE" \
  -d "$VIEW_DOWN" \
  -ch "$SIGNAL_CH_NAME" \
  -vx "$VX" -vz "$VZ" -dz 0

echo "----- verify -----"
[[ -f "$SIGNAL_NII" ]] || { echo "FAIL: expected $SIGNAL_NII"; exit 1; }
singularity exec "$SIF" PrintHeader "$SIGNAL_NII" 2>&1 | grep -E "Size|Spacing"

echo "Done. Copy both to your laptop and open in ITK-SNAP:"
echo "  scp '$USER@sherlock.stanford.edu:$SIGNAL_NII' ~/Downloads/"
if vox=$(resolve_vox_seg); then
  echo "  scp '$USER@sherlock.stanford.edu:${vox%.*}.nii.gz' ~/Downloads/"
fi
