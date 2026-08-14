#!/bin/bash
#SBATCH --job-name=tiff_nii
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "${PIPELINE_DIR:-$HOME/pipeline}/config.sh"
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

# -----------------------------------------------------------------------------
# STAGE 01 — Autofluorescence TIFF stack  ->  downsampled NIfTI
#            (the input to CLARITY-Allen registration)
# -----------------------------------------------------------------------------
# The AUTOFLUORESCENCE channel is the one that gets registered: it shows brain
# anatomy rather than sparse cFos puncta, so it is what the Allen template can
# actually be matched against. The signal channel is never registered — it is
# segmented in stage 03 and mapped through these labels in stage 05.
#
# CRITICAL: pass the true voxel size (-vx/-vz) and -dz 0.
#   MIRACL's defaults assume 5 um in X/Y AND Z. With a 40 um Z step, the default
#   silently DECIMATES the Z-stack (~110 slices -> ~15) and stamps a bogus
#   spacing. Registration (stage 02) then dies producing no warp. Keeping -dz 0
#   preserves every Z plane; -vx/-vz set the correct physical scale.
# -----------------------------------------------------------------------------

mapfile -t CONV < <(miracl_cmd conv tiff_nii "$MIRACL_TIFFNII_PY")

singularity exec "$SIF" "${CONV[@]}" \
  -f "$CH_AUTO" \
  -w "$SAMPLE_DIR" \
  -o "$SAMPLE" \
  -d "$DOWN" \
  -ch "$AUTO_CH_NAME" \
  -vx "$VX" -vz "$VZ" -dz 0

echo "----- verify -----"
if [[ ! -f "$AUTO_NII" ]]; then
  echo "FAIL: expected $AUTO_NII — check the -o/-d/-ch values in config.sh."
  exit 1
fi
singularity exec "$SIF" PrintHeader "$AUTO_NII" 2>&1 | grep -E "Size|Spacing"
echo "Expect one Z plane per input slice (NOT ~15) and a Z spacing of $(awk "BEGIN{print $VZ/1000}")."
