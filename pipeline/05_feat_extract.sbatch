#!/bin/bash
#SBATCH --job-name=feat_extract
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "${PIPELINE_DIR:-$HOME/pipeline}/config.sh"
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

# -----------------------------------------------------------------------------
# STAGE 05 — Summarize the segmentation per Allen region  (final deliverable)
# -----------------------------------------------------------------------------
# `miracl seg feat_extract` joins the two halves of the pipeline: the voxelized
# cFos segmentation (stage 04, from the SIGNAL channel) and the Allen labels
# warped into this sample's space (stage 02, via the AUTOFLUORESCENCE channel).
#
# feat_extract resamples the label volume onto the segmentation grid itself
# (including the X/Y swap MIRACL's registration introduces), so the two do not
# need to share dimensions — they only need to share orientation, which they do
# because both derive from the same padded TIFF stacks and the same ORIENT code.
#
# Output: clarity_segmentation_features_ara_labels.csv, written next to the
# voxelized segmentation, with per-region Count, Density and cell-volume stats.
# -----------------------------------------------------------------------------

VOX_SEG=$(resolve_vox_seg) || { echo "FAIL: no voxelized segmentation in $SEG_DIR — run stage 04."; exit 1; }
[[ -f "$REG_LABELS_VOX" ]] || { echo "FAIL: missing $REG_LABELS_VOX — run stage 02."; exit 1; }
echo "segmentation: $VOX_SEG"
echo "labels:       $REG_LABELS_VOX"

# MIRACL's own `flow seg` looks for voxelized_seg_*.tif while voxelize writes
# .tiff, so we pass the real paths explicitly instead of using the flow wrapper.
mapfile -t FEAT < <(miracl_cmd seg feat_extract "$MIRACL_FEATEXTRACT_PY")

singularity exec "$SIF" "${FEAT[@]}" \
  -s "$VOX_SEG" \
  -l "$REG_LABELS_VOX"

echo "----- verify -----"
if [[ ! -f "$FEATURES_CSV" ]]; then
  echo "FAIL: expected $FEATURES_CSV"
  exit 1
fi
echo "Rows: $(( $(wc -l < "$FEATURES_CSV") - 1 ))"
echo "----- top regions by count -----"
singularity exec "$SIF" python - "$FEATURES_CSV" <<'PY'
import sys, pandas as pd
df = pd.read_csv(sys.argv[1])
print(df.sort_values("Count", ascending=False)
        .head(20)[["LabelAbrv", "LabelName", "Count", "Density"]]
        .to_string(index=False))
print("\ntotal counted:", int(df.Count.sum()), "over", (df.Count > 0).sum(), "regions")
PY
