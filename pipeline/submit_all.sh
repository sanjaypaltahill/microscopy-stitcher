#!/bin/bash
# =============================================================================
#  Submit the whole pipeline as one dependent Slurm chain.
#
#  Each job starts only when the ones it needs have finished successfully, so
#  you submit once and walk away. If a job fails, everything downstream of it is
#  cancelled by Slurm rather than running on bad input.
#
#      01 tiff_nii (Ch 0) ─> 02 register (Ch 0) ─┐
#                                                ├─> 05 feat_extract -> CSV
#      03 seg_clar (Ch 1) ─> 04 voxelize ────────┘
#                        └─> 06 qc_overlay
#
#  Usage:  ./submit_all.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
export PIPELINE_DIR="$PWD"
source ./config.sh

echo "sample:     $SAMPLE_DIR"
echo "register:   $CH_AUTO"
echo "segment:    $CH_SIGNAL"
echo "orientation: $ORIENT   hemisphere: $SIDE"
echo

fail=0
for d in "$SAMPLE_DIR" "$CH_AUTO" "$CH_SIGNAL"; do
  if [[ ! -d "$d" ]]; then echo "MISSING: $d"; fail=1; fi
done
[[ -f "$SIF" ]] || { echo "MISSING: $SIF"; fail=1; }
(( fail )) && { echo; echo "Fix the paths in config.sh and re-run."; exit 1; }

n_auto=$(find "$CH_AUTO" -maxdepth 1 -name '*.tif' | wc -l)
n_sig=$(find "$CH_SIGNAL" -maxdepth 1 -name '*.tif' | wc -l)
echo "slices: $n_auto autofluo, $n_sig signal"
if [[ "$n_auto" -ne "$n_sig" ]]; then
  echo "WARNING: the two channels have different slice counts — did padding finish?"
fi
echo

sub() {  # sub <script> [dependency]
  local script=$1 dep=${2:-}
  local args=(--parsable --export=ALL,PIPELINE_DIR="$PWD")
  [[ -n "$dep" ]] && args+=(--dependency=afterok:"$dep")
  sbatch "${args[@]}" "$script"
}

j01=$(sub 01_tiff_to_nii.sbatch)
j02=$(sub 02_register.sbatch "$j01")
j03=$(sub 03_seg_clar.sbatch)
j04=$(sub 04_voxelize.sbatch "$j03")
j06=$(sub 06_qc_overlay.sbatch "$j03")
j05=$(sub 05_feat_extract.sbatch "$j02:$j04")

cat <<EOF
submitted:
  $j01  01_tiff_to_nii    Ch 0 -> NIfTI
  $j02  02_register       Ch 0 -> Allen atlas          (after $j01)
  $j03  03_seg_clar       Ch 1 -> segmentation
  $j04  04_voxelize       voxelize                     (after $j03)
  $j06  06_qc_overlay     QC images                    (after $j03)
  $j05  05_feat_extract   per-region CSV               (after $j02 and $j04)

Watch:   squeue -u \$USER
Logs:    ls -lt ~/pipeline/*.out

Look at the QC images from job $j06 before trusting the CSV:
  scp '$USER@sherlock.stanford.edu:$QC_DIR/*.png' ~/Downloads/

Final result:
  $FEATURES_CSV
EOF
