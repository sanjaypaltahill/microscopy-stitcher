#!/bin/bash
#SBATCH --job-name=qc_overlay
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "${PIPELINE_DIR:-$HOME/pipeline}/config.sh"
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

# -----------------------------------------------------------------------------
# STAGE 06 — Visual QC of the segmentation (optional, produces no numbers)
# -----------------------------------------------------------------------------
# MIRACL has no built-in overlay for the CLARITY segmentation route, so this is
# a local helper: detected cells in red over the raw signal channel, plus
# intensity stats that reveal a bad 8-bit conversion. Run it after stage 03 and
# before you trust stage 05.
# -----------------------------------------------------------------------------

singularity exec "$SIF" python "${PIPELINE_DIR:-$HOME/pipeline}/qc_overlay.py" \
  "$CH_SIGNAL" "$SEG_BIN_TIF" "$QC_DIR"

echo "Done. Copy the images to your laptop to view:"
echo "  scp '$USER@sherlock.stanford.edu:$QC_DIR/*.png' ~/Downloads/"
