#!/bin/bash
#SBATCH --job-name=seg_clar
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --partition=bigmem
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "${PIPELINE_DIR:-$HOME/pipeline}/config.sh"
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

# -----------------------------------------------------------------------------
# STAGE 03 — MIRACL CLARITY segmentation of the SIGNAL channel (cFos)
# -----------------------------------------------------------------------------
# This runs MIRACL's own Fiji macro, miracl_seg_neurons_clarity_3D_${SEG_TYPE}.ijm
# — the exact file `miracl seg clar` runs — but launches it ourselves.
#
# WHY NOT `miracl seg clar`: its wrapper starts Fiji as `Fiji -macro ...`, with no
# --headless. This container has no X server (no Xvfb, no xvfb-run), so that dies
# before the macro starts. The wrapper also looks for Fiji under
# $MIRACL_HOME/../depends or as "Fiji" on PATH; in this image the launcher is at
# $FIJI_BIN. Everything the wrapper does besides that — create the output folder,
# pick the macro by type, pass the folder (and optional prefix) as the macro
# argument — is reproduced below. Same segmentation, same outputs.
#
# Notes:
#   * CPU only, but memory-hungry: the macro loads the whole stack into the JVM
#     and keeps several intermediates. Keep JAVA_MEM inside the job's --mem.
#   * Outputs land NEXT TO the input channel folder, in $SEG_DIR:
#         seg_${SEG_TYPE}.tif      labelled cells
#         seg_bin_${SEG_TYPE}.tif  binarized cells  (what stage 04 consumes)
#         norm.tif, median_*.tif   intermediates — inspect these if results look
#                                  wrong, delete them to force a full recompute
#   * The macro RESUMES from whichever intermediates already exist and skips
#     entirely if seg_bin is present. To redo from scratch: rm -rf "$SEG_DIR"
# -----------------------------------------------------------------------------

if [[ -f "$SEG_BIN_TIF" ]]; then
  echo "Segmentation already present: $SEG_BIN_TIF"
  echo "Delete $SEG_DIR to recompute. Nothing to do."
  exit 0
fi

# The macro writes into this folder but does not create it (the wrapper does).
mkdir -p "$SEG_DIR"

# Bind arguments: optional user-tuned copy of the Fiji macro.
BIND=()
if [[ -n "$SEG_MACRO_OVERRIDE" ]]; then
  echo "Using custom macro: $SEG_MACRO_OVERRIDE -> $SEG_MACRO"
  BIND=(-B "$SEG_MACRO_OVERRIDE:$SEG_MACRO")
fi

# The macro reads its input as a single space-split argument string:
#   args[0] = tiff folder, args[1] = optional filename filter.
if [[ -n "$SEG_PREFIX" ]]; then
  MACRO_ARG="$CH_SIGNAL $SEG_PREFIX"
else
  MACRO_ARG="$CH_SIGNAL/"
fi

echo "macro:  $SEG_MACRO"
echo "input:  $MACRO_ARG"
echo "output: $SEG_DIR"

# ${arr[@]+...} keeps `set -u` happy when the array is empty (older bash).
singularity exec ${BIND[@]+"${BIND[@]}"} "$SIF" \
  "$FIJI_BIN" --headless --mem="$JAVA_MEM" \
  -macro "$SEG_MACRO" "$MACRO_ARG"

echo "----- verify -----"
if [[ ! -f "$SEG_BIN_TIF" ]]; then
  echo "FAIL: $SEG_BIN_TIF was not written."
  echo "      Check $SEG_DIR/seg_${SEG_TYPE}_log.txt and the Fiji output above."
  echo "      A plugin that needs a GUI would fail here with a HeadlessException;"
  echo "      the intermediates in $SEG_DIR show how far it got."
  ls -la "$SEG_DIR" || true
  exit 1
fi
ls -la "$SEG_DIR"
echo "OK: segmentation written. Run stage 06 before trusting it — look at norm.tif"
echo "    and the overlays. A flat white/black norm.tif means the 8-bit conversion"
echo "    clipped your 16-bit range; switch SEG_TYPE to cfos2 in config.sh."
