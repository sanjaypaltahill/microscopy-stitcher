#!/bin/bash
# =============================================================================
#  Ten-minute check that the Fiji segmentation actually runs in this container.
#
#  Copies a handful of slices from the middle of the signal channel into a
#  throwaway sample under $SCRATCH and submits stage 03 against it with a small
#  heap and a short time limit. If it produces seg_bin, the real run is a
#  scale-up question rather than a "does this work at all" question.
#
#  Usage:  ./smoketest_seg.sh [N_SLICES]     (default 10)
# =============================================================================
set -euo pipefail
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
export PIPELINE_DIR="$PWD"
source ./config.sh

N="${1:-10}"
TEST_SAMPLE="seg_smoketest"
TEST_ROOT="${SCRATCH:?SCRATCH is not set}/$TEST_SAMPLE"
TEST_DIR="$TEST_ROOT/$SIGNAL_CH_DIR"

[[ -d "$CH_SIGNAL" ]] || { echo "no such folder: $CH_SIGNAL"; exit 1; }
mapfile -t slices < <(find "$CH_SIGNAL" -maxdepth 1 -name '*.tif' | sort)
(( ${#slices[@]} )) || { echo "no .tif files in $CH_SIGNAL"; exit 1; }

# Take them from the middle of the stack — the ends are often empty.
start=$(( ${#slices[@]} / 2 - N / 2 )); (( start < 0 )) && start=0
rm -rf "$TEST_ROOT"; mkdir -p "$TEST_DIR"
cp "${slices[@]:start:N}" "$TEST_DIR/"
echo "copied $(ls "$TEST_DIR" | wc -l) slices to $TEST_DIR"

# Ten slices need only a few GB. Keep the ask small so it schedules quickly;
# override from the environment if the queue or the data says otherwise.
SMOKE_PART="${SMOKE_PART:-normal}"
SMOKE_MEM="${SMOKE_MEM:-16G}"
SMOKE_CPUS="${SMOKE_CPUS:-4}"
SMOKE_TIME="${SMOKE_TIME:-00:45:00}"
SMOKE_HEAP="${SMOKE_HEAP:-12g}"

echo "requesting: $SMOKE_PART, $SMOKE_MEM, $SMOKE_CPUS cpus, $SMOKE_TIME (Fiji heap $SMOKE_HEAP)"

jid=$(sbatch --parsable \
  --job-name=seg_smoke \
  --partition="$SMOKE_PART" --mem="$SMOKE_MEM" \
  --cpus-per-task="$SMOKE_CPUS" --time="$SMOKE_TIME" \
  --export=ALL,PIPELINE_DIR="$PWD",MAPDIR="$SCRATCH",SAMPLE="$TEST_SAMPLE",JAVA_MEM="$SMOKE_HEAP" \
  03_seg_clar.sbatch)

cat <<EOF

Submitted smoke test as job $jid.

Watch it:        squeue -j $jid
Read the log:    cat seg_clar-$jid.out
Success looks like:
  $TEST_ROOT/segmentation_${SEG_TYPE}/seg_bin_${SEG_TYPE}.tif

If that file appears, run ./submit_all.sh for the real sample.
If it fails, send the log — the error names the plugin that broke.
EOF
