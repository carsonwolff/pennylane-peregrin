#!/usr/bin/env bash
# Phase 1 CPU sweep on the local machine. Every run lands in results/<tag>-<label>/.
# Budget and timeout match the handoff's reference run (results/stateprep).
# Run from anywhere; paths are relative to bench/.
# Usage: [BACKENDS="a b"] [LABEL=suffix] sweeps/cpu.sh [threads]
#   threads defaults to the physical core count; BACKENDS limits the run to those backends.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-../.venv/bin/python}
T=${1:-$(lscpu -p=Core,Socket | grep -v '^#' | sort -u | wc -l)}
TAG=$(date +%Y%m%d-%H%M)
B="100 1000 10000 100000"
COMMON="--threads $T --budget 10 --timeout 300 --batches $B${BACKENDS:+ --backends $BACKENDS}"
run() { local label=$1${LABEL:+-$LABEL}; shift; "$PY" run_bench.py $COMMON --out "results/$TAG-$label" "$@"; "$PY" summarize.py "results/$TAG-$label" > /dev/null; }

run owner-mri-f64      --circuit owner-mri
run generic-f64        --circuit generic
run owner-mri-f32      --circuit owner-mri --precision f32
run generic-f32        --circuit generic   --precision f32
run owner-mri-amp-f64  --circuit owner-mri --embed amplitude
run demo-f64           --circuit pennylane-demo
run angle-f64          --circuit angle-trainable
run amp-sel-f64        --circuit amp-sel
