#!/usr/bin/env bash
# Phase 1 CPU sweep on the local machine. Every run lands in results/<tag>-<label>/.
# Usage: sweeps/cpu.sh [threads]   (threads defaults to the physical core count)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
T=${1:-$(lscpu -p=Core,Socket | grep -v '^#' | sort -u | wc -l)}
TAG=$(date +%Y%m%d-%H%M)
B="100 1000 10000 100000"
run() { local label=$1; shift; "$PY" run_bench.py --threads "$T" --out "results/$TAG-$label" "$@"; "$PY" summarize.py "results/$TAG-$label" > /dev/null; }

run generic-f64        --circuit generic   --batches $B
run generic-f32        --circuit generic   --batches $B --precision f32
run owner-mri-f64      --circuit owner-mri --batches $B
run owner-mri-f32      --circuit owner-mri --batches $B --precision f32
run owner-mri-amp-f64  --circuit owner-mri --batches $B --embed amplitude
run demo-f64           --circuit pennylane-demo  --batches $B
run angle-f64          --circuit angle-trainable --batches $B
run amp-sel-f64        --circuit amp-sel         --batches $B
