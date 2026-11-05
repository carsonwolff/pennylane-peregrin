# qbench: batched small-circuit benchmark for PennyLane backends

Measures throughput (circuits/s) for MRI-style patch preprocessing: one 2 to 8 qubit
circuit per image patch, run in batches of 100 to 100,000+.

Circuit (see `workload.py`): amplitude-load a 2^n-pixel patch, then `n_layers` x
(RY, RZ on every qubit, CNOT ring), then measure <Z_i> and <X_i> on every qubit (2n outputs).
If your paper's circuit differs, edit `apply_circuit` and the matching hand-written JAX
code in `backends.py`. The correctness check will catch it if the two disagree.

## Backends

| name | what it is |
|---|---|
| `default.qubit` | numpy, native parameter broadcasting, no jit. Also the correctness reference. |
| `dq-jax-cpu` / `dq-jax-gpu` | default.qubit + broadcasting + `jax.jit` |
| `jax-hand-cpu` / `jax-hand-gpu` | hand-written batched state-vector sim, one `jax.jit` for the whole batch |
| `lightning.qubit` / `lightning.gpu` | plain QNode; broadcasting is split into one tape per circuit (the paper's setup) |
| `lightning.qubit+qjit` / `lightning.gpu+qjit` | Catalyst `qjit` + `catalyst.vmap` over the batch |
| `qulacs` / `qulacs-gpu` | `qulacs.simulator` via pennylane-qulacs |

Unavailable backends (missing package, no GPU) are skipped automatically. `python run_bench.py --list` shows what was found.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# GPU box, additionally:
pip install -r requirements-gpu.txt
```

## Run

```bash
python run_bench.py                                      # all available backends, n=2,4,6,8, B=100,1k,10k
python run_bench.py --backends jax-hand-gpu lightning.gpu+qjit --qubits 4 8 --batches 1000 100000
python run_bench.py --precision f32                     # f32 where the backend supports it
python run_bench.py --embed amplitude                   # reproduce the AmplitudeEmbedding path
python run_bench.py --threads 8                         # pin OMP_NUM_THREADS for lightning
python summarize.py results/<timestamp>                 # markdown tables + throughput.png
```

## Methodology

- **Isolation.** Every (backend, n, batch) case runs in a fresh subprocess, so JIT caches, GPU
  contexts, thread pools and memory never leak between cases, and a crash or hang in one backend
  is recorded instead of killing the sweep.
- **What is timed.** numpy in, numpy out, including host/device copies, because that is what a
  preprocessing pipeline pays. The first call (tracing, JIT, qjit compile) is reported separately
  as `first_call_s`; throughput uses the median of the steady-state reps.
- **Correctness.** Before the sweep, default.qubit in f64 computes reference outputs for the first
  64 circuits per qubit count. Every case is compared against it (tolerance 1e-9 in f64, 1e-4 in
  f32) and mismatches are flagged in both the console and the summary. A fast wrong answer never
  looks like a win.
- **Budgets.** `--budget` caps the timed reps per case (min 3); `--timeout` hard-kills a case.
  Larger batches are skipped when a smaller one projects past the timeout.
- **Provenance.** `env.json` records CPU, GPU (nvidia-smi), package versions, thread env vars,
  CLI args and git commit.

## Things the harness already caught

1. **`AmplitudeEmbedding` is not free everywhere.** default.qubit and plain lightning load the
   state directly, but under `qjit` and on qulacs it decomposes into a Möttönen cascade
   (about 520 gates at 8 qubits). With `normalize=False` and pre-normalized rows, `qml.StatePrep`
   is mathematically identical and loads natively everywhere, so it is the default here.
   `--embed amplitude` keeps the old path so you can quantify what it cost.
2. **qulacs runs each circuit twice.** <X_i> and <Z_i> on the same wire don't commute, so
   PennyLane splits the tape. That is real plugin behavior and stays in the numbers, but it is
   worth one sentence in the paper.
3. **`qjit` + `catalyst.vmap` was slower than plain lightning.qubit** on CPU (about 140 µs per
   2-qubit circuit), so the per-circuit cost in Catalyst's batch loop dominates at this size.
   Worth re-checking on the GPU box and against newer Catalyst releases before claiming it.

## Precision notes

`--precision f32` applies to the JAX backends (complex64) and plain lightning (`c_dtype=complex64`).
Catalyst and qulacs run f64 regardless; each result row records `precision_effective`.
For the headline comparison against the planned f32 Rust/CUDA device, report both an f32 table
and an f64 table rather than mixing them.
