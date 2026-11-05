# qbench: batched small-circuit benchmark for PennyLane backends

This is the benchmark harness behind peregrin's numbers (see the [top-level README](../README.md)
for the simulator itself, and [RESULTS.md](RESULTS.md) for the CPU results). It is independent
of peregrin: every other backend runs without it. Run everything below from this `bench/` folder.

Measures throughput (circuits/s) for MRI-style patch preprocessing: one 2 to 8 qubit
circuit per image patch, run in batches of 100 to 100,000+.

Circuits (`--circuit`, see `circuits.py`):

| name | qubits | what it is |
|---|---|---|
| `generic` (default) | 2, 4, 6, 8 | the original qbench circuit: amplitude-load a 2^n-pixel patch, `n_layers` x (RY, RZ on every qubit, CNOT ring), <Z_i> and <X_i> on every qubit |
| `owner-mri` | 2, 4 | the paper's circuits, copied from `parallel_quantum_processing.py`; per-patch angles from patch statistics |
| `pennylane-demo` | 4 | the PennyLane quanvolution demo: RY(pi x), `RandomLayers` (seed 42), <Z_i> |
| `angle-trainable` | 4 | RY(x), trainable RX, open CNOT chain, <Z_i> (ResQuNNs) |
| `amp-sel` | 5 | amplitude embedding, 3 x (RX RY RZ, open CNOT chain), <Z_i> |

Each circuit has a PennyLane body (what the QNode backends run) and a gate-list template (what
`jax-hand` and `peregrin` run). `tests/test_circuits.py` checks every template against
default.qubit to 1e-12.

## Backends

| name | what it is |
|---|---|
| `default.qubit` | numpy, native parameter broadcasting, no jit. Also the correctness reference. |
| `dq-jax-cpu` / `dq-jax-gpu` | default.qubit + broadcasting + `jax.jit` |
| `jax-hand-cpu` / `jax-hand-gpu` | hand-written batched state-vector sim, one `jax.jit` for the whole batch |
| `lightning.qubit` / `lightning.gpu` | plain QNode; broadcasting is split into one tape per circuit (the paper's setup) |
| `lightning.qubit+qjit` / `lightning.gpu+qjit` | Catalyst `qjit` + `catalyst.vmap` over the batch |
| `qulacs` / `qulacs-gpu` | `qulacs.simulator` via pennylane-qulacs |
| `peregrin-cpu` | peregrin (`../crates/`) through its Python bindings; threads follow `--threads` |

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
python run_bench.py --circuit owner-mri                 # the paper's circuits
python summarize.py results/<timestamp>                 # markdown tables + throughput.png
python compare.py results/<a> results/<b> ...           # peregrin vs the best other backend, per cell
sweeps/cpu.sh                                           # the full CPU sweep behind RESULTS.md
```

peregrin needs a Rust toolchain to build: `pip install maturin && (cd ../crates/python && maturin develop --release)`.

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
3. **`qjit` + `catalyst.vmap` vs plain lightning.qubit depends on the machine.** On the
   handoff's Xeon, qjit was slower (about 140 µs per 2-qubit circuit). On the i5-6300U laptop
   (RESULTS.md) it is 1.2 to 5.4x faster in 66 of 68 cells, and slightly slower only at n=8,
   B=100k. Both are far
   behind the batched simulators, so this does not change any conclusion.

## Precision notes

`--precision f32` applies to the JAX backends (complex64), peregrin, and plain lightning (`c_dtype=complex64`).
Catalyst and qulacs run f64 regardless; each result row records `precision_effective`.
For the headline comparison against the planned f32 Rust/CUDA device, report both an f32 table
and an f64 table rather than mixing them.
