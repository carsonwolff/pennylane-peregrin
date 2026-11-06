# Design

> **Unreviewed draft.** Claude Code wrote this from the code and the commit history. Each
> decision is marked with who made it. The project was archived in November 2026 before
> Carson reviewed this document, so every entry marked "not yet reviewed" stays that way. It
> describes the reasoning behind the code as it was written, not decisions Carson confirmed.

peregrin simulates large batches of small circuits: 1 to 10 qubits, thousands to hundreds of
thousands of circuits per call. The case it is built for is quanvolutional preprocessing,
where one circuit runs once per image patch and only the inputs change between circuits. The
decisions below all follow from that case, in which per-circuit overhead matters more than
anything else.

## Scope

**CPU first, GPU later.** Carson's decision. The device is `peregrin.cpu`, and the name
`peregrin.gpu` is kept for a later GPU backend.

**f32 and f64 both supported.** Carson's decision. f64 matches `default.qubit` to 1e-12, and
f32 to 1e-4. f32 roughly doubles the number of circuits per SIMD register.

**Analytic mode only, for now.** This is the current state, not a decision yet. Shots are
open in #3.

## The API: one batch-first call

The core takes one circuit template, a `(batch, d_x)` array of per-circuit inputs, and a
vector of weights shared by the whole batch. It returns `(batch, n_outputs)`. Each gate
parameter reads an input column, reads a shared weight, or is a constant.

- **Why:** in quanvolution, every circuit in a batch has the same structure, so the
  structure only needs to be processed once. Gate matrices that depend only on shared weights
  are computed once per call, not once per circuit.
- **Alternatives:**
  - One call per circuit, like most simulators. Per-call overhead then dominates for circuits
    this small.
  - A batch of different circuits. That is more general, but it gives up most of the
    precomputation.
- **Status:** proposed and implemented by Claude Code; not yet reviewed by Carson.

## Tiles of 8 circuits per SIMD lane group

A state of 4 to 256 amplitudes is too short to fill SIMD registers well. So the core stores 8
circuits' states side by side: one row per amplitude, holding that amplitude for all 8
circuits, with real and imaginary parts in separate arrays. Every gate then becomes a loop
over 8 lanes, which the compiler turns into SIMD instructions.

- **Why 8:** it is two AVX2 registers in f64 and one in f32. 8 was tried first, and other
  widths have not been measured yet.
- **Runtime dispatch:** AVX2 and FMA are detected when the program runs, so a single wheel
  works on any x86-64 CPU. Other CPUs, including Apple Silicon, run the same code compiled
  without those features. That code is tested in CI but has not been benchmarked.
- **Limit:** tiles are used up to 10 qubits. Above that, the core runs one circuit at a time,
  because a tile of large states no longer fits in cache.
- **Status:** proposed and implemented by Claude Code; not yet reviewed by Carson.

## Permutations move no data

A run of CNOT and SWAP gates permutes amplitudes. Instead of moving data, the core records
which row holds which amplitude when the program is built, and every later gate and
measurement uses precomputed row lists. CNOT chains, common in these circuits, then cost
nothing at run time.

- **Status:** proposed and implemented by Claude Code; not yet reviewed by Carson.

## Real-state tracking

If the state starts real (real amplitudes, or |0>) and every gate so far is real (H, RY, X,
Z, CNOT and so on), all imaginary parts are zero. The program tracks this when it is built
and skips the imaginary arithmetic until the first complex gate.

- **Status:** proposed and implemented by Claude Code; not yet reviewed by Carson.

## Gate fusion

Consecutive single-qubit gates on the same wire are multiplied into one 2x2 matrix. If none
of them depends on per-circuit inputs, the product is computed once per call.

- **Status:** proposed and implemented by Claude Code; not yet reviewed by Carson.

## Vectorized sin and cos

Per-circuit angles need sin and cos for every circuit. The library functions handle one
value at a time and became the bottleneck. The core reduces the angle by multiples of pi/2,
using fdlibm's three-part constant, then applies Cephes' polynomials across all 8 lanes at
once. This is accurate to about 1 ulp. Angles of 1e5 or more, and NaN, fall back to the
library functions.

- **Status:** proposed and implemented by Claude Code; not yet reviewed by Carson.

## The PennyLane device

- **Batching:** a broadcast tape (parameters with a leading batch dimension, or a batch of
  states) becomes one core call. Each broadcast parameter becomes an input column, and each
  scalar parameter becomes a shared weight.
- **Caching:** compiled circuits are cached by structure, so changing the weights between
  calls reuses them.
- **Measurements:** observables are converted to sums of Pauli words, and each distinct word
  is measured once. `var` is computed from the expectations of H and H^2. Matrix observables
  are expanded into Pauli words, which is limited to 4 wires because the number of terms
  grows as 4^k.
- **Gradients:** parameter-shift, chosen by PennyLane. Device gradients (adjoint) would be
  faster for training, but are not written yet.
- **Unsupported input:** anything unsupported is decomposed by PennyLane or rejected with a
  `DeviceError`. It is never silently given a wrong answer.
- **Status:** proposed and implemented by Claude Code. Carson chose the name. Not yet
  reviewed by Carson otherwise.

## Benchmarks

- **Separate from the tool:** the harness is in `bench/`, separate from the simulator.
  Carson's decision.
- **Timing boundary:** numpy in, numpy out, the same for every backend.
- **Each case in its own process:** so no backend's caches or thread pools affect another.
- **Correctness first:** every case is checked against `default.qubit`, and a fast wrong
  answer never counts.
- **The strongest baseline:** the hand-written JAX simulator was made faster before any
  comparison, and its slower first numbers are kept in the results.
- **Status:** these rules were set by Claude Code and are written down in
  `bench/RESULTS.md`. Not yet reviewed by Carson.
