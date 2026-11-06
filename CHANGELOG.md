# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Until 1.0, minor versions may
change the API.

## [Unreleased]

### Changed

- The project is archived as of November 2026. It works as documented but is no longer
  developed or maintained, and issues and pull requests are closed. The README, the
  contributing guide and `DESIGN.md` say so, and the planned GPU backend is dropped.

### Removed

- An incomplete benchmark run (`bench/results/20261104-2333-generic-f64`) that was stopped
  partway through and never summarized.

### Added

- `peregrin-core`, a Rust state-vector simulator for large batches of small circuits:
  - one batch-first call: a circuit template, per-circuit inputs and shared weights;
  - tiles of 8 circuits per SIMD lane group, with AVX2 chosen at runtime and a portable path
    everywhere else;
  - CNOT and SWAP runs folded into a row relabeling, so they move no data;
  - real-state tracking, gate fusion and a vectorized sin/cos;
  - f32 and f64, seeded sampling, and a JSON circuit format.
- Golden tests against PennyLane's `default.qubit` (1e-12 in f64), plus unit tests.
- `pennylane-peregrin`, Python bindings (PyO3 and maturin) imported as `peregrin`, with
  `peregrin.Circuit` for direct batched execution.
- The `peregrin.cpu` PennyLane device:
  - a broadcast tape runs as one call;
  - expval and var of Pauli words, Hamiltonians and matrix observables on up to 4 wires;
  - probs and state;
  - StatePrep, AmplitudeEmbedding or BasisState as the first operation;
  - parameter-shift gradients.

  Analytic mode only. Tested on PennyLane 0.42 to 0.45 and Python 3.10 to 3.13.
- `bench/`, the benchmark harness: 8 circuit configurations with batches of 100 to 100,000,
  against default.qubit, lightning.qubit (with and without qjit), qulacs and a hand-written
  batched JAX simulator. Results for one machine are in `bench/RESULTS.md`.
- Continuous integration: rustfmt, clippy, Rust 1.80, and tests on Linux, Windows and macOS.
- A release workflow that builds wheels for Linux, macOS and Windows and publishes them to PyPI.
- Project files: this changelog, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff`, and
  issue and pull request templates.
- `AI_USAGE.md`, a record of how AI tools were used, and a first draft of `DESIGN.md`, the
  design decisions with who made each one.
