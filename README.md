# peregrin

> **Status: archived (November 2026).** peregrin works as described below, but it is no longer developed or maintained. Issues and pull requests are closed. You're welcome to fork it under either license.

[![CI](https://github.com/carsonwolff/pennylane-peregrin/actions/workflows/ci.yml/badge.svg)](https://github.com/carsonwolff/pennylane-peregrin/actions/workflows/ci.yml)

A fast simulator for large batches of small quantum circuits (1 to 10 qubits), built for
quanvolutional preprocessing: one circuit per image patch, hundreds of thousands of patches per
call. The core is written in Rust and runs one batch-first call: a circuit template, a
`(batch, d_x)` array of per-circuit inputs, and shared weights, giving `(batch, n_outputs)`.

On a 2-core laptop CPU it beats every PennyLane CPU backend we measured, including a
hand-written batched JAX simulator, at every qubit count and batch size we tested (68 of 68
cells). The smallest margin is 1.34x, and at B=100k it is 2.1x to 15.7x. Outputs match
default.qubit to 1e-12 in f64. Used through PennyLane as `qml.device("peregrin.cpu")`, it is
faster than every other PennyLane device in all 68 cells (5.3x to 73x at B=100k). Full numbers
and methodology: [bench/RESULTS.md](bench/RESULTS.md).

Status: archived. The CPU core, the Python bindings and the PennyLane device work as described
here. No GPU backend is planned.

## Layout

| path | what it is |
|---|---|
| `crates/core` | `peregrin-core`, the Rust simulator |
| `crates/python` | `pennylane-peregrin`, the Python package (PyO3 + maturin), imported as `peregrin` |
| `bench/` | qbench, the benchmark harness that compares peregrin with other PennyLane backends; see [bench/README.md](bench/README.md) |

## Install

peregrin was never published to PyPI, so it installs from source. You need Python 3.10+,
PennyLane 0.42+ and a Rust toolchain (from [rustup.rs](https://rustup.rs)):

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin
cd crates/python && maturin develop --release
```

The fast SIMD path needs AVX2, which is detected at runtime. Other CPUs, including Apple
Silicon, run a portable path that the compiler vectorizes on its own. Its speed was never
benchmarked.

## Use from PennyLane

```python
import numpy as np
import pennylane as qml

dev = qml.device("peregrin.cpu", wires=4)          # precision="f32" and threads=N are optional

@qml.qnode(dev)
def quanv(patches, weights):                   # patches: (batch, 4), one row per image patch
    for i in range(4):
        qml.RY(np.pi * patches[:, i], wires=i)
    qml.RandomLayers(weights, wires=range(4))
    return [qml.expval(qml.Z(i)) for i in range(4)]

out = quanv(np.random.rand(100_000, 4), np.random.uniform(0, 2 * np.pi, (1, 4)))
```

Write the circuit for the whole batch with parameter broadcasting (a leading batch dimension on
the angles, or a batch of states in `qml.StatePrep` / `qml.AmplitudeEmbedding`). The whole
batch then runs as one call. A Python loop over patches still works, but each patch becomes
its own call and PennyLane's per-call overhead dominates.

Supported:
- **Gates:** any gate PennyLane can decompose into the gate set below.
- **State preparation:** StatePrep, AmplitudeEmbedding or BasisState as the first operation.
- **Measurements:** `expval` and `var` of Pauli words, Hamiltonians, and sums or products of
  Paulis; matrix observables such as `Hermitian` on up to 4 wires; `probs`; `state`.
- **Gradients:** parameter-shift, with any interface. PennyLane selects it automatically.

Not supported yet: shots (sampling), mid-circuit measurements, and device-native gradients.

## Use from Python

```python
import numpy as np
import peregrin

# RY(x0) on wire 0, then CNOT(0, 1), then <Z1>; wire 0 is the most significant bit, as in PennyLane.
c = peregrin.Circuit({
    "n_qubits": 2,
    "ops": [
        {"name": "RY", "wires": [0], "params": [["x", 0, 1.0]]},
        {"name": "CNOT", "wires": [0, 1]},
    ],
    "measurements": [{"type": "expval", "word": [[1, "Z"]]}],
})
X = np.random.uniform(0, np.pi, (100_000, 1))
out = c.execute(X)              # shape (100000, 1), float64; float32 inputs give float32
peregrin.set_num_threads(4)     # optional; default is one thread per logical CPU
```

A param is `["x", col, scale]` (column `col` of each circuit's inputs, times `scale`),
`["w", idx, scale]` (shared weight `idx`, passed as `execute(X, weights)`) or `["c", value]`.
`"state_prep": {"kind": "real", "start": 0}` loads each circuit's state from its input row,
starting at column `start` (2^n real amplitudes, already normalized).

Measurements are `expval` and `var` of Pauli words (`[[wire, "X" | "Y" | "Z"], ...]`) and `probs` on
a list of wires. Gates: PauliX/Y/Z, Hadamard, S, Sdg, T, Tdg, SX, RX, RY, RZ, PhaseShift, Rot, U3,
CNOT, CY, CZ, SWAP, CRX, CRY, CRZ, ControlledPhaseShift, IsingXX/YY/ZZ, MultiRZ, PauliRot,
QubitUnitary (1 or 2 wires). The full spec is in
[crates/core/src/json.rs](crates/core/src/json.rs).

## Use from Rust

```rust
use peregrin_core::{Circuit, Gate, Param, Pauli};

let mut c = Circuit::new(2);
c.gate(Gate::RY(Param::input(0)), &[0]).gate(Gate::CNOT, &[0, 1]);
c.expval(&[(1, Pauli::Z)]);
let out = peregrin_core::execute::<f64>(&c, batch, &inputs, &[])?;
```

`execute` also comes in f32. `final_states` returns state vectors, and `sample` draws shots
with a seed per circuit, so results don't depend on the thread count.

## How it is fast

- **Tiles of 8 circuits.** The core stores 8 circuits' states side by side (struct-of-arrays),
  so each gate runs as one SIMD loop across circuits. AVX2 is picked at runtime.
- **Free permutations.** CNOT and SWAP chains move no data. The core tracks which row holds
  which amplitude instead.
- **Real-state tracking.** While the state is known to be real, the imaginary half of the work
  is skipped.
- **Gate fusion.** Gates on the same wire are fused, and matrices that use only shared weights
  are computed once per call.
- **Vectorized sin/cos.** Per-circuit angles use a vectorized sin/cos.

## Tests

```bash
cargo test --release                              # golden outputs from default.qubit, unit tests
pytest crates/python/tests                        # the PennyLane device against default.qubit
cd bench && ../.venv/bin/python tests/test_circuits.py
```

## Support

- **Maintenance:** none. The project is archived (see the status note at the top), so issues
  and pull requests are closed. To fix or extend it, fork it under either license.
- **Tested versions:** at the time of archiving, CI tested Python 3.10 to 3.13 and PennyLane
  0.42 to 0.45, on Linux, macOS and Windows. Newer releases of either may break it.
- **Contributing:** see [CONTRIBUTING.md](CONTRIBUTING.md) and the
  [code of conduct](CODE_OF_CONDUCT.md).
- **Citing:** see [CITATION.cff](CITATION.cff), or use "Cite this repository" on GitHub.

## CI and releases

CI (`.github/workflows/ci.yml`) runs rustfmt, clippy, the Rust tests on Linux, Windows and macOS,
and the device tests on PennyLane 0.42 through the latest. A release workflow
(`.github/workflows/release.yml`) can build wheels for Linux, macOS and Windows and publish
them to PyPI on a `v*` tag, but it was never run, and no release was made.

## License

MIT OR Apache-2.0, at your option.
