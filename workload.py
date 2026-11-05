"""MRI-style patch preprocessing workload.

One circuit per image patch:
  * load a 2**n-pixel patch onto n qubits as amplitudes (rows are pre-normalized,
    so every backend gets identical inputs). Two spellings, chosen by `embed`:
      "stateprep" (default): qml.StatePrep, loaded natively by every backend
      "amplitude": qml.AmplitudeEmbedding(normalize=False), as in the paper's code.
        Under qjit and on qulacs this decomposes into a Mottonen gate cascade
        (~520 gates at 8 qubits), which dominates their runtime. Keep it to
        measure that pitfall, not as the headline comparison.
  * n_layers x [RY, RZ on every qubit, then a CNOT entangler]
  * <Z_i> for every qubit, then <X_i> for every qubit  -> 2n outputs

Edit `apply_circuit` (and the matching hand-written JAX version in
backends.py) if your paper's circuit differs. The correctness check will
tell you if the two drift apart.
"""
from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class Workload:
    n_qubits: int
    batch: int
    n_layers: int = 1
    seed: int = 0
    embed: str = "stateprep"
    circuit: str = "generic"  # key into circuits.CIRCUITS

    @property
    def dim(self) -> int:
        return 2 ** self.n_qubits

    @property
    def n_out(self) -> int:
        """Output width of the generic circuit. Other circuits: circuits.n_out(wl)."""
        return 2 * self.n_qubits

    def to_dict(self):
        return asdict(self)


def make_features(wl: Workload, dtype=np.float64) -> np.ndarray:
    """(batch, 2**n) unit-norm, non-negative rows, like intensity patches.

    The first k rows are identical for any batch size >= k (same seed),
    which is what lets the correctness check compare across batch sizes.
    A small floor is added because amplitude embedding cannot encode an
    all-zero patch (brain MRI background), and real pipelines must do the same.
    """
    rng = np.random.default_rng(wl.seed)
    x = rng.random((wl.batch, wl.dim)) + 1e-6
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x.astype(dtype)


def make_weights(wl: Workload, dtype=np.float64) -> np.ndarray:
    """(n_layers, n_qubits, 2) fixed rotation angles shared by every circuit."""
    rng = np.random.default_rng(wl.seed + 10_000)
    return rng.uniform(0.0, 2.0 * np.pi, (wl.n_layers, wl.n_qubits, 2)).astype(dtype)


def entangler_pairs(n_qubits: int):
    """CNOT chain, closed into a ring when n > 2."""
    pairs = [(i, i + 1) for i in range(n_qubits - 1)]
    if n_qubits > 2:
        pairs.append((n_qubits - 1, 0))
    return pairs


def apply_circuit(qml, features, weights, n_qubits: int, n_layers: int, embed: str = "stateprep"):
    """Shared PennyLane circuit body. Returns the measurement list."""
    wires = list(range(n_qubits))
    if embed == "stateprep":
        qml.StatePrep(features, wires=wires)
    elif embed == "amplitude":
        qml.AmplitudeEmbedding(features, wires=wires, normalize=False)
    else:
        raise ValueError(f"unknown embed {embed!r}")
    for layer in range(n_layers):
        for q in wires:
            qml.RY(weights[layer, q, 0], wires=q)
            qml.RZ(weights[layer, q, 1], wires=q)
        for c, t in entangler_pairs(n_qubits):
            qml.CNOT(wires=[c, t])
    return [qml.expval(qml.PauliZ(q)) for q in wires] + [
        qml.expval(qml.PauliX(q)) for q in wires
    ]
