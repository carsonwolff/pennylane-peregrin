"""Fast correctness checks for circuits.py (run: python -m pytest tests, or python tests/test_circuits.py).

1. owner-mri matches a verbatim copy of the per-patch QNodes in parallel_quantum_processing.py.
2. every circuit's template, run by the JAX interpreter, matches its PennyLane body on default.qubit.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pennylane as qml  # noqa: E402

import circuits  # noqa: E402
from backends import build_default_qubit, build_jax_hand  # noqa: E402
from workload import Workload  # noqa: E402


def _original_2q(dev, patch_flat, patch_normalized):
    @qml.qnode(dev, interface="numpy")
    def circuit():
        qml.AmplitudeEmbedding(patch_normalized, wires=range(2), normalize=True)
        for i in range(2):
            qml.Hadamard(wires=i)
        qml.RY(np.pi * np.mean(patch_flat), wires=0)
        qml.RZ(np.pi * np.std(patch_flat), wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(np.pi * np.var(patch_flat), wires=1)
        qml.RX(np.pi * np.median(patch_flat), wires=0)
        return [qml.expval(qml.PauliZ(i)) for i in range(2)] + [qml.expval(qml.PauliX(i)) for i in range(2)]

    return np.array(circuit())


def _original_4q(dev, patch_flat, patch_normalized):
    @qml.qnode(dev, interface="numpy")
    def circuit():
        qml.AmplitudeEmbedding(patch_normalized, wires=range(4), normalize=True)
        for layer in range(2):
            for i in range(4):
                qml.Hadamard(wires=i)
            qml.RY(np.pi * np.mean(patch_flat), wires=0)
            qml.RZ(np.pi * np.std(patch_flat), wires=1)
            qml.RX(np.pi * np.median(patch_flat), wires=2)
            qml.RY(np.pi * np.max(patch_flat), wires=3)
            for i in range(3):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[3, 0])
        measurements = []
        for i in range(4):
            measurements.extend([qml.expval(qml.PauliZ(i)), qml.expval(qml.PauliX(i))])
        return measurements

    return np.array(circuit())


def test_owner_mri_matches_original_code():
    for n, orig in ((2, _original_2q), (4, _original_4q)):
        wl = Workload(n, 8, circuits.resolve_layers("owner-mri", n, None), 0, "stateprep", "owner-mri")
        X, W = circuits.inputs(wl)
        raw = circuits._pixels(wl, wl.dim)  # the patches the inputs were derived from
        run, _ = build_default_qubit(wl, "f64")
        got = run(X, W)
        dev = qml.device("default.qubit", wires=n)
        want = np.stack([orig(dev, p, p / np.linalg.norm(p)) for p in raw])
        err = np.max(np.abs(got - want))
        assert err < 1e-12, (n, err)


def test_templates_match_pennylane():
    for name, c in circuits.CIRCUITS.items():
        for n in c.qubits:
            wl = Workload(n, 16, circuits.resolve_layers(name, n, None), 0, "stateprep", name)
            X, W = circuits.inputs(wl)
            ref = build_default_qubit(wl, "f64")[0](X, W)
            got = build_jax_hand(wl, "f64")[0](X, W)
            err = np.max(np.abs(got - ref))
            assert err < 1e-12, (name, n, err)


if __name__ == "__main__":
    test_owner_mri_matches_original_code()
    test_templates_match_pennylane()
    print("ok")
