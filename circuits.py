"""Circuit registry: every workload qbench can run.

Each circuit defines four things that must agree with each other:
  * inputs(wl, dtype) -> (X, W). X is per-circuit data, shape (batch, d_x); W is shared
    by every circuit in the batch (any shape). Rows of X are seed-stable: the first k rows
    are identical for any batch >= k.
  * body(qml, x, w, wl): the PennyLane circuit, written the way a user would write it.
    x is one row or a broadcast batch of rows, so index it with x[..., i].
  * template(wl) -> (ops, meas): the same circuit as a flat gate list. This is what the
    hand-written JAX interpreter (and later the Rust core) executes. The correctness
    check against default.qubit catches any drift between body and template.
  * defaults: qubit counts and layer count used when the CLI does not override them.

Template ops are (name, wires, param). param is None, ("x", col, scale) for a
per-circuit value scale * X[:, col], or ("w", flat_index, scale) for a shared value
scale * W.ravel()[flat_index]. The first op of a state-encoded circuit is
("StatePrep", wires, ("x", (start, stop), 1.0)). meas is a list of ("Z", q) / ("X", q).

The "generic" circuit is the original qbench workload and keeps the conventions in
the handoff unchanged (workload.apply_circuit, make_features, make_weights).
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

from workload import Workload, apply_circuit, entangler_pairs, make_features, make_weights

PI = np.pi


def _embed(qml, amps, wires, embed):
    if embed == "stateprep":
        qml.StatePrep(amps, wires=wires)
    elif embed == "amplitude":
        qml.AmplitudeEmbedding(amps, wires=wires, normalize=False)
    else:
        raise ValueError(f"unknown embed {embed!r}")


def _rng(wl, offset=0):
    return np.random.default_rng(wl.seed + offset)


def _pixels(wl, width):
    """(batch, width) pixel intensities in [0, 1), with a small floor so no patch is all zero."""
    return _rng(wl).random((wl.batch, width)) + 1e-6


def _unit_rows(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _sp(wl):
    return ("StatePrep", tuple(range(wl.n_qubits)), ("x", (0, wl.dim), 1.0))


# ================================================================ generic (original qbench)
def generic_inputs(wl, dtype):
    return make_features(wl, dtype), make_weights(wl, dtype)


def generic_body(qml, x, w, wl):
    return apply_circuit(qml, x, w, wl.n_qubits, wl.n_layers, wl.embed)


def generic_template(wl):
    n = wl.n_qubits
    ops = [_sp(wl)]
    for l in range(wl.n_layers):
        for q in range(n):
            ops.append(("RY", (q,), ("w", (l * n + q) * 2 + 0, 1.0)))
            ops.append(("RZ", (q,), ("w", (l * n + q) * 2 + 1, 1.0)))
        ops += [("CNOT", (c, t), None) for c, t in entangler_pairs(n)]
    meas = [("Z", q) for q in range(n)] + [("X", q) for q in range(n)]
    return ops, meas


# ================================================================ owner-mri (the paper)
# Copied from parallel_quantum_processing.py (quantum_patch_processor_2q / _4q), the code
# behind the paper's results. Note it differs from Fig. 2 in the paper: the 4 qubit code
# closes the CNOT chain into a ring with CNOT(3, 0), and the 2 qubit circuit has its own
# gate sequence. The code is what produced the data, so the code is what we benchmark.
#
# Per-circuit angles come from statistics of the raw patch (np.mean, np.std with ddof=0,
# np.var, np.median, np.max), scaled by pi. They are computed here, outside the timed
# region, because in a PennyLane pipeline that numpy work happens before any device sees
# the circuit and is identical for every backend. The original code adds N(0, 1e-8) noise
# before normalizing; that is irrelevant to cost and omitted so inputs are deterministic.
# X layout: [amplitudes (2^n) | pi*mean, pi*std, pi*var, pi*median, pi*max].
MRI_STATS = ("mean", "std", "var", "median", "max")


def mri_inputs(wl, dtype):
    raw = _pixels(wl, wl.dim)
    stats = np.stack([raw.mean(1), raw.std(1), raw.var(1), np.median(raw, 1), raw.max(1)], axis=1)
    X = np.hstack([_unit_rows(raw), PI * stats])
    return X.astype(dtype), np.zeros(1, dtype)  # W unused; size 1 keeps qjit/vmap happy


def mri_body(qml, x, w, wl):
    n, D = wl.n_qubits, wl.dim
    mean, std, var, med, mx = (x[..., D + i] for i in range(5))
    _embed(qml, x[..., :D], list(range(n)), wl.embed)
    if n == 2:
        for i in range(2):
            qml.Hadamard(wires=i)
        qml.RY(mean, wires=0)
        qml.RZ(std, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(var, wires=1)
        qml.RX(med, wires=0)
        return [qml.expval(qml.PauliZ(i)) for i in range(2)] + [qml.expval(qml.PauliX(i)) for i in range(2)]
    for _layer in range(2):
        for i in range(4):
            qml.Hadamard(wires=i)
        qml.RY(mean, wires=0)
        qml.RZ(std, wires=1)
        qml.RX(med, wires=2)
        qml.RY(mx, wires=3)
        for i in range(3):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[3, 0])
    out = []
    for i in range(4):
        out += [qml.expval(qml.PauliZ(i)), qml.expval(qml.PauliX(i))]
    return out


def mri_template(wl):
    n, D = wl.n_qubits, wl.dim
    s = {k: ("x", D + i, 1.0) for i, k in enumerate(MRI_STATS)}
    ops = [_sp(wl)]
    if n == 2:
        ops += [("H", (0,), None), ("H", (1,), None), ("RY", (0,), s["mean"]), ("RZ", (1,), s["std"]),
                ("CNOT", (0, 1), None), ("RY", (1,), s["var"]), ("RX", (0,), s["median"])]
        return ops, [("Z", 0), ("Z", 1), ("X", 0), ("X", 1)]
    for _layer in range(2):
        ops += [("H", (i,), None) for i in range(4)]
        ops += [("RY", (0,), s["mean"]), ("RZ", (1,), s["std"]), ("RX", (2,), s["median"]), ("RY", (3,), s["max"])]
        ops += [("CNOT", (i, i + 1), None) for i in range(3)] + [("CNOT", (3, 0), None)]
    return ops, [m for i in range(4) for m in (("Z", i), ("X", i))]


# ================================================================ pennylane-demo
# The PennyLane quanvolution demo (Mari), which most applied papers copy: RY(pi x) angle
# encoding of n pixels, qml.RandomLayers with fixed weights, <Z> on every qubit.
# RandomLayers draws its gate layout from `seed`; PennyLane's default is 42 and the demo
# does not override it, so we pin 42 explicitly.
RANDOM_LAYERS_SEED = 42


def demo_inputs(wl, dtype):
    W = _rng(wl, 10_000).uniform(0.0, 2.0 * PI, (wl.n_layers, wl.n_qubits))
    return _pixels(wl, wl.n_qubits).astype(dtype), W.astype(dtype)


def demo_body(qml, x, w, wl):
    n = wl.n_qubits
    for j in range(n):
        qml.RY(PI * x[..., j], wires=j)
    qml.RandomLayers(w, wires=list(range(n)), seed=RANDOM_LAYERS_SEED)
    return [qml.expval(qml.PauliZ(j)) for j in range(n)]


def demo_template(wl):
    """Expand RandomLayers to gates; recover which weight each rotation reads by decoding
    marker values (RandomLayers consumes weights in order, but this does not rely on it)."""
    import pennylane as qml

    n, L = wl.n_qubits, wl.n_layers
    marker = (np.arange(L * n, dtype=np.float64) + 0.25).reshape(L, n)
    ops = [("RY", (j,), ("x", j, PI)) for j in range(n)]
    for op in qml.RandomLayers(marker, wires=list(range(n)), seed=RANDOM_LAYERS_SEED).decomposition():
        wires = tuple(int(w) for w in op.wires)
        if op.name == "CNOT":
            ops.append(("CNOT", wires, None))
        elif op.name in ("RX", "RY", "RZ"):
            ops.append((op.name, wires, ("w", int(np.floor(float(op.parameters[0]))), 1.0)))
        else:
            raise NotImplementedError(f"RandomLayers emitted {op.name}")
    return ops, [("Z", j) for j in range(n)]


# ================================================================ angle-trainable (ResQuNNs)
# Kashif and Shafique, arXiv 2402.09146: RY(x_i) encoding (no pi factor), trainable
# RX(theta_i) on every qubit, then an open CNOT chain (j, j+1); <Z> on every qubit.
# n_layers > 1 repeats the RX + chain block.
def angle_inputs(wl, dtype):
    W = _rng(wl, 10_000).uniform(0.0, 2.0 * PI, (wl.n_layers, wl.n_qubits))
    return _pixels(wl, wl.n_qubits).astype(dtype), W.astype(dtype)


def angle_body(qml, x, w, wl):
    n = wl.n_qubits
    for i in range(n):
        qml.RY(x[..., i], wires=i)
    for l in range(wl.n_layers):
        for i in range(n):
            qml.RX(w[l, i], wires=i)
        for j in range(n - 1):
            qml.CNOT(wires=[j, j + 1])
    return [qml.expval(qml.PauliZ(i)) for i in range(n)]


def angle_template(wl):
    n = wl.n_qubits
    ops = [("RY", (i,), ("x", i, 1.0)) for i in range(n)]
    for l in range(wl.n_layers):
        ops += [("RX", (i,), ("w", l * n + i, 1.0)) for i in range(n)]
        ops += [("CNOT", (j, j + 1), None) for j in range(n - 1)]
    return ops, [("Z", i) for i in range(n)]


# ================================================================ amp-sel (spectrum paper)
# Bischof et al., arXiv 2512.13125, ansatz 1: amplitude embedding of an L2-normalized
# window (32 values on 5 qubits), then 3 x (RX, RY, RZ on every qubit, open CNOT chain);
# <Z> on every qubit. Trainable weights, shape (n_layers, n, 3).
def sel_inputs(wl, dtype):
    W = _rng(wl, 10_000).uniform(0.0, 2.0 * PI, (wl.n_layers, wl.n_qubits, 3))
    return _unit_rows(_pixels(wl, wl.dim)).astype(dtype), W.astype(dtype)


def sel_body(qml, x, w, wl):
    n = wl.n_qubits
    _embed(qml, x, list(range(n)), wl.embed)
    for l in range(wl.n_layers):
        for q in range(n):
            qml.RX(w[l, q, 0], wires=q)
            qml.RY(w[l, q, 1], wires=q)
            qml.RZ(w[l, q, 2], wires=q)
        for j in range(n - 1):
            qml.CNOT(wires=[j, j + 1])
    return [qml.expval(qml.PauliZ(q)) for q in range(n)]


def sel_template(wl):
    n = wl.n_qubits
    ops = [_sp(wl)]
    for l in range(wl.n_layers):
        for q in range(n):
            for k, g in enumerate(("RX", "RY", "RZ")):
                ops.append((g, (q,), ("w", (l * n + q) * 3 + k, 1.0)))
        ops += [("CNOT", (j, j + 1), None) for j in range(n - 1)]
    return ops, [("Z", q) for q in range(n)]


# ================================================================ registry
@dataclass(frozen=True)
class Circuit:
    inputs: Callable
    body: Callable
    template: Callable
    qubits: tuple  # default sweep
    layers: int  # default n_layers
    allowed_qubits: tuple = ()  # empty = any
    fixed_layers: dict = None  # n_qubits -> n_layers the circuit always has (ignores --layers)
    state_encoded: bool = True  # False: --embed does not apply
    source: str = ""


CIRCUITS = {
    "generic": Circuit(generic_inputs, generic_body, generic_template, (2, 4, 6, 8), 1,
                       source="original qbench workload; see handoff conventions"),
    "owner-mri": Circuit(mri_inputs, mri_body, mri_template, (2, 4), 1, allowed_qubits=(2, 4),
                         fixed_layers={2: 1, 4: 2},
                         source="Wolff et al., IISA 2026; gates from parallel_quantum_processing.py"),
    "pennylane-demo": Circuit(demo_inputs, demo_body, demo_template, (4,), 1, state_encoded=False,
                              source="PennyLane quanvolution demo; also Tanbhir et al. arXiv 2510.23660"),
    "angle-trainable": Circuit(angle_inputs, angle_body, angle_template, (4,), 1, state_encoded=False,
                               source="ResQuNNs, Kashif and Shafique, arXiv 2402.09146"),
    "amp-sel": Circuit(sel_inputs, sel_body, sel_template, (5,), 3,
                       source="Bischof et al. arXiv 2512.13125, ansatz 1"),
}


def get(wl: Workload) -> Circuit:
    return CIRCUITS[wl.circuit]


def resolve_layers(name, n_qubits, requested):
    """n_layers for one case: the circuit's fixed value, else the CLI value, else the default."""
    c = CIRCUITS[name]
    if c.allowed_qubits and n_qubits not in c.allowed_qubits:
        raise ValueError(f"circuit {name} is only defined for n_qubits in {c.allowed_qubits}")
    if c.fixed_layers:
        fixed = c.fixed_layers[n_qubits]
        if requested not in (None, fixed):
            raise ValueError(f"circuit {name} has a fixed {fixed} layer(s) at n={n_qubits}")
        return fixed
    return c.layers if requested is None else requested


def inputs(wl, dtype=np.float64):
    return get(wl).inputs(wl, dtype)


def n_out(wl):
    return len(get(wl).template(wl)[1])
