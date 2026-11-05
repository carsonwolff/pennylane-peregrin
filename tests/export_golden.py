"""Export golden outputs from default.qubit for the Rust core's tests.

Writes crates/core/tests/golden/{gates,qbench,random}.json. Each case is a circuit spec in
the JSON format below, a batch of inputs and weights, and default.qubit's outputs (and final
states). The Rust tests rebuild the circuit from the spec and must match to 1e-12 in f64.

Circuit spec:
  {"n_qubits": n,
   "state_prep": null | {"kind": "real" | "complex", "start": col},
   "ops": [{"name": "RY", "wires": [0], "params": [["x", col, scale] | ["w", idx, scale] | ["c", value]],
            "word": "XZ" (PauliRot only), "matrix": [[re, im], ...] (QubitUnitary only)}],
   "measurements": [{"type": "expval" | "var", "word": [[wire, "X"], ...]} | {"type": "probs", "wires": [...]}]}

Run: .venv/bin/python tests/export_golden.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pennylane as qml  # noqa: E402

import circuits  # noqa: E402
from workload import Workload  # noqa: E402

OUT = ROOT / "crates" / "core" / "tests" / "golden"

FIXED_1Q = ["PauliX", "PauliY", "PauliZ", "Hadamard", "S", "Sdg", "T", "Tdg", "SX"]
PARAM_1Q = {"RX": 1, "RY": 1, "RZ": 1, "PhaseShift": 1, "Rot": 3, "U3": 3}
FIXED_2Q = ["CNOT", "CY", "CZ", "SWAP"]
PARAM_2Q = ["CRX", "CRY", "CRZ", "ControlledPhaseShift", "IsingXX", "IsingYY", "IsingZZ"]
PAULI = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}


# ---------------------------------------------------------------- spec -> PennyLane
def _value(p, X, W):
    kind = p[0]
    if kind == "c":
        return p[1]
    if kind == "x":
        return p[2] * X[:, p[1]]
    return p[2] * W[p[1]]


def _apply(op, X, W):
    name, wires = op["name"], op["wires"]
    ps = [_value(p, X, W) for p in op.get("params", [])]
    if name == "Sdg":
        qml.adjoint(qml.S(wires=wires))
    elif name == "Tdg":
        qml.adjoint(qml.T(wires=wires))
    elif name == "PauliRot":
        qml.PauliRot(ps[0], op["word"], wires=wires)
    elif name == "QubitUnitary":
        m = np.array([complex(re, im) for re, im in op["matrix"]]).reshape(2 ** len(wires), -1)
        qml.QubitUnitary(m, wires=wires)
    else:
        getattr(qml, name)(*ps, wires=wires)


def _measure(m):
    if m["type"] == "probs":
        return qml.probs(wires=m["wires"])
    terms = [PAULI[p](w) for w, p in m["word"]]
    obs = terms[0] if len(terms) == 1 else qml.prod(*terms)
    return qml.expval(obs) if m["type"] == "expval" else qml.var(obs)


def reference(spec, X, W):
    """default.qubit outputs (B, n_out) and final states (B, 2^n) for a spec."""
    n, B = spec["n_qubits"], X.shape[0]
    dev = qml.device("default.qubit", wires=n)

    def prep():
        sp = spec["state_prep"]
        if sp is None:
            return
        D, s = 2**n, sp["start"]
        if sp["kind"] == "real":
            amps = X[:, s:s + D]
        else:
            amps = X[:, s:s + 2 * D:2] + 1j * X[:, s + 1:s + 2 * D:2]
        qml.StatePrep(amps, wires=range(n))

    @qml.qnode(dev, interface=None, diff_method=None)
    def outputs():
        prep()
        for op in spec["ops"]:
            _apply(op, X, W)
        return [_measure(m) for m in spec["measurements"]]

    @qml.qnode(dev, interface=None, diff_method=None)
    def state():
        prep()
        for op in spec["ops"]:
            _apply(op, X, W)
        return qml.state()

    cols = []
    for r, m in zip(outputs(), spec["measurements"]):
        width = 2 ** len(m["wires"]) if m["type"] == "probs" else 1
        cols.append(np.broadcast_to(np.reshape(r, (-1, width)), (B, width)))
    psi = np.broadcast_to(np.reshape(state(), (-1, 2**n)), (B, 2**n))
    return np.concatenate(cols, axis=1), psi


def case(name, spec, X, W):
    W = np.asarray(W, float).ravel()  # template weight indices are flat
    out, psi = reference(spec, X, W)
    return {"name": name, "circuit": spec, "batch": int(X.shape[0]), "inputs": X.ravel().tolist(),
            "weights": np.asarray(W, float).ravel().tolist(), "expected": out.ravel().tolist(),
            "states": [[float(z.real), float(z.imag)] for z in psi.ravel()]}


# ---------------------------------------------------------------- random inputs
def random_state_inputs(rng, B, n, kind):
    D = 2**n
    if kind == "real":
        x = rng.normal(size=(B, D))
        return x / np.linalg.norm(x, axis=1, keepdims=True)
    z = rng.normal(size=(B, D)) + 1j * rng.normal(size=(B, D))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    out = np.empty((B, 2 * D))
    out[:, 0::2], out[:, 1::2] = z.real, z.imag
    return out


def random_unitary(rng, k):
    z = rng.normal(size=(2**k, 2**k)) + 1j * rng.normal(size=(2**k, 2**k))
    q, r = np.linalg.qr(z)
    q = q * (np.diag(r) / np.abs(np.diag(r)))
    return [[float(v.real), float(v.imag)] for v in q.ravel()]


def random_param(rng, n_in_cols, in_start, n_w):
    r = rng.random()
    if r < 0.4:
        return ["x", int(in_start + rng.integers(n_in_cols)), float(rng.choice([1.0, np.pi, -0.5]))]
    if r < 0.8:
        return ["w", int(rng.integers(n_w)), float(rng.choice([1.0, 2.0]))]
    return ["c", float(rng.uniform(-np.pi, np.pi))]


def random_spec(rng, n, n_ops, prep_kind, n_in_cols=4, n_w=6):
    D = 2**n
    prep_width = {"real": D, "complex": 2 * D, None: 0}[prep_kind]
    in_start = prep_width  # angle columns come after the state block
    ops = []
    for _ in range(n_ops):
        choices = ["f1", "p1", "u1"] + (["f2", "p2", "u2", "mrz", "prot"] if n >= 2 else ["mrz", "prot"])
        kind = rng.choice(choices)
        wires = [int(w) for w in rng.permutation(n)]
        if kind == "f1":
            ops.append({"name": str(rng.choice(FIXED_1Q)), "wires": wires[:1]})
        elif kind == "p1":
            g = str(rng.choice(list(PARAM_1Q)))
            ops.append({"name": g, "wires": wires[:1],
                        "params": [random_param(rng, n_in_cols, in_start, n_w) for _ in range(PARAM_1Q[g])]})
        elif kind == "u1":
            ops.append({"name": "QubitUnitary", "wires": wires[:1], "matrix": random_unitary(rng, 1)})
        elif kind == "f2":
            ops.append({"name": str(rng.choice(FIXED_2Q)), "wires": wires[:2]})
        elif kind == "p2":
            ops.append({"name": str(rng.choice(PARAM_2Q)), "wires": wires[:2],
                        "params": [random_param(rng, n_in_cols, in_start, n_w)]})
        elif kind == "u2":
            ops.append({"name": "QubitUnitary", "wires": wires[:2], "matrix": random_unitary(rng, 2)})
        elif kind == "mrz":
            k = int(rng.integers(1, n + 1))
            ops.append({"name": "MultiRZ", "wires": wires[:k], "params": [random_param(rng, n_in_cols, in_start, n_w)]})
        else:
            k = int(rng.integers(1, n + 1))
            word = "".join(rng.choice(list("IXYZ"), size=k))
            ops.append({"name": "PauliRot", "wires": wires[:k], "word": word,
                        "params": [random_param(rng, n_in_cols, in_start, n_w)]})
    meas = []
    for _ in range(int(rng.integers(1, 5))):
        t = rng.choice(["expval", "expval", "var", "probs"])
        wires = [int(w) for w in rng.permutation(n)][: int(rng.integers(1, min(n, 3) + 1))]
        if t == "probs":
            meas.append({"type": "probs", "wires": wires})
        else:
            meas.append({"type": str(t), "word": [[w, str(rng.choice(list("XYZ")))] for w in wires]})
    sp = None if prep_kind is None else {"kind": prep_kind, "start": 0}
    return {"n_qubits": n, "state_prep": sp, "ops": ops, "measurements": meas}, in_start + n_in_cols, n_w


def random_inputs(rng, B, spec, d_x, n_w):
    n, sp = spec["n_qubits"], spec["state_prep"]
    parts = []
    if sp is not None:
        parts.append(random_state_inputs(rng, B, n, sp["kind"]))
    d_angles = d_x - sum(p.shape[1] for p in parts)
    parts.append(rng.uniform(-np.pi, np.pi, (B, d_angles)))
    return np.hstack(parts), rng.uniform(-np.pi, np.pi, n_w)


# ---------------------------------------------------------------- qbench templates -> spec
def template_spec(wl):
    ops, meas = circuits.get(wl).template(wl)
    spec = {"n_qubits": wl.n_qubits, "state_prep": None, "ops": [], "measurements": []}
    for name, wires, param in ops:
        if name == "StatePrep":
            spec["state_prep"] = {"kind": "real", "start": param[1][0]}
            continue
        names = {"H": "Hadamard"}
        op = {"name": names.get(name, name), "wires": list(wires)}
        if param is not None:
            op["params"] = [[param[0], int(param[1]), float(param[2])]]
        spec["ops"].append(op)
    spec["measurements"] = [{"type": "expval", "word": [[q, k]]} for k, q in meas]
    return spec


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234)

    # 1. every gate alone, on random wires of a 3 qubit random complex state
    gates = []
    n = 3
    all_gates = ([(g, 1, 0) for g in FIXED_1Q] + [(g, 1, k) for g, k in PARAM_1Q.items()]
                 + [(g, 2, 0) for g in FIXED_2Q] + [(g, 2, 1) for g in PARAM_2Q]
                 + [("MultiRZ", 3, 1), ("PauliRot", 3, 1), ("QubitUnitary", 1, 0), ("QubitUnitary", 2, 0)])
    for g, k, n_par in all_gates:
        wires = [int(w) for w in rng.permutation(n)[:k]]
        op = {"name": g, "wires": wires}
        if n_par:
            op["params"] = [["w", i, 1.0] for i in range(n_par)]
        if g == "PauliRot":
            op["word"] = "XYZ"
        if g == "QubitUnitary":
            op["matrix"] = random_unitary(rng, k)
        spec = {"n_qubits": n, "state_prep": {"kind": "complex", "start": 0}, "ops": [op],
                "measurements": [{"type": "probs", "wires": [0, 1, 2]}]}
        X = random_state_inputs(rng, 4, n, "complex")
        W = rng.uniform(-np.pi, np.pi, 3)
        gates.append(case(f"{g}{wires}", spec, X, W))

    # 2. the qbench circuits, 64 rows each, from circuits.py
    bench = []
    for name, c in circuits.CIRCUITS.items():
        for nq in sorted(set(c.qubits) | ({6, 8} if name == "generic" else set())):
            wl = Workload(nq, 64, circuits.resolve_layers(name, nq, None), 0, "stateprep", name)
            X, W = circuits.inputs(wl)
            bench.append(case(f"{name}-n{nq}", template_spec(wl), X, W))
    wl = Workload(4, 64, 3, 0, "stateprep", "generic")  # multi-layer generic
    X, W = circuits.inputs(wl)
    bench.append(case("generic-n4-L3", template_spec(wl), X, W))

    # 3. random circuits over the whole gate set
    rand = []
    for i in range(60):
        nq = int(rng.integers(1, 7))
        prep = [None, "real", "complex"][i % 3]
        spec, d_x, n_w = random_spec(rng, nq, int(rng.integers(1, 25)), prep)
        X, W = random_inputs(rng, 8, spec, d_x, n_w)
        rand.append(case(f"random-{i}-n{nq}", spec, X, W))

    for fname, cases in (("gates", gates), ("qbench", bench), ("random", rand)):
        (OUT / f"{fname}.json").write_text(json.dumps({"pennylane": qml.__version__, "cases": cases}))
        print(f"{fname}.json: {len(cases)} cases")


if __name__ == "__main__":
    main()
