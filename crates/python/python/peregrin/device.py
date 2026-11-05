"""PennyLane device: ``qml.device("peregrin", wires=n)``.

A broadcast tape (parameters with a leading batch dimension, or a batched StatePrep or
AmplitudeEmbedding) runs as one peregrin call over the whole batch. That is the case this
device is built for: quanvolution, where one circuit runs once per image patch.

    dev = qml.device("peregrin", wires=4)

    @qml.qnode(dev)
    def circuit(patches, weights):            # patches: (batch, 4)
        for i in range(4):
            qml.RY(np.pi * patches[:, i], wires=i)
        qml.RandomLayers(weights, wires=range(4))
        return [qml.expval(qml.Z(i)) for i in range(4)]

Supported: every gate PennyLane can decompose to peregrin's gate set; a StatePrep or
AmplitudeEmbedding as the first operation; expval and var of any observable with a Pauli
representation (Pauli words, Hamiltonians, sums and products of Paulis) or a matrix on up to 4
wires (Hermitian, Projector, ...); probs on wires; state.
Analytic mode only. Shots, mid-circuit measurements and device gradients are not supported yet.
Gradients use parameter-shift, which PennyLane picks automatically.

Options: ``precision="f64"`` (default) or ``"f32"``, and ``threads`` (default: one per logical
CPU).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pennylane as qml
from pennylane.devices import Device, ExecutionConfig
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.devices.preprocess import (
    decompose,
    no_sampling,
    validate_device_wires,
    validate_measurements,
)
from pennylane.exceptions import DeviceError
from pennylane.measurements import ExpectationMP, ProbabilityMP, StateMP, VarianceMP
from pennylane.operation import StatePrepBase
from pennylane.transforms.core import CompilePipeline

from . import Circuit, set_num_threads

MAX_QUBITS = 16

# PennyLane name -> peregrin name.
GATES = {
    "PauliX": "PauliX", "PauliY": "PauliY", "PauliZ": "PauliZ", "Hadamard": "Hadamard",
    "S": "S", "T": "T", "SX": "SX", "Adjoint(S)": "Sdg", "Adjoint(T)": "Tdg",
    "RX": "RX", "RY": "RY", "RZ": "RZ", "PhaseShift": "PhaseShift", "Rot": "Rot", "U3": "U3",
    "CNOT": "CNOT", "CY": "CY", "CZ": "CZ", "SWAP": "SWAP",
    "CRX": "CRX", "CRY": "CRY", "CRZ": "CRZ", "ControlledPhaseShift": "ControlledPhaseShift",
    "IsingXX": "IsingXX", "IsingYY": "IsingYY", "IsingZZ": "IsingZZ",
    "MultiRZ": "MultiRZ", "PauliRot": "PauliRot",
}
# Handled outside the gate list: identity is dropped, global phase only touches state().
PASS_THROUGH = {"Identity", "GlobalPhase"}


def _stopping_condition(op) -> bool:
    if op.name in GATES or op.name in PASS_THROUGH:
        return True
    if op.name == "QubitUnitary":
        return len(op.wires) <= 2 and op.batch_size is None
    return False


# Observables without a Pauli representation (Hermitian, ...) are converted from their matrix,
# which costs 4^k terms on k wires, so only up to this many wires.
MAX_MATRIX_OBS_WIRES = 4


def _pauli_sentence(obs):
    if obs.pauli_rep is not None:
        return obs.pauli_rep
    return qml.pauli_decompose(qml.matrix(obs), wire_order=obs.wires, pauli=True)


def _accepted_measurement(m) -> bool:
    if isinstance(m, (ExpectationMP, VarianceMP)):
        if m.obs is None:
            return False
        return m.obs.pauli_rep is not None or (m.obs.has_matrix and len(m.obs.wires) <= MAX_MATRIX_OBS_WIRES)
    if isinstance(m, ProbabilityMP):
        return m.obs is None
    return type(m) is StateMP


def _np(x):
    return np.asarray(qml.math.unwrap(x))


class _Lowered:
    """A tape as a peregrin circuit spec plus its numeric inputs, and how to read results."""

    def __init__(self, tape, wire_order, dtype=np.float64):
        self.n = len(wire_order)
        if self.n > MAX_QUBITS:
            raise DeviceError(f"peregrin supports up to {MAX_QUBITS} wires, got {self.n}")
        idx = {w: i for i, w in enumerate(wire_order)}
        self.batch = tape.batch_size
        B = self.batch or 1
        cols, weights = [], []  # per-circuit columns (arrays broadcastable to (B,)), shared scalars
        col_of = {}  # the same array view used by several gates gets one column
        key, ops = [], []
        prep = None
        self.phase = np.zeros(B)

        def param(p, ndim):
            a = _np(p)
            if a.ndim > ndim:  # broadcast: one value per circuit
                a = a.reshape(-1)
                view = (a.__array_interface__["data"][0], a.strides, a.shape, a.dtype.str)
                if view not in col_of:
                    cols.append(a)
                    col_of[view] = len(cols)
                return ["x", -col_of[view], 1.0], "x"  # column fixed up after state prep is known
            weights.append(float(a))
            return ["w", len(weights) - 1, 1.0], "w"

        for k, op in enumerate(tape.operations):
            if isinstance(op, StatePrepBase):
                if k != 0:
                    raise DeviceError("peregrin: state preparation must be the first operation")
                data = _np(op.data[0]) if op.data else None
                if (type(op) in (qml.StatePrep, qml.AmplitudeEmbedding) and list(op.wires) == list(wire_order)
                        and data is not None and data.shape[-1] == 2**self.n):
                    prep = data.reshape(B, 2**self.n)  # already the state, in device order
                else:
                    prep = _np(op.state_vector(wire_order=wire_order)).reshape(B, 2**self.n)
                continue
            if op.name == "Identity":
                continue
            if op.name == "GlobalPhase":
                self.phase = self.phase + np.broadcast_to(_np(op.data[0]), (B,))
                continue
            wires = [idx[w] for w in op.wires]
            if op.name == "QubitUnitary":
                m = _np(op.matrix())
                ops.append({"name": "QubitUnitary", "wires": wires,
                            "matrix": [[float(z.real), float(z.imag)] for z in m.reshape(-1)]})
                key.append(("QubitUnitary", tuple(wires), m.tobytes()))
                continue
            name = GATES[op.name]
            spec = {"name": name, "wires": wires}
            kinds = []
            if op.data:
                spec["params"] = []
                for p, nd in zip(op.data, op.ndim_params):
                    pr, kind = param(p, nd)
                    spec["params"].append(pr)
                    kinds.append(kind)
            if name == "PauliRot":
                spec["word"] = op.hyperparameters["pauli_word"]
            key.append((name, tuple(wires), tuple(kinds), spec.get("word")))
            ops.append(spec)

        # One input row per circuit: state prep columns first, then gate columns, written
        # straight into a single array of the execution dtype.
        D = 2**self.n
        if prep is None:
            state_prep, off = None, 0
        elif np.iscomplexobj(prep) and np.any(prep.imag != 0):
            state_prep, off = {"kind": "complex", "start": 0}, 2 * D
        else:
            state_prep, off = {"kind": "real", "start": 0}, D
        self.X = np.empty((B, off + len(cols)), dtype=dtype)
        if state_prep is not None and state_prep["kind"] == "complex":
            self.X[:, 0:off:2], self.X[:, 1:off:2] = prep.real, prep.imag
        elif state_prep is not None:
            self.X[:, :off] = prep.real if np.iscomplexobj(prep) else prep
        for j, c in enumerate(cols):
            self.X[:, off + j] = c
        for spec in ops:
            for pr in spec.get("params", []):
                if pr[0] == "x":
                    pr[1] = off + (-pr[1] - 1)
        self.W = np.asarray(weights, dtype=dtype)

        # Measurements: every distinct Pauli word is one peregrin output; probs follow.
        words, meas, self.plans = {}, [], []

        def terms(ps):
            out, const = [], 0.0
            for pw, c in ps.items():
                c = float(np.real(c))
                w = tuple(sorted((idx[q], p) for q, p in pw.items()))
                if not w:
                    const += c
                    continue
                if w not in words:
                    words[w] = len(words)
                out.append((words[w], c))
            return out, const

        probs = []
        for m in tape.measurements:
            if isinstance(m, ExpectationMP):
                self.plans.append(("expval", terms(_pauli_sentence(m.obs))))
            elif isinstance(m, VarianceMP):
                ps = _pauli_sentence(m.obs)
                self.plans.append(("var", terms(ps), terms(ps @ ps)))
            elif isinstance(m, ProbabilityMP):
                pw = [idx[w] for w in (m.wires or wire_order)]
                probs.append(pw)
                self.plans.append(("probs", len(probs) - 1, 2 ** len(pw)))
            else:
                self.plans.append(("state",))
        meas = [{"type": "expval", "word": [[q, p] for q, p in w]} for w in words]
        meas += [{"type": "probs", "wires": pw} for pw in probs]
        self.n_words = len(words)
        self.needs_state = any(p[0] == "state" for p in self.plans)
        self.spec = {"n_qubits": self.n, "state_prep": state_prep, "ops": ops, "measurements": meas}
        self.key = (self.n, None if state_prep is None else state_prep["kind"], tuple(key),
                    tuple(tuple(map(tuple, m["word"])) if "word" in m else ("p",) + tuple(m["wires"]) for m in meas))

    def results(self, out, psi):
        """Assemble PennyLane results from peregrin's (B, n_out) outputs and optional states."""
        res = []
        probs_at = self.n_words
        prob_offsets, o = [], probs_at
        for p in self.plans:
            if p[0] == "probs":
                prob_offsets.append(o)
                o += p[2]

        def lin(t):
            ts, const = t
            if const == 0 and len(ts) == 1 and ts[0][1] == 1.0:
                return out[:, ts[0][0]]  # a single Pauli word: no arithmetic, no copy
            v = np.full(out.shape[0], const, dtype=out.dtype)
            for i, c in ts:
                v = v + c * out[:, i]
            return v

        for p in self.plans:
            if p[0] == "expval":
                r = lin(p[1])
            elif p[0] == "var":
                e = lin(p[1])
                r = lin(p[2]) - e * e
            elif p[0] == "probs":
                a = prob_offsets[p[1]]
                r = out[:, a:a + p[2]]
            else:
                r = psi * np.exp(-1j * self.phase)[:, None]
            res.append(r if self.batch is not None else r[0])
        return res[0] if len(res) == 1 else tuple(res)


@simulator_tracking
@single_tape_support
class PeregrinDevice(Device):
    """Batched state-vector simulator for many small circuits (see module docs)."""

    name = "peregrin"
    _CACHE_SIZE = 64

    def __init__(self, wires=None, shots=None, precision="f64", threads=None):
        super().__init__(wires=wires, shots=shots)
        if precision not in ("f64", "f32"):
            raise ValueError("precision must be 'f64' or 'f32'")
        self.precision = precision
        self._dtype = np.float64 if precision == "f64" else np.float32
        if threads is not None:
            set_num_threads(int(threads))
        self._cache = {}

    def setup_execution_config(self, config: ExecutionConfig | None = None, circuit=None) -> ExecutionConfig:
        config = config or ExecutionConfig()
        updates = {}
        if config.gradient_method in ("best", None):
            updates["gradient_method"] = "parameter-shift"
        updates["use_device_gradient"] = False
        updates["convert_to_numpy"] = True
        return replace(config, **updates)

    def preprocess_transforms(self, execution_config: ExecutionConfig | None = None) -> CompilePipeline:
        program = CompilePipeline()
        program.add_transform(validate_device_wires, self.wires, name=self.name)
        program.add_transform(no_sampling, name=self.name)
        program.add_transform(decompose, stopping_condition=_stopping_condition, device_wires=self.wires,
                              skip_initial_state_prep=True, name=self.name)
        program.add_transform(validate_measurements, analytic_measurements=_accepted_measurement, name=self.name)
        return program

    def supports_derivatives(self, execution_config=None, circuit=None) -> bool:
        return False

    def _circuit(self, low):
        c = self._cache.get(low.key)
        if c is None:
            if len(self._cache) >= self._CACHE_SIZE:
                self._cache.pop(next(iter(self._cache)))
            c = self._cache[low.key] = Circuit(low.spec)
        return c

    def execute(self, circuits, execution_config: ExecutionConfig | None = None):
        results = []
        for tape in circuits:
            wire_order = self.wires if self.wires is not None else tape.wires
            low = _Lowered(tape, list(wire_order), self._dtype)
            c = self._circuit(low)
            out = c.execute(low.X, low.W, dtype=self._dtype) if c.n_outputs else np.zeros((low.X.shape[0], 0))
            psi = c.final_states(low.X, low.W) if low.needs_state else None
            results.append(low.results(out, psi))
        return tuple(results)
