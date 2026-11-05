"""Backend builders.

Each builder takes (Workload, precision) and returns (run, info):
  run(features: np.ndarray, weights: np.ndarray) -> np.ndarray of shape (batch, 2n)
  info: dict recorded with the result (effective precision, device notes, ...)

Timing is end to end, numpy in -> numpy out, because that is what a
preprocessing pipeline actually pays (host->device copies included).
Imports are local so a missing package only disables that backend.
"""
import numpy as np

import circuits
from workload import Workload


# ---------------------------------------------------------------- helpers
def _jax_setup(precision):
    import jax

    jax.config.update("jax_enable_x64", precision == "f64")
    return jax


def _stack_outputs(out, xp):
    """PennyLane returns a tuple of 2n arrays of shape (batch,)."""
    return xp.stack(list(out), axis=-1)


# ---------------------------------------------------------------- default.qubit
def build_default_qubit(wl: Workload, precision):
    """default.qubit, native parameter broadcasting, numpy, no jit. Also the reference."""
    import pennylane as qml

    dev = qml.device("default.qubit", wires=wl.n_qubits)

    @qml.qnode(dev, interface=None, diff_method=None)
    def qnode(f, w):
        return circuits.get(wl).body(qml, f, w, wl)

    def run(F, W):
        return np.asarray(_stack_outputs(qnode(F, W), np))

    return run, {"precision_effective": "f64"}


def build_dq_jax(wl: Workload, precision):
    """default.qubit + parameter broadcasting + jax.jit (platform set by JAX_PLATFORMS)."""
    jax = _jax_setup(precision)
    import jax.numpy as jnp
    import pennylane as qml

    dev = qml.device("default.qubit", wires=wl.n_qubits)

    @qml.qnode(dev, interface="jax", diff_method=None)
    def qnode(f, w):
        return circuits.get(wl).body(qml, f, w, wl)

    fn = jax.jit(lambda F, W: _stack_outputs(qnode(F, W), jnp))

    def run(F, W):
        return np.asarray(fn(F, W))

    return run, {"precision_effective": precision, "jax_device": str(jax.devices()[0])}


# ---------------------------------------------------------------- hand-written JAX
def build_jax_hand(wl: Workload, precision):
    """Batched state-vector simulator written directly in JAX, one jit for the batch.

    State is (batch, 2**n); wire 0 is the most significant bit (PennyLane order).
    Every circuit, generic included, runs through the template interpreter below. Until
    2026-11-05 the generic circuit used a specialized einsum version (the original phase 1
    bar, see git history); the elementwise interpreter measured 7 to 41% faster on it on
    the i5-6300U, so it replaced it to keep the baseline as strong as possible.
    """
    return _build_jax_template(wl, precision)


def _build_jax_template(wl: Workload, precision):
    """Hand-written batched JAX simulator driven by circuits.<name>.template.

    Same tricks as the specialized version, applied generically: consecutive single-qubit
    gates on a wire are fused into one 2x2 matrix (per-circuit when any of them takes a
    per-circuit angle), and runs of permutation gates (CNOT) become a single gather.
    """
    jax = _jax_setup(precision)
    import jax.numpy as jnp

    n, D = wl.n_qubits, wl.dim
    cdt = jnp.complex128 if precision == "f64" else jnp.complex64
    ops, meas = circuits.get(wl).template(wl)
    idx = np.arange(D)

    def bit(q):
        return n - 1 - q  # wire 0 is the most significant bit

    def perm_of(name, wires):
        if name == "CNOT":
            c, t = wires
            return idx ^ (((idx >> bit(c)) & 1) << bit(t))  # psi_new[i] = psi_old[p[i]]
        raise NotImplementedError(f"jax template interpreter: no permutation for {name}")

    # Compile the op list into stages: ("prep", (a, b)), ("u", wire, [ops]), ("perm", p).
    stages, pending = [], {}

    def flush():
        for q in sorted(pending):
            stages.append(("u", q, pending[q]))
        pending.clear()

    for name, wires, param in ops:
        if name == "StatePrep":
            assert not stages and not pending, "StatePrep must come first"
            stages.append(("prep", param[1]))
        elif len(wires) == 1:
            pending.setdefault(wires[0], []).append((name, param))
        else:
            flush()
            p = perm_of(name, wires)
            if stages and stages[-1][0] == "perm":
                stages[-1] = ("perm", stages[-1][1][p])  # compose in application order
            else:
                stages.append(("perm", p))
    flush()
    stages = [("perm", jnp.asarray(st[1])) if st[0] == "perm" else st for st in stages]

    def angle(param, X, Wf):
        kind, i, scale = param
        return scale * (X[:, i] if kind == "x" else Wf[i])

    # A 2x2 matrix is a tuple of four entries (u00, u01, u10, u11). Each entry is a scalar
    # when the gate is shared by the whole batch, or a (B,) array when it takes a
    # per-circuit angle. Entries are combined and applied elementwise, so per-circuit gates
    # cost a few fused multiply-adds per amplitude instead of a batched matmul.
    def mat(name, th):
        if name == "H":
            h = 1 / np.sqrt(2)
            return (h, h, h, -h)
        c, s = jnp.cos(th / 2), jnp.sin(th / 2)
        if name == "RX":
            return (c, -1j * s, -1j * s, c)
        if name == "RY":
            return (c, -s, s, c)
        if name == "RZ":
            e = jnp.exp(-0.5j * th)
            return (e, 0.0, 0.0, jnp.conj(e))
        raise NotImplementedError(f"jax template interpreter: no matrix for {name}")

    def mul(G, U):  # G @ U
        g00, g01, g10, g11 = G
        u00, u01, u10, u11 = U
        return (g00 * u00 + g01 * u10, g00 * u01 + g01 * u11, g10 * u00 + g11 * u10, g10 * u01 + g11 * u11)

    def apply_1q(psi, U, q):
        B = psi.shape[0]
        s = psi.reshape(B, 2**q, 2, 2 ** (n - q - 1))
        a0, a1 = s[:, :, 0, :], s[:, :, 1, :]
        u = [x.astype(cdt)[:, None, None] if jnp.ndim(x) else x for x in U]
        b0 = u[0] * a0 + u[1] * a1
        b1 = u[2] * a0 + u[3] * a1
        return jnp.stack([b0, b1], axis=2).reshape(B, D)

    @jax.jit
    def fn(X, W):
        Wf = W.reshape(-1)
        B = X.shape[0]
        psi = None
        for st in stages:
            if st[0] == "prep":
                a, b = st[1]
                psi = X[:, a:b].astype(cdt)
            elif st[0] == "u":
                if psi is None:
                    psi = jnp.zeros((B, D), cdt).at[:, 0].set(1)
                U = None
                for name, param in st[2]:
                    G = mat(name, None if param is None else angle(param, X, Wf))
                    U = G if U is None else mul(G, U)  # later gates multiply on the left
                psi = apply_1q(psi, U, st[1])
            else:
                if psi is None:
                    psi = jnp.zeros((B, D), cdt).at[:, 0].set(1)
                psi = psi[:, st[1]]
        if psi is None:
            psi = jnp.zeros((B, D), cdt).at[:, 0].set(1)
        cols = []
        for kind, q in meas:
            a = psi.reshape(B, 2**q, 2, 2 ** (n - q - 1))
            if kind == "Z":
                p = jnp.real(a) ** 2 + jnp.imag(a) ** 2
                cols.append(jnp.sum(p[:, :, 0, :] - p[:, :, 1, :], axis=(1, 2)))
            else:
                cols.append(2 * jnp.sum(jnp.real(jnp.conj(a[:, :, 0, :]) * a[:, :, 1, :]), axis=(1, 2)))
        return jnp.stack(cols, axis=-1)

    def run(F, W):
        return np.asarray(fn(F, W))

    n_u = sum(1 for st in stages if st[0] == "u")
    n_p = sum(1 for st in stages if st[0] == "perm")
    return run, {"precision_effective": precision, "jax_device": str(jax.devices()[0]),
                 "note": f"template interpreter: {n_u} fused 1q stages, {n_p} gathers"}


# ---------------------------------------------------------------- lightning (no qjit)
def _build_lightning_plain(device_name):
    def build(wl: Workload, precision):
        import pennylane as qml

        kw = {"c_dtype": np.complex64} if precision == "f32" else {}
        dev = qml.device(device_name, wires=wl.n_qubits, **kw)

        @qml.qnode(dev, diff_method=None)
        def qnode(f, w):
            return circuits.get(wl).body(qml, f, w, wl)

        def run(F, W):
            # Broadcasting is supported by splitting into `batch` tapes.
            return np.asarray(_stack_outputs(qnode(F, W), np))

        return run, {"precision_effective": precision, "note": "broadcast split into per-circuit tapes"}

    return build


# ---------------------------------------------------------------- lightning + qjit
def _build_lightning_qjit(device_name):
    def build(wl: Workload, precision):
        # Catalyst runs in f64; f32 is not exposed through qjit, so we record f64.
        jax = _jax_setup("f64")
        import jax.numpy as jnp
        import catalyst
        import pennylane as qml

        dev = qml.device(device_name, wires=wl.n_qubits)

        @qml.qnode(dev)
        def qnode(f, w):
            return circuits.get(wl).body(qml, f, w, wl)

        @qml.qjit
        def fn(F, W):
            out = catalyst.vmap(qnode, in_axes=(0, None))(F, W)
            return jnp.stack(list(out), axis=-1)

        def run(F, W):
            return np.asarray(fn(np.asarray(F, np.float64), np.asarray(W, np.float64)))

        return run, {"precision_effective": "f64", "note": "catalyst.vmap over batch"}

    return build


# ---------------------------------------------------------------- qulacs
def _build_qulacs(gpu):
    def build(wl: Workload, precision):
        import pennylane as qml

        dev = qml.device("qulacs.simulator", wires=wl.n_qubits, gpu=gpu)

        @qml.qnode(dev, diff_method=None)
        def qnode(f, w):
            return circuits.get(wl).body(qml, f, w, wl)

        def run(F, W):
            return np.asarray(_stack_outputs(qnode(F, W), np))

        return run, {"precision_effective": "f64",
                     "note": "broadcast split per circuit; X and Z on one wire do not commute, so 2 executions per circuit"}

    return build


# ---------------------------------------------------------------- registry
# name -> (builder, env overrides for the worker process, needs_gpu)
BACKENDS = {
    "default.qubit": (build_default_qubit, {"JAX_PLATFORMS": "cpu"}, False),
    "dq-jax-cpu": (build_dq_jax, {"JAX_PLATFORMS": "cpu"}, False),
    "dq-jax-gpu": (build_dq_jax, {"JAX_PLATFORMS": "cuda"}, True),
    "jax-hand-cpu": (build_jax_hand, {"JAX_PLATFORMS": "cpu"}, False),
    "jax-hand-gpu": (build_jax_hand, {"JAX_PLATFORMS": "cuda"}, True),
    "lightning.qubit": (_build_lightning_plain("lightning.qubit"), {"JAX_PLATFORMS": "cpu"}, False),
    "lightning.qubit+qjit": (_build_lightning_qjit("lightning.qubit"), {"JAX_PLATFORMS": "cpu"}, False),
    "lightning.gpu": (_build_lightning_plain("lightning.gpu"), {"JAX_PLATFORMS": "cpu"}, True),
    "lightning.gpu+qjit": (_build_lightning_qjit("lightning.gpu"), {"JAX_PLATFORMS": "cpu"}, True),
    "qulacs": (_build_qulacs(False), {"JAX_PLATFORMS": "cpu"}, False),
    "qulacs-gpu": (_build_qulacs(True), {"JAX_PLATFORMS": "cpu"}, True),
}

REFERENCE = "default.qubit"
