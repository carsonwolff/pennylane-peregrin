"""Backend builders.

Each builder takes (Workload, precision) and returns (run, info):
  run(features: np.ndarray, weights: np.ndarray) -> np.ndarray of shape (batch, 2n)
  info: dict recorded with the result (effective precision, device notes, ...)

Timing is end to end, numpy in -> numpy out, because that is what a
preprocessing pipeline actually pays (host->device copies included).
Imports are local so a missing package only disables that backend.
"""
import numpy as np

from workload import Workload, apply_circuit, entangler_pairs


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
        return apply_circuit(qml, f, w, wl.n_qubits, wl.n_layers, wl.embed)

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
        return apply_circuit(qml, f, w, wl.n_qubits, wl.n_layers, wl.embed)

    fn = jax.jit(lambda F, W: _stack_outputs(qnode(F, W), jnp))

    def run(F, W):
        return np.asarray(fn(F, W))

    return run, {"precision_effective": precision, "jax_device": str(jax.devices()[0])}


# ---------------------------------------------------------------- hand-written JAX
def build_jax_hand(wl: Workload, precision):
    """Batched state-vector simulator written directly in JAX, one jit for the batch.

    State is (batch, 2**n); wire 0 is the most significant bit (PennyLane order).
    Each CNOT layer is precomposed into a single gather.
    """
    jax = _jax_setup(precision)
    import jax.numpy as jnp

    n, L, D = wl.n_qubits, wl.n_layers, wl.dim
    cdt = jnp.complex128 if precision == "f64" else jnp.complex64

    idx = np.arange(D)
    layer_perm = idx.copy()
    for c, t in entangler_pairs(n):
        cbit = (idx >> (n - 1 - c)) & 1
        p = idx ^ (cbit << (n - 1 - t))  # CNOT: psi_new[i] = psi_old[p[i]]
        layer_perm = layer_perm[p]  # compose in application order
    layer_perm = jnp.asarray(layer_perm)

    def apply_1q(psi, U, q):
        B = psi.shape[0]
        s = psi.reshape(B, 2**q, 2, 2 ** (n - q - 1))
        s = jnp.einsum("ab,xibj->xiaj", U, s)
        return s.reshape(B, D)

    def gate(ry, rz):
        c, s = jnp.cos(ry / 2), jnp.sin(ry / 2)
        RY = jnp.array([[c, -s], [s, c]], dtype=cdt)
        e = jnp.exp(-0.5j * rz).astype(cdt)
        RZ = jnp.array([[e, 0], [0, jnp.conj(e)]], dtype=cdt)
        return RZ @ RY  # RY first, then RZ

    @jax.jit
    def fn(F, W):
        psi = F.astype(cdt)
        for l in range(L):
            for q in range(n):
                psi = apply_1q(psi, gate(W[l, q, 0], W[l, q, 1]), q)
            psi = psi[:, layer_perm]
        B = psi.shape[0]
        zs, xs = [], []
        for q in range(n):
            a = psi.reshape(B, 2**q, 2, 2 ** (n - q - 1))
            p = jnp.abs(a) ** 2
            zs.append(jnp.sum(p[:, :, 0, :] - p[:, :, 1, :], axis=(1, 2)))
            xs.append(2 * jnp.sum(jnp.real(jnp.conj(a[:, :, 0, :]) * a[:, :, 1, :]), axis=(1, 2)))
        return jnp.stack(zs + xs, axis=-1)

    def run(F, W):
        return np.asarray(fn(F, W))

    return run, {"precision_effective": precision, "jax_device": str(jax.devices()[0])}


# ---------------------------------------------------------------- lightning (no qjit)
def _build_lightning_plain(device_name):
    def build(wl: Workload, precision):
        import pennylane as qml

        kw = {"c_dtype": np.complex64} if precision == "f32" else {}
        dev = qml.device(device_name, wires=wl.n_qubits, **kw)

        @qml.qnode(dev, diff_method=None)
        def qnode(f, w):
            return apply_circuit(qml, f, w, wl.n_qubits, wl.n_layers, wl.embed)

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
            return apply_circuit(qml, f, w, wl.n_qubits, wl.n_layers, wl.embed)

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
            return apply_circuit(qml, f, w, wl.n_qubits, wl.n_layers, wl.embed)

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
