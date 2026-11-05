"""The peregrin PennyLane device must agree with default.qubit (run: pytest crates/python/tests)."""
import numpy as np
import pennylane as qml
import pytest
from pennylane import numpy as pnp

TOL = 1e-12


def both(n, fn, **kw):
    """The same QNode body on peregrin and default.qubit."""
    return (qml.QNode(fn, qml.device("peregrin.cpu", wires=n, **kw)), qml.QNode(fn, qml.device("default.qubit", wires=n)))


def close(a, b, tol=TOL):
    if isinstance(a, (tuple, list)):
        assert isinstance(b, (tuple, list)) and len(a) == len(b)
        for x, y in zip(a, b):
            close(x, y, tol)
        return
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, f"shape {a.shape} vs {b.shape}"
    assert np.allclose(a, b, atol=tol, rtol=0), f"max err {np.abs(a - b).max():.2e}"


def test_device_loads_by_name():
    dev = qml.device("peregrin.cpu", wires=3)
    assert dev.name == "peregrin.cpu" and len(dev.wires) == 3


GATES_1 = [qml.Hadamard, qml.PauliX, qml.PauliY, qml.PauliZ, qml.S, qml.T, qml.SX, qml.adjoint(qml.S), qml.adjoint(qml.T)]
PARAM_1 = [qml.RX, qml.RY, qml.RZ, qml.PhaseShift]
PARAM_2 = [qml.CRX, qml.CRY, qml.CRZ, qml.ControlledPhaseShift, qml.IsingXX, qml.IsingYY, qml.IsingZZ]
FIXED_2 = [qml.CNOT, qml.CY, qml.CZ, qml.SWAP]


def kitchen_sink(x, w):
    """Every native gate, plus some that must decompose, on 4 wires."""
    for i in range(4):
        qml.Hadamard(wires=i)
    for k, g in enumerate(GATES_1):
        g(wires=k % 4)
    for k, g in enumerate(PARAM_1):
        g(x[..., k], wires=k % 4)
    qml.Rot(x[..., 0], w[0], x[..., 1], wires=1)
    qml.U3(w[1], x[..., 2], w[2], wires=2)
    for k, g in enumerate(PARAM_2):
        g(x[..., k % 3] * (k + 1), wires=[k % 4, (k + 1) % 4])
    for k, g in enumerate(FIXED_2):
        g(wires=[(k + 2) % 4, k % 4])
    qml.MultiRZ(w[0], wires=[0, 1, 3])
    qml.PauliRot(x[..., 1], "XYZ", wires=[3, 1, 0])
    qml.QubitUnitary(qml.matrix(qml.CRot(0.3, 0.4, 0.5, wires=[0, 1])), wires=[2, 0])
    qml.QubitUnitary(qml.matrix(qml.Rot(0.1, 0.2, 0.3, wires=0)), wires=3)
    qml.Toffoli(wires=[0, 1, 2])  # decomposes
    qml.CRot(w[0], x[..., 0], w[1], wires=[3, 2])  # decomposes
    qml.QFT(wires=[0, 1, 2, 3])  # decomposes
    qml.GlobalPhase(0.7)
    return [qml.expval(qml.Z(i)) for i in range(4)] + [qml.expval(qml.X(i) @ qml.Y((i + 1) % 4)) for i in range(4)]


def test_every_gate_matches_default_qubit_single_and_broadcast():
    rng = np.random.default_rng(0)
    w = rng.uniform(-np.pi, np.pi, 3)
    pg, dq = both(4, kitchen_sink)
    x1 = rng.uniform(-np.pi, np.pi, 4)
    close(pg(x1, w), dq(x1, w))
    xb = rng.uniform(-np.pi, np.pi, (37, 4))
    close(pg(xb, w), dq(xb, w))


def test_measurements():
    def body(x):
        qml.RY(x, wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RX(0.3 * x, wires=2)
        qml.Hadamard(wires=1)
        H = 0.5 * qml.X(0) @ qml.Z(1) - 1.25 * qml.Y(2) + 0.1 * qml.Identity(0) + qml.Z(0) @ qml.Z(2)
        return (qml.expval(H), qml.var(qml.Y(1)), qml.var(H), qml.var(2.0 * qml.X(0)), qml.probs(wires=[2, 0]),
                qml.probs(), qml.state(), qml.expval(qml.Identity(1)))

    pg, dq = both(3, body)
    close(pg(0.7), dq(0.7))
    xb = np.linspace(-2, 2, 11)
    close(pg(xb), dq(xb))


def test_state_prep_and_amplitude_embedding():
    rng = np.random.default_rng(1)

    def sp(v):
        qml.StatePrep(v, wires=range(3))
        qml.RY(0.4, wires=1)
        qml.CNOT(wires=[1, 2])
        return qml.expval(qml.Z(2)), qml.expval(qml.X(0)), qml.state()

    def amp(v):
        qml.AmplitudeEmbedding(v, wires=range(3), normalize=True)
        qml.RZ(0.2, wires=0)
        qml.Hadamard(wires=0)
        return [qml.expval(qml.Z(i)) for i in range(3)]

    def amp_sub(v):  # embedding on a subset of the wires
        qml.AmplitudeEmbedding(v, wires=[2, 0], normalize=True)
        qml.CNOT(wires=[2, 1])
        return qml.probs()

    real = rng.normal(size=(9, 8))
    real /= np.linalg.norm(real, axis=1, keepdims=True)
    cplx = rng.normal(size=(9, 8)) + 1j * rng.normal(size=(9, 8))
    cplx /= np.linalg.norm(cplx, axis=1, keepdims=True)
    for body, inputs in ((sp, [real, real[0], cplx, cplx[3]]), (amp, [rng.uniform(size=(6, 8))]),
                         (amp_sub, [rng.uniform(size=(5, 4))])):
        pg, dq = both(3, body)
        for v in inputs:
            close(pg(v), dq(v))


def test_basis_state_first():
    def body(b):
        qml.BasisState(b, wires=[0, 1, 2])
        qml.RY(0.3, wires=1)
        qml.CNOT(wires=[0, 2])
        return qml.probs(), qml.expval(qml.Z(2))

    pg, dq = both(3, body)
    for b in ([1, 0, 1], [0, 1, 1]):
        close(pg(np.array(b)), dq(np.array(b)))


def test_wire_labels_and_no_device_wires():
    def body(x):
        qml.RY(x, wires="a")
        qml.CNOT(wires=["a", 7])
        return qml.expval(qml.Z(7)), qml.probs(wires=["a"])

    pg, dq = (qml.QNode(body, qml.device("peregrin.cpu", wires=["a", 7])), qml.QNode(body, qml.device("default.qubit", wires=["a", 7])))
    close(pg(np.array([0.1, 0.2])), dq(np.array([0.1, 0.2])))
    pg, dq = (qml.QNode(body, qml.device("peregrin.cpu")), qml.QNode(body, qml.device("default.qubit")))
    close(pg(0.4), dq(0.4))


def test_gradients_with_parameter_shift():
    def body(x, w):
        qml.RY(x, wires=0)
        qml.RX(w[0], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.Rot(w[1], w[2], x, wires=1)
        return qml.expval(qml.Z(1) @ qml.X(0) + 0.5 * qml.Y(1))

    pg = qml.QNode(body, qml.device("peregrin.cpu", wires=2))
    dq = qml.QNode(body, qml.device("default.qubit", wires=2), diff_method="parameter-shift")
    x = pnp.array(0.3, requires_grad=True)
    w = pnp.array([0.1, -0.7, 1.2], requires_grad=True)
    close(qml.grad(pg)(x, w), qml.grad(dq)(x, w), 1e-10)


def test_jax_interface():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)

    def body(x):
        qml.RY(x, wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.Z(1))

    pg = qml.QNode(body, qml.device("peregrin.cpu", wires=2), interface="jax")
    assert abs(float(pg(jax.numpy.array(0.3))) - np.cos(0.3)) < TOL
    assert abs(float(jax.grad(pg)(jax.numpy.array(0.3))) + np.sin(0.3)) < 1e-10


def test_f32_precision():
    rng = np.random.default_rng(2)
    w = rng.uniform(-np.pi, np.pi, 3)
    pg = qml.QNode(kitchen_sink, qml.device("peregrin.cpu", wires=4, precision="f32"))
    dq = qml.QNode(kitchen_sink, qml.device("default.qubit", wires=4))
    xb = rng.uniform(-np.pi, np.pi, (20, 4))
    close(pg(xb, w), dq(xb, w), 1e-4)


def test_shots_raise_and_matrix_observables_work():
    def shots():
        qml.Hadamard(0)
        return qml.sample(qml.Z(0))

    with pytest.raises(Exception, match="shots|sampl|analytic"):
        qml.set_shots(qml.QNode(shots, qml.device("peregrin.cpu", wires=1)), shots=10)()

    def mats(v):
        qml.RY(v, wires=0)
        qml.CNOT(wires=[0, 1])
        A = np.array([[1, 0.5j], [-0.5j, 2.0]])
        B = np.kron(A, np.diag([1.0, -3.0]))
        return (qml.expval(qml.Hermitian(A, wires=0)), qml.var(qml.Hermitian(B, wires=[1, 0])),
                qml.expval(qml.Projector([1, 0], wires=[0, 1])))

    pg, dq = both(2, mats)
    close(pg(0.8), dq(0.8))
    close(pg(np.array([0.1, 2.0])), dq(np.array([0.1, 2.0])))


def test_cache_reuses_circuits_across_weight_changes():
    dev = qml.device("peregrin.cpu", wires=2)

    @qml.qnode(dev)
    def f(w):
        qml.RY(w, wires=0)
        return qml.expval(qml.Z(0))

    for w in np.linspace(0, 1, 5):
        assert abs(f(w) - np.cos(w)) < TOL
    assert len(dev._cache) == 1
