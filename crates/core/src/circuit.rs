//! Circuit templates: what runs, independent of the batch it runs on.

use num_complex::Complex;

use crate::{err, Result, MAX_QUBITS};

/// Where a gate angle comes from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Param {
    /// The same value for every circuit.
    Const(f64),
    /// `scale * inputs[circuit, col]`: differs per circuit (data encoding, per-patch angles).
    Input { col: usize, scale: f64 },
    /// `scale * weights[idx]`: shared by the whole batch (trainable or fixed filter weights).
    Weight { idx: usize, scale: f64 },
}

impl Param {
    pub fn input(col: usize) -> Self {
        Param::Input { col, scale: 1.0 }
    }
    pub fn weight(idx: usize) -> Self {
        Param::Weight { idx, scale: 1.0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pauli {
    I,
    X,
    Y,
    Z,
}

/// Gates, with PennyLane's names and matrix conventions.
#[derive(Clone, Debug, PartialEq)]
pub enum Gate {
    PauliX,
    PauliY,
    PauliZ,
    Hadamard,
    S,
    /// Adjoint of S.
    Sdg,
    T,
    /// Adjoint of T.
    Tdg,
    SX,
    RX(Param),
    RY(Param),
    RZ(Param),
    PhaseShift(Param),
    /// `Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi)`.
    Rot(Param, Param, Param),
    /// `U3(theta, phi, delta)`.
    U3(Param, Param, Param),
    CNOT,
    CY,
    CZ,
    SWAP,
    CRX(Param),
    CRY(Param),
    CRZ(Param),
    ControlledPhaseShift(Param),
    IsingXX(Param),
    IsingYY(Param),
    IsingZZ(Param),
    /// `exp(-i theta/2 Z...Z)` on all of the op's wires.
    MultiRZ(Param),
    /// `exp(-i theta/2 P)` for a Pauli word given per wire (same length as the op's wires).
    PauliRot(Param, Vec<Pauli>),
    /// Constant 2^k x 2^k matrix, row-major, for k = 1 or 2 wires.
    QubitUnitary(Vec<Complex<f64>>),
}

impl Gate {
    /// Number of wires the gate acts on, or None if it takes any number (MultiRZ, PauliRot,
    /// QubitUnitary, which are checked against their own data instead).
    pub fn arity(&self) -> Option<usize> {
        use Gate::*;
        match self {
            PauliX | PauliY | PauliZ | Hadamard | S | Sdg | T | Tdg | SX | RX(_) | RY(_) | RZ(_)
            | PhaseShift(_) | Rot(..) | U3(..) => Some(1),
            CNOT | CY | CZ | SWAP | CRX(_) | CRY(_) | CRZ(_) | ControlledPhaseShift(_) | IsingXX(_)
            | IsingYY(_) | IsingZZ(_) => Some(2),
            MultiRZ(_) | PauliRot(..) | QubitUnitary(_) => None,
        }
    }

    pub fn params(&self) -> Vec<Param> {
        use Gate::*;
        match self {
            RX(p) | RY(p) | RZ(p) | PhaseShift(p) | CRX(p) | CRY(p) | CRZ(p) | ControlledPhaseShift(p)
            | IsingXX(p) | IsingYY(p) | IsingZZ(p) | MultiRZ(p) | PauliRot(p, _) => vec![*p],
            Rot(a, b, c) | U3(a, b, c) => vec![*a, *b, *c],
            _ => vec![],
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Op {
    pub gate: Gate,
    pub wires: Vec<usize>,
}

/// Load the initial state from the per-circuit inputs, on all wires, before any gate.
/// Loading is a copy, never a gate cascade. Inputs must already be normalized, as with
/// `qml.StatePrep(..., normalize=False)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatePrep {
    /// 2^n real amplitudes from input columns `start..start + 2^n`.
    Real { start: usize },
    /// 2^n complex amplitudes as interleaved (re, im) pairs in columns `start..start + 2^(n+1)`.
    Complex { start: usize },
}

/// Product of single-wire Paulis on distinct wires. An empty word is the identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PauliWord(pub Vec<(usize, Pauli)>);

#[derive(Clone, Debug, PartialEq)]
pub enum Measurement {
    Expval(PauliWord),
    Var(PauliWord),
    /// Probabilities of the 2^k basis states of these wires, first wire most significant.
    Probs(Vec<usize>),
}

impl Measurement {
    pub fn width(&self) -> usize {
        match self {
            Measurement::Expval(_) | Measurement::Var(_) => 1,
            Measurement::Probs(w) => 1 << w.len(),
        }
    }
}

/// A circuit template: optional state loading, a gate list, and measurements.
#[derive(Clone, Debug, PartialEq)]
pub struct Circuit {
    pub n_qubits: usize,
    pub state_prep: Option<StatePrep>,
    pub ops: Vec<Op>,
    pub measurements: Vec<Measurement>,
}

impl Circuit {
    pub fn new(n_qubits: usize) -> Self {
        Circuit { n_qubits, state_prep: None, ops: vec![], measurements: vec![] }
    }

    pub fn dim(&self) -> usize {
        1 << self.n_qubits
    }

    pub fn state_prep(&mut self, sp: StatePrep) -> &mut Self {
        self.state_prep = Some(sp);
        self
    }

    pub fn gate(&mut self, gate: Gate, wires: &[usize]) -> &mut Self {
        self.ops.push(Op { gate, wires: wires.to_vec() });
        self
    }

    pub fn expval(&mut self, word: &[(usize, Pauli)]) -> &mut Self {
        self.measurements.push(Measurement::Expval(PauliWord(word.to_vec())));
        self
    }

    pub fn var(&mut self, word: &[(usize, Pauli)]) -> &mut Self {
        self.measurements.push(Measurement::Var(PauliWord(word.to_vec())));
        self
    }

    pub fn probs(&mut self, wires: &[usize]) -> &mut Self {
        self.measurements.push(Measurement::Probs(wires.to_vec()));
        self
    }

    /// Total width of one circuit's output row.
    pub fn n_outputs(&self) -> usize {
        self.measurements.iter().map(Measurement::width).sum()
    }

    /// Smallest per-circuit input width and weight count this circuit can run with.
    pub fn min_shapes(&self) -> (usize, usize) {
        let (mut dx, mut nw) = (0, 0);
        match self.state_prep {
            Some(StatePrep::Real { start }) => dx = start + self.dim(),
            Some(StatePrep::Complex { start }) => dx = start + 2 * self.dim(),
            None => {}
        }
        for op in &self.ops {
            for p in op.gate.params() {
                match p {
                    Param::Input { col, .. } => dx = dx.max(col + 1),
                    Param::Weight { idx, .. } => nw = nw.max(idx + 1),
                    Param::Const(_) => {}
                }
            }
        }
        (dx, nw)
    }

    /// Structural checks that do not depend on the batch.
    pub fn validate(&self) -> Result<()> {
        let n = self.n_qubits;
        if n == 0 || n > MAX_QUBITS {
            return err(format!("n_qubits must be 1..={MAX_QUBITS}, got {n}"));
        }
        let distinct = |wires: &[usize], what: &str| -> Result<()> {
            for (i, &w) in wires.iter().enumerate() {
                if w >= n {
                    return err(format!("{what}: wire {w} out of range for {n} qubits"));
                }
                if wires[..i].contains(&w) {
                    return err(format!("{what}: repeated wire {w}"));
                }
            }
            Ok(())
        };
        for (k, op) in self.ops.iter().enumerate() {
            let what = format!("op {k} ({:?})", op.gate);
            distinct(&op.wires, &what)?;
            let k_w = op.wires.len();
            let ok = match &op.gate {
                Gate::MultiRZ(_) => k_w >= 1,
                Gate::PauliRot(_, word) => k_w >= 1 && word.len() == k_w,
                Gate::QubitUnitary(m) => (k_w == 1 || k_w == 2) && m.len() == 1 << (2 * k_w),
                g => g.arity() == Some(k_w),
            };
            if !ok {
                return err(format!("{what}: wrong number of wires ({k_w}) or matrix size"));
            }
        }
        for (k, m) in self.measurements.iter().enumerate() {
            let what = format!("measurement {k}");
            match m {
                Measurement::Expval(w) | Measurement::Var(w) => {
                    let wires: Vec<usize> = w.0.iter().map(|t| t.0).collect();
                    distinct(&wires, &what)?;
                }
                Measurement::Probs(wires) => {
                    if wires.is_empty() {
                        return err(format!("{what}: probs needs at least one wire"));
                    }
                    distinct(wires, &what)?;
                }
            }
        }
        Ok(())
    }
}
