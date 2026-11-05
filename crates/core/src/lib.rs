//! Batched state-vector simulator for large batches of small circuits.
//!
//! The target workload is quanvolutional preprocessing: one 1 to 8 qubit circuit per image
//! patch, thousands to hundreds of thousands of patches per call. Everything is built around
//! one batch-first call: a [`Circuit`] template, a `(batch, d_x)` matrix of per-circuit inputs,
//! and a vector of weights shared by the whole batch, giving a `(batch, n_outputs)` result.
//!
//! Conventions match PennyLane: wire 0 is the most significant bit of the basis index, gate
//! matrices are PennyLane's, and multi-wire measurement outputs (probs) order wires as given.
//!
//! ```
//! use qbatch_core::{Circuit, Gate, Param, Pauli};
//!
//! // RY(x_0) on wire 0, CNOT(0, 1), then <Z_0> and <Z_1>.
//! let mut c = Circuit::new(2);
//! c.gate(Gate::RY(Param::input(0)), &[0]).gate(Gate::CNOT, &[0, 1]);
//! c.expval(&[(0, Pauli::Z)]).expval(&[(1, Pauli::Z)]);
//! let inputs = [0.0_f64, std::f64::consts::PI]; // two circuits, one input column each
//! let out = qbatch_core::execute(&c, 2, &inputs, &[]).unwrap();
//! assert!((out[0] - 1.0).abs() < 1e-12 && (out[2] + 1.0).abs() < 1e-12);
//! ```

mod circuit;
mod exec;
mod gates;
mod program;
mod real;
mod rng;

pub use circuit::{Circuit, Gate, Measurement, Op, Param, Pauli, PauliWord, StatePrep};
pub use exec::{execute, execute_into, final_states, sample};
pub use num_complex::Complex;
pub use real::Real;

/// Largest supported register. States live in per-thread scratch buffers of 2^n amplitudes.
pub const MAX_QUBITS: usize = 16;

/// Any invalid circuit or input shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error(msg.into()))
}
