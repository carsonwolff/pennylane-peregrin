//! Circuits as JSON, the format the Python bindings and the golden tests use.
//!
//! ```json
//! {"n_qubits": 2,
//!  "state_prep": null | {"kind": "real" | "complex", "start": 0},
//!  "ops": [{"name": "RY", "wires": [0], "params": [["x", 4, 1.0]]},
//!          {"name": "CNOT", "wires": [0, 1]},
//!          {"name": "PauliRot", "wires": [0, 1], "params": [["w", 0, 1.0]], "word": "XY"},
//!          {"name": "QubitUnitary", "wires": [0], "matrix": [[re, im], ...]}],
//!  "measurements": [{"type": "expval" | "var", "word": [[0, "Z"], [1, "X"]]},
//!                   {"type": "probs", "wires": [0, 1]}]}
//! ```
//! Params are `["x", col, scale]` (per-circuit input), `["w", idx, scale]` (shared weight) or
//! `["c", value]` (constant).

use num_complex::Complex;
use serde_json::Value;

use crate::{err, Circuit, Error, Gate, Measurement, Op, Param, Pauli, PauliWord, Result, StatePrep};

fn bad<T>(what: &str) -> Result<T> {
    err(format!("circuit json: {what}"))
}

fn num(v: &Value, what: &str) -> Result<f64> {
    v.as_f64().ok_or_else(|| Error(format!("circuit json: {what} must be a number")))
}

fn idx(v: &Value, what: &str) -> Result<usize> {
    v.as_u64().map(|u| u as usize).ok_or_else(|| Error(format!("circuit json: {what} must be a non-negative integer")))
}

fn arr<'a>(v: &'a Value, what: &str) -> Result<&'a Vec<Value>> {
    v.as_array().ok_or_else(|| Error(format!("circuit json: {what} must be an array")))
}

fn pauli(s: &str) -> Result<Pauli> {
    Ok(match s {
        "I" => Pauli::I,
        "X" => Pauli::X,
        "Y" => Pauli::Y,
        "Z" => Pauli::Z,
        _ => return bad(&format!("unknown Pauli {s:?}")),
    })
}

fn param(v: &Value) -> Result<Param> {
    let a = arr(v, "param")?;
    let kind = a.first().and_then(Value::as_str).unwrap_or("");
    let at = |i: usize| a.get(i).ok_or_else(|| Error(format!("circuit json: param {v} is too short")));
    Ok(match kind {
        "c" => Param::Const(num(at(1)?, "constant")?),
        "x" => Param::Input { col: idx(at(1)?, "input column")?, scale: num(at(2)?, "scale")? },
        "w" => Param::Weight { idx: idx(at(1)?, "weight index")?, scale: num(at(2)?, "scale")? },
        _ => return bad(&format!("param kind must be \"x\", \"w\" or \"c\", got {v}")),
    })
}

impl Circuit {
    /// Parse a circuit from JSON text (format in the module docs).
    pub fn from_json(text: &str) -> Result<Circuit> {
        let v: Value = serde_json::from_str(text).map_err(|e| Error(format!("circuit json: {e}")))?;
        Circuit::from_json_value(&v)
    }

    pub fn from_json_value(spec: &Value) -> Result<Circuit> {
        let n = idx(&spec["n_qubits"], "n_qubits")?;
        let mut c = Circuit::new(n);
        if let Some(sp) = spec.get("state_prep").and_then(Value::as_object) {
            let start = idx(&sp["start"], "state_prep.start")?;
            c.state_prep = Some(match sp.get("kind").and_then(Value::as_str) {
                Some("real") => StatePrep::Real { start },
                Some("complex") => StatePrep::Complex { start },
                _ => return bad("state_prep.kind must be \"real\" or \"complex\""),
            });
        }
        for op in arr(&spec["ops"], "ops")? {
            let name = op["name"].as_str().ok_or_else(|| Error("circuit json: op without a name".into()))?;
            let wires = arr(&op["wires"], "wires")?.iter().map(|w| idx(w, "wire")).collect::<Result<Vec<_>>>()?;
            let ps = match op.get("params") {
                Some(p) => arr(p, "params")?.iter().map(param).collect::<Result<Vec<_>>>()?,
                None => vec![],
            };
            let p = |i: usize| {
                ps.get(i).copied().ok_or_else(|| Error(format!("circuit json: {name} needs {} params", i + 1)))
            };
            let gate = match name {
                "PauliX" | "X" => Gate::PauliX,
                "PauliY" | "Y" => Gate::PauliY,
                "PauliZ" | "Z" => Gate::PauliZ,
                "Hadamard" | "H" => Gate::Hadamard,
                "S" => Gate::S,
                "Sdg" => Gate::Sdg,
                "T" => Gate::T,
                "Tdg" => Gate::Tdg,
                "SX" => Gate::SX,
                "RX" => Gate::RX(p(0)?),
                "RY" => Gate::RY(p(0)?),
                "RZ" => Gate::RZ(p(0)?),
                "PhaseShift" => Gate::PhaseShift(p(0)?),
                "Rot" => Gate::Rot(p(0)?, p(1)?, p(2)?),
                "U3" => Gate::U3(p(0)?, p(1)?, p(2)?),
                "CNOT" => Gate::CNOT,
                "CY" => Gate::CY,
                "CZ" => Gate::CZ,
                "SWAP" => Gate::SWAP,
                "CRX" => Gate::CRX(p(0)?),
                "CRY" => Gate::CRY(p(0)?),
                "CRZ" => Gate::CRZ(p(0)?),
                "ControlledPhaseShift" => Gate::ControlledPhaseShift(p(0)?),
                "IsingXX" => Gate::IsingXX(p(0)?),
                "IsingYY" => Gate::IsingYY(p(0)?),
                "IsingZZ" => Gate::IsingZZ(p(0)?),
                "MultiRZ" => Gate::MultiRZ(p(0)?),
                "PauliRot" => {
                    let word =
                        op["word"].as_str().ok_or_else(|| Error("circuit json: PauliRot needs a word".into()))?;
                    let word = word.chars().map(|ch| pauli(&ch.to_string())).collect::<Result<Vec<_>>>()?;
                    Gate::PauliRot(p(0)?, word)
                }
                "QubitUnitary" => {
                    let m = arr(&op["matrix"], "matrix")?
                        .iter()
                        .map(|z| {
                            let z = arr(z, "matrix entry")?;
                            match z.as_slice() {
                                [re, im] => Ok(Complex::new(num(re, "re")?, num(im, "im")?)),
                                _ => bad("matrix entries are [re, im]"),
                            }
                        })
                        .collect::<Result<Vec<_>>>()?;
                    Gate::QubitUnitary(m)
                }
                g => return bad(&format!("unknown gate {g:?}")),
            };
            c.ops.push(Op { gate, wires });
        }
        for m in arr(&spec["measurements"], "measurements")? {
            let meas = match m["type"].as_str() {
                Some("probs") => Measurement::Probs(
                    arr(&m["wires"], "wires")?.iter().map(|w| idx(w, "wire")).collect::<Result<_>>()?,
                ),
                Some(t @ ("expval" | "var")) => {
                    let word = arr(&m["word"], "word")?
                        .iter()
                        .map(|t| {
                            let w = idx(&t[0], "word wire")?;
                            let p = pauli(t[1].as_str().unwrap_or(""))?;
                            Ok((w, p))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    if t == "expval" {
                        Measurement::Expval(PauliWord(word))
                    } else {
                        Measurement::Var(PauliWord(word))
                    }
                }
                _ => return bad("measurement type must be \"expval\", \"var\" or \"probs\""),
            };
            c.measurements.push(meas);
        }
        c.validate()?;
        Ok(c)
    }
}
