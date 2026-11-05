//! Build a `Circuit` from the JSON spec written by tests/export_golden.py.

use peregrin_core::{Circuit, Complex, Gate, Measurement, Op, Param, Pauli, PauliWord, StatePrep};
use serde_json::Value;

pub struct Case {
    pub name: String,
    pub circuit: Circuit,
    pub batch: usize,
    pub inputs: Vec<f64>,
    pub weights: Vec<f64>,
    pub expected: Vec<f64>,
    pub states: Vec<Complex<f64>>,
}

fn f(v: &Value) -> f64 {
    v.as_f64().expect("number")
}

fn u(v: &Value) -> usize {
    v.as_u64().expect("index") as usize
}

fn pauli(s: &str) -> Pauli {
    match s {
        "I" => Pauli::I,
        "X" => Pauli::X,
        "Y" => Pauli::Y,
        "Z" => Pauli::Z,
        _ => panic!("bad pauli {s}"),
    }
}

fn param(v: &Value) -> Param {
    let a = v.as_array().unwrap();
    match a[0].as_str().unwrap() {
        "c" => Param::Const(f(&a[1])),
        "x" => Param::Input { col: u(&a[1]), scale: f(&a[2]) },
        "w" => Param::Weight { idx: u(&a[1]), scale: f(&a[2]) },
        k => panic!("bad param kind {k}"),
    }
}

pub fn circuit(spec: &Value) -> Circuit {
    let mut c = Circuit::new(u(&spec["n_qubits"]));
    if let Some(sp) = spec["state_prep"].as_object() {
        let start = u(&sp["start"]);
        c.state_prep = Some(match sp["kind"].as_str().unwrap() {
            "real" => StatePrep::Real { start },
            _ => StatePrep::Complex { start },
        });
    }
    for op in spec["ops"].as_array().unwrap() {
        let wires: Vec<usize> = op["wires"].as_array().unwrap().iter().map(u).collect();
        let ps: Vec<Param> = op.get("params").and_then(Value::as_array).map(|a| a.iter().map(param).collect()).unwrap_or_default();
        let p = |i: usize| ps[i];
        let gate = match op["name"].as_str().unwrap() {
            "PauliX" => Gate::PauliX,
            "PauliY" => Gate::PauliY,
            "PauliZ" => Gate::PauliZ,
            "Hadamard" => Gate::Hadamard,
            "S" => Gate::S,
            "Sdg" => Gate::Sdg,
            "T" => Gate::T,
            "Tdg" => Gate::Tdg,
            "SX" => Gate::SX,
            "RX" => Gate::RX(p(0)),
            "RY" => Gate::RY(p(0)),
            "RZ" => Gate::RZ(p(0)),
            "PhaseShift" => Gate::PhaseShift(p(0)),
            "Rot" => Gate::Rot(p(0), p(1), p(2)),
            "U3" => Gate::U3(p(0), p(1), p(2)),
            "CNOT" => Gate::CNOT,
            "CY" => Gate::CY,
            "CZ" => Gate::CZ,
            "SWAP" => Gate::SWAP,
            "CRX" => Gate::CRX(p(0)),
            "CRY" => Gate::CRY(p(0)),
            "CRZ" => Gate::CRZ(p(0)),
            "ControlledPhaseShift" => Gate::ControlledPhaseShift(p(0)),
            "IsingXX" => Gate::IsingXX(p(0)),
            "IsingYY" => Gate::IsingYY(p(0)),
            "IsingZZ" => Gate::IsingZZ(p(0)),
            "MultiRZ" => Gate::MultiRZ(p(0)),
            "PauliRot" => {
                let word = op["word"].as_str().unwrap().chars().map(|ch| pauli(&ch.to_string())).collect();
                Gate::PauliRot(p(0), word)
            }
            "QubitUnitary" => Gate::QubitUnitary(
                op["matrix"].as_array().unwrap().iter().map(|z| Complex::new(f(&z[0]), f(&z[1]))).collect(),
            ),
            g => panic!("unknown gate {g}"),
        };
        c.ops.push(Op { gate, wires });
    }
    for m in spec["measurements"].as_array().unwrap() {
        let meas = match m["type"].as_str().unwrap() {
            "probs" => Measurement::Probs(m["wires"].as_array().unwrap().iter().map(u).collect()),
            t => {
                let word = PauliWord(
                    m["word"].as_array().unwrap().iter().map(|t| (u(&t[0]), pauli(t[1].as_str().unwrap()))).collect(),
                );
                if t == "expval" {
                    Measurement::Expval(word)
                } else {
                    Measurement::Var(word)
                }
            }
        };
        c.measurements.push(meas);
    }
    c
}

pub fn load(file: &str) -> Vec<Case> {
    let path = format!("{}/tests/golden/{file}", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}. Run tests/export_golden.py"));
    let v: Value = serde_json::from_str(&text).unwrap();
    let nums = |v: &Value| -> Vec<f64> { v.as_array().unwrap().iter().map(f).collect() };
    v["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| Case {
            name: c["name"].as_str().unwrap().to_string(),
            circuit: circuit(&c["circuit"]),
            batch: u(&c["batch"]),
            inputs: nums(&c["inputs"]),
            weights: nums(&c["weights"]),
            expected: nums(&c["expected"]),
            states: c["states"].as_array().unwrap().iter().map(|z| Complex::new(f(&z[0]), f(&z[1]))).collect(),
        })
        .collect()
}
