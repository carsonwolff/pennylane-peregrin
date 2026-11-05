//! Build a `Circuit` from the JSON spec written by tests/export_golden.py.

use peregrin_core::{Circuit, Complex};
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

pub fn circuit(spec: &Value) -> Circuit {
    Circuit::from_json_value(spec).expect("valid circuit spec")
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
