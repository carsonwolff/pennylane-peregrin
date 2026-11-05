//! Outputs and final states must match default.qubit: 1e-12 in f64, 1e-4 in f32.

mod common;

use peregrin_core::{execute, final_states};

fn check(file: &str) {
    let cases = common::load(file);
    assert!(!cases.is_empty());
    let mut worst = (0.0f64, String::new());
    for c in &cases {
        let out = execute::<f64>(&c.circuit, c.batch, &c.inputs, &c.weights).unwrap_or_else(|e| panic!("{}: {e}", c.name));
        assert_eq!(out.len(), c.expected.len(), "{}: output width", c.name);
        let e_out = out.iter().zip(&c.expected).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);

        let psi = final_states::<f64>(&c.circuit, c.batch, &c.inputs, &c.weights).unwrap();
        let e_psi = psi.iter().zip(&c.states).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max);
        assert!(e_out < 1e-12 && e_psi < 1e-12, "{}: f64 max err outputs {e_out:.2e}, state {e_psi:.2e}", c.name);
        if e_out.max(e_psi) >= worst.0 {
            worst = (e_out.max(e_psi), c.name.clone());
        }

        let x32: Vec<f32> = c.inputs.iter().map(|&v| v as f32).collect();
        let w32: Vec<f32> = c.weights.iter().map(|&v| v as f32).collect();
        let out32 = execute::<f32>(&c.circuit, c.batch, &x32, &w32).unwrap();
        let e32 = out32.iter().zip(&c.expected).map(|(&a, b)| (a as f64 - b).abs()).fold(0.0, f64::max);
        assert!(e32 < 1e-4, "{}: f32 max err {e32:.2e}", c.name);
    }
    eprintln!("{file}: {} cases, worst f64 error {:.1e} ({})", cases.len(), worst.0, worst.1);
}

#[test]
fn every_gate_matches_default_qubit() {
    check("gates.json");
}

#[test]
fn qbench_circuits_match_default_qubit() {
    check("qbench.json");
}

#[test]
fn random_circuits_match_default_qubit() {
    check("random.json");
}
