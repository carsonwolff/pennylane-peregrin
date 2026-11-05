//! Quick in-process throughput check for development. Not the published benchmark: that runs
//! through qbench (numpy in, numpy out) once the Python bindings exist.
//!
//!   cargo run --release --bin peregrin-bench -- [generic|owner-mri] [f64|f32] [threads]

use std::time::Instant;

use peregrin_core::{execute_into, Circuit, Gate, Param, Pauli, Real, StatePrep};

fn generic(n: usize) -> Circuit {
    let mut c = Circuit::new(n);
    c.state_prep(StatePrep::Real { start: 0 });
    for q in 0..n {
        c.gate(Gate::RY(Param::weight(2 * q)), &[q]).gate(Gate::RZ(Param::weight(2 * q + 1)), &[q]);
    }
    for q in 0..n - 1 {
        c.gate(Gate::CNOT, &[q, q + 1]);
    }
    if n > 2 {
        c.gate(Gate::CNOT, &[n - 1, 0]);
    }
    for q in 0..n {
        c.expval(&[(q, Pauli::Z)]);
    }
    for q in 0..n {
        c.expval(&[(q, Pauli::X)]);
    }
    c
}

/// The paper's circuits; inputs are [amplitudes | mean, std, var, median, max] (already * pi).
fn owner_mri(n: usize) -> Circuit {
    let d = 1 << n;
    let s = |i: usize| Param::input(d + i);
    let mut c = Circuit::new(n);
    c.state_prep(StatePrep::Real { start: 0 });
    if n == 2 {
        c.gate(Gate::Hadamard, &[0]).gate(Gate::Hadamard, &[1]);
        c.gate(Gate::RY(s(0)), &[0]).gate(Gate::RZ(s(1)), &[1]).gate(Gate::CNOT, &[0, 1]);
        c.gate(Gate::RY(s(2)), &[1]).gate(Gate::RX(s(3)), &[0]);
        c.expval(&[(0, Pauli::Z)]).expval(&[(1, Pauli::Z)]).expval(&[(0, Pauli::X)]).expval(&[(1, Pauli::X)]);
    } else {
        for _ in 0..2 {
            for q in 0..4 {
                c.gate(Gate::Hadamard, &[q]);
            }
            c.gate(Gate::RY(s(0)), &[0]).gate(Gate::RZ(s(1)), &[1]).gate(Gate::RX(s(3)), &[2]).gate(Gate::RY(s(4)), &[3]);
            for q in 0..3 {
                c.gate(Gate::CNOT, &[q, q + 1]);
            }
            c.gate(Gate::CNOT, &[3, 0]);
        }
        for q in 0..4 {
            c.expval(&[(q, Pauli::Z)]).expval(&[(q, Pauli::X)]);
        }
    }
    c
}

fn inputs<T: Real>(batch: usize, d: usize, extra: usize) -> Vec<T> {
    let mut s = 0x2545F4914F6CDD1Du64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut x = Vec::with_capacity(batch * (d + extra));
    for _ in 0..batch {
        let row: Vec<f64> = (0..d).map(|_| next() + 1e-6).collect();
        let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        x.extend(row.iter().map(|v| T::of(v / norm)));
        x.extend((0..extra).map(|_| T::of(next())));
    }
    x
}

fn bench<T: Real>(name: &str, qubits: &[usize], batches: &[usize]) {
    println!("{name} ({}): circuits/s", std::any::type_name::<T>());
    for &n in qubits {
        let (c, extra) = if name == "owner-mri" { (owner_mri(n), 5) } else { (generic(n), 0) };
        let w: Vec<T> = (0..2 * n).map(|i| T::of(0.37 * i as f64 + 0.1)).collect();
        let mut line = format!("  n={n}");
        for &b in batches {
            let x = inputs::<T>(b, 1 << n, extra);
            let mut out = vec![T::zero(); b * c.n_outputs()];
            execute_into(&c, b, &x, &w, &mut out).unwrap(); // warm up
            let mut times = vec![];
            let t_all = Instant::now();
            while times.len() < 5 || (t_all.elapsed().as_secs_f64() < 1.0 && times.len() < 200) {
                let t = Instant::now();
                execute_into(&c, b, &x, &w, &mut out).unwrap();
                times.push(t.elapsed().as_secs_f64());
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            line += &format!("  B={b}: {:>12.0}", b as f64 / times[times.len() / 2]);
        }
        println!("{line}");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let circuit = args.get(1).map(String::as_str).unwrap_or("generic");
    let prec = args.get(2).map(String::as_str).unwrap_or("f64");
    if let Some(t) = args.get(3) {
        rayon::ThreadPoolBuilder::new().num_threads(t.parse().unwrap()).build_global().unwrap();
    }
    let qubits: &[usize] = if circuit == "owner-mri" { &[2, 4] } else { &[2, 4, 6, 8] };
    let batches = [100, 1000, 10_000, 100_000];
    match prec {
        "f32" => bench::<f32>(circuit, qubits, &batches),
        _ => bench::<f64>(circuit, qubits, &batches),
    }
}
