//! Gates against known matrices, norm preservation, sampling, threading, and input checks.

use std::f64::consts::{FRAC_1_SQRT_2, PI};

use peregrin_core::{execute, final_states, sample, Circuit, Complex, Gate, Param, Pauli, StatePrep};

type C = Complex<f64>;

fn c(re: f64, im: f64) -> C {
    Complex::new(re, im)
}

/// Final state of one circuit started from basis state `basis`.
fn run_from(n: usize, basis: usize, ops: &[(Gate, Vec<usize>)]) -> Vec<C> {
    let mut circ = Circuit::new(n);
    circ.state_prep(StatePrep::Real { start: 0 });
    for (g, w) in ops {
        circ.gate(g.clone(), w);
    }
    let mut x = vec![0.0; 1 << n];
    x[basis] = 1.0;
    final_states::<f64>(&circ, 1, &x, &[]).unwrap()
}

fn assert_state(got: &[C], want: &[C]) {
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!((a - b).norm() < 1e-12, "amplitude {i}: got {a}, want {b}\nfull: {got:?}");
    }
}

fn k(x: f64) -> Param {
    Param::Const(x)
}

#[test]
fn single_qubit_known_values() {
    let h = FRAC_1_SQRT_2;
    assert_state(&run_from(1, 0, &[(Gate::Hadamard, vec![0])]), &[c(h, 0.), c(h, 0.)]);
    assert_state(&run_from(1, 1, &[(Gate::Hadamard, vec![0])]), &[c(h, 0.), c(-h, 0.)]);
    assert_state(&run_from(1, 0, &[(Gate::PauliX, vec![0])]), &[c(0., 0.), c(1., 0.)]);
    assert_state(&run_from(1, 0, &[(Gate::PauliY, vec![0])]), &[c(0., 0.), c(0., 1.)]);
    assert_state(&run_from(1, 1, &[(Gate::PauliZ, vec![0])]), &[c(0., 0.), c(-1., 0.)]);
    assert_state(&run_from(1, 1, &[(Gate::S, vec![0])]), &[c(0., 0.), c(0., 1.)]);
    assert_state(&run_from(1, 1, &[(Gate::Sdg, vec![0])]), &[c(0., 0.), c(0., -1.)]);
    assert_state(&run_from(1, 1, &[(Gate::T, vec![0])]), &[c(0., 0.), c(h, h)]);
    assert_state(&run_from(1, 1, &[(Gate::Tdg, vec![0])]), &[c(0., 0.), c(h, -h)]);
    assert_state(&run_from(1, 0, &[(Gate::SX, vec![0]), (Gate::SX, vec![0])]), &[c(0., 0.), c(1., 0.)]);
    assert_state(&run_from(1, 0, &[(Gate::RX(k(PI)), vec![0])]), &[c(0., 0.), c(0., -1.)]);
    assert_state(&run_from(1, 0, &[(Gate::RY(k(PI / 2.)), vec![0])]), &[c(h, 0.), c(h, 0.)]);
    let t = 0.7;
    assert_state(
        &run_from(1, 0, &[(Gate::Hadamard, vec![0]), (Gate::RZ(k(t)), vec![0])]),
        &[C::from_polar(h, -t / 2.), C::from_polar(h, t / 2.)],
    );
    assert_state(&run_from(1, 1, &[(Gate::PhaseShift(k(t)), vec![0])]), &[c(0., 0.), C::from_polar(1., t)]);
}

#[test]
fn rot_and_u3_match_their_definitions() {
    let (a, b, d) = (0.3, 1.1, -0.8);
    for basis in 0..2 {
        let rot = run_from(1, basis, &[(Gate::Rot(k(a), k(b), k(d)), vec![0])]);
        let seq =
            run_from(1, basis, &[(Gate::RZ(k(a)), vec![0]), (Gate::RY(k(b)), vec![0]), (Gate::RZ(k(d)), vec![0])]);
        assert_state(&rot, &seq);
        // U3(theta, phi, delta) = e^{i(phi+delta)/2} RZ(phi) RY(theta) RZ(delta)
        let u3 = run_from(1, basis, &[(Gate::U3(k(b), k(a), k(d)), vec![0])]);
        let seq =
            run_from(1, basis, &[(Gate::RZ(k(d)), vec![0]), (Gate::RY(k(b)), vec![0]), (Gate::RZ(k(a)), vec![0])]);
        let g = C::from_polar(1., (a + d) / 2.);
        assert_state(&u3, &seq.iter().map(|z| z * g).collect::<Vec<_>>());
    }
}

#[test]
fn two_qubit_known_values_and_wire_order() {
    // wire 0 is the most significant bit: |10> is index 2.
    let e = |i: usize| (0..4).map(|j| if i == j { c(1., 0.) } else { c(0., 0.) }).collect::<Vec<_>>();
    assert_state(&run_from(2, 2, &[(Gate::CNOT, vec![0, 1])]), &e(3));
    assert_state(&run_from(2, 1, &[(Gate::CNOT, vec![0, 1])]), &e(1));
    assert_state(&run_from(2, 1, &[(Gate::CNOT, vec![1, 0])]), &e(3));
    assert_state(&run_from(2, 2, &[(Gate::SWAP, vec![0, 1])]), &e(1));
    assert_state(&run_from(2, 3, &[(Gate::CZ, vec![0, 1])]), &e(3).iter().map(|z| -z).collect::<Vec<_>>());
    assert_state(&run_from(2, 2, &[(Gate::CY, vec![0, 1])]), &[c(0., 0.), c(0., 0.), c(0., 0.), c(0., 1.)]);
    // IsingXX(pi) = -i XX
    assert_state(
        &run_from(2, 0, &[(Gate::IsingXX(k(PI)), vec![0, 1])]),
        &[c(0., 0.), c(0., 0.), c(0., 0.), c(0., -1.)],
    );
    // ControlledPhaseShift only touches |11>
    let t = 0.4;
    assert_state(
        &run_from(2, 3, &[(Gate::ControlledPhaseShift(k(t)), vec![0, 1])]),
        &[c(0., 0.), c(0., 0.), c(0., 0.), C::from_polar(1., t)],
    );
    assert_state(&run_from(2, 2, &[(Gate::ControlledPhaseShift(k(t)), vec![0, 1])]), &e(2));
    // CRX(pi) with control set acts as -iX on the target
    assert_state(&run_from(2, 2, &[(Gate::CRX(k(PI)), vec![0, 1])]), &[c(0., 0.), c(0., 0.), c(0., 0.), c(0., -1.)]);
    // PauliRot with word "ZZ" equals IsingZZ and MultiRZ
    let a = run_from(2, 1, &[(Gate::Hadamard, vec![0]), (Gate::PauliRot(k(t), vec![Pauli::Z, Pauli::Z]), vec![0, 1])]);
    let b = run_from(2, 1, &[(Gate::Hadamard, vec![0]), (Gate::IsingZZ(k(t)), vec![0, 1])]);
    let m = run_from(2, 1, &[(Gate::Hadamard, vec![0]), (Gate::MultiRZ(k(t)), vec![0, 1])]);
    assert_state(&a, &b);
    assert_state(&a, &m);
}

/// Minimal xorshift so the property tests need no extra crates.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn angle(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 * PI - PI
    }
}

fn random_circuit(rng: &mut Rng, n: usize, n_ops: usize) -> Circuit {
    let mut circ = Circuit::new(n);
    circ.state_prep(StatePrep::Real { start: 0 });
    for _ in 0..n_ops {
        let a = rng.below(n);
        let b = (a + 1 + rng.below(n.max(2) - 1)) % n;
        let p = if rng.below(2) == 0 {
            Param::Input { col: (1 << n) + rng.below(3), scale: 1.0 }
        } else {
            Param::weight(rng.below(4))
        };
        let pick = if n == 1 { rng.below(6) } else { rng.below(13) };
        let (g, w) = match pick {
            0 => (Gate::Hadamard, vec![a]),
            1 => (Gate::RX(p), vec![a]),
            2 => (Gate::RY(p), vec![a]),
            3 => (Gate::RZ(p), vec![a]),
            4 => (Gate::Rot(p, Param::Const(0.3), p), vec![a]),
            5 => (Gate::T, vec![a]),
            6 => (Gate::CNOT, vec![a, b]),
            7 => (Gate::CZ, vec![a, b]),
            8 => (Gate::SWAP, vec![a, b]),
            9 => (Gate::CRY(p), vec![a, b]),
            10 => (Gate::IsingYY(p), vec![a, b]),
            11 => (Gate::PauliRot(p, vec![Pauli::X, Pauli::Y]), vec![a, b]),
            _ => (Gate::ControlledPhaseShift(p), vec![a, b]),
        };
        circ.gate(g, &w);
    }
    circ
}

fn random_inputs(rng: &mut Rng, n: usize, batch: usize) -> Vec<f64> {
    let d = 1 << n;
    let mut x = Vec::with_capacity(batch * (d + 3));
    for _ in 0..batch {
        let row: Vec<f64> = (0..d).map(|_| rng.angle()).collect();
        let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        x.extend(row.iter().map(|v| v / norm));
        x.extend((0..3).map(|_| rng.angle()));
    }
    x
}

#[test]
fn random_circuits_preserve_norm_up_to_12_qubits() {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    for n in 1..=12 {
        for _ in 0..4 {
            let circ = random_circuit(&mut rng, n, 40);
            let batch = if n <= 8 { 16 } else { 2 };
            let x = random_inputs(&mut rng, n, batch);
            let w: Vec<f64> = (0..4).map(|_| rng.angle()).collect();
            let psi = final_states::<f64>(&circ, batch, &x, &w).unwrap();
            for row in psi.chunks(1 << n) {
                let norm: f64 = row.iter().map(|z| z.norm_sqr()).sum();
                assert!((norm - 1.0).abs() < 1e-12, "n={n}: norm {norm}");
            }
        }
    }
}

#[test]
fn f32_tracks_f64() {
    let mut rng = Rng(42);
    for n in 1..=8 {
        let mut circ = random_circuit(&mut rng, n, 30);
        for q in 0..n {
            circ.expval(&[(q, Pauli::Z)]).expval(&[(q, Pauli::X)]);
        }
        if n > 1 {
            circ.probs(&[n - 1, 0]);
        }
        let x = random_inputs(&mut rng, n, 64);
        let w: Vec<f64> = (0..4).map(|_| rng.angle()).collect();
        let a = execute::<f64>(&circ, 64, &x, &w).unwrap();
        let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let w32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let b = execute::<f32>(&circ, 64, &x32, &w32).unwrap();
        let err = a.iter().zip(&b).map(|(p, &q)| (p - q as f64).abs()).fold(0.0, f64::max);
        assert!(err < 1e-4, "n={n}: f32 error {err}");
    }
}

#[test]
fn results_do_not_depend_on_thread_count() {
    let mut rng = Rng(7);
    let n = 4;
    let mut circ = random_circuit(&mut rng, n, 25);
    circ.expval(&[(0, Pauli::Z), (2, Pauli::Y)]).var(&[(1, Pauli::X)]).probs(&[3, 1]);
    let batch = 5000;
    let x = random_inputs(&mut rng, n, batch);
    let w = [0.1, 0.2, 0.3, 0.4];
    let pool = |t| rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap();
    let one = pool(1).install(|| {
        (execute::<f64>(&circ, batch, &x, &w).unwrap(), sample::<f64>(&circ, batch, &x, &w, &[0, 3], 50, 9).unwrap())
    });
    let many = pool(4).install(|| {
        (execute::<f64>(&circ, batch, &x, &w).unwrap(), sample::<f64>(&circ, batch, &x, &w, &[0, 3], 50, 9).unwrap())
    });
    assert_eq!(one.0, many.0);
    assert_eq!(one.1, many.1);
}

#[test]
fn sample_frequencies_match_probabilities() {
    // RY(theta) on wire 0, CNOT: P(00) = cos^2(theta/2), P(11) = sin^2(theta/2).
    let theta = 1.2_f64;
    let mut circ = Circuit::new(2);
    circ.gate(Gate::RY(Param::Const(theta)), &[0]).gate(Gate::CNOT, &[0, 1]);
    let shots = 200_000;
    let s = sample::<f64>(&circ, 1, &[], &[], &[0, 1], shots, 123).unwrap();
    let mut counts = [0usize; 4];
    for &v in &s {
        counts[v as usize] += 1;
    }
    assert_eq!(counts[1] + counts[2], 0);
    let p11 = (theta / 2.0).sin().powi(2);
    let f11 = counts[3] as f64 / shots as f64;
    let sigma = (p11 * (1.0 - p11) / shots as f64).sqrt();
    assert!((f11 - p11).abs() < 5.0 * sigma, "freq {f11} vs prob {p11}");
    // same seed, same samples; different seed, different samples
    assert_eq!(s, sample::<f64>(&circ, 1, &[], &[], &[0, 1], shots, 123).unwrap());
    assert_ne!(s, sample::<f64>(&circ, 1, &[], &[], &[0, 1], shots, 124).unwrap());
}

#[test]
fn x_and_z_on_one_wire_come_from_one_execution() {
    // Not a timing test: just that non-commuting observables share one output row.
    let mut circ = Circuit::new(1);
    circ.gate(Gate::RY(Param::input(0)), &[0]).expval(&[(0, Pauli::Z)]).expval(&[(0, Pauli::X)]).var(&[(0, Pauli::Z)]);
    let t = 0.9;
    let out = execute::<f64>(&circ, 1, &[t], &[]).unwrap();
    assert!((out[0] - t.cos()).abs() < 1e-12);
    assert!((out[1] - t.sin()).abs() < 1e-12);
    assert!((out[2] - t.sin().powi(2)).abs() < 1e-12);
}

#[test]
fn bad_inputs_are_errors_not_panics() {
    let mut circ = Circuit::new(2);
    circ.state_prep(StatePrep::Real { start: 0 }).gate(Gate::RY(Param::input(4)), &[0]).expval(&[(0, Pauli::Z)]);
    assert!(execute::<f64>(&circ, 2, &[0.0; 8], &[]).is_err(), "needs 5 columns");
    assert!(execute::<f64>(&circ, 2, &[0.0; 9], &[]).is_err(), "not a multiple of batch");
    assert!(execute::<f64>(&circ, 2, &[0.0; 10], &[]).is_ok());

    let mut bad = Circuit::new(2);
    bad.gate(Gate::CNOT, &[0, 0]);
    assert!(execute::<f64>(&bad, 1, &[], &[]).is_err(), "repeated wire");
    let mut bad = Circuit::new(2);
    bad.gate(Gate::Hadamard, &[2]);
    assert!(execute::<f64>(&bad, 1, &[], &[]).is_err(), "wire out of range");
    let mut bad = Circuit::new(2);
    bad.gate(Gate::RX(Param::weight(3)), &[0]);
    assert!(execute::<f64>(&bad, 1, &[], &[0.0; 3]).is_err(), "needs 4 weights");
}

/// <psi| P |psi> computed directly from a state, for checking the batched measurement path.
fn pauli_expval_direct(n: usize, psi: &[C], word: &[(usize, Pauli)]) -> f64 {
    let mut acc = c(0.0, 0.0);
    for (i, a) in psi.iter().enumerate() {
        // P|i> = phase |j>
        let (mut j, mut ph) = (i, c(1.0, 0.0));
        for &(w, p) in word {
            let bit = 1 << (n - 1 - w);
            let one = i & bit != 0;
            match p {
                Pauli::I => {}
                Pauli::X => j ^= bit,
                Pauli::Y => {
                    j ^= bit;
                    ph *= if one { c(0.0, -1.0) } else { c(0.0, 1.0) };
                }
                Pauli::Z => {
                    if one {
                        ph = -ph;
                    }
                }
            }
        }
        acc += psi[j].conj() * ph * a;
    }
    acc.re
}

#[test]
fn batched_outputs_match_final_states_at_every_tile_size() {
    let mut rng = Rng(1234);
    let words: Vec<Vec<(usize, Pauli)>> = vec![
        vec![(0, Pauli::Z)],
        vec![(0, Pauli::X)],
        vec![(0, Pauli::Y)],
        vec![(0, Pauli::X), (1, Pauli::Y)],
        vec![(0, Pauli::Y), (1, Pauli::Y)],
        vec![(0, Pauli::Z), (1, Pauli::Z)],
        vec![(0, Pauli::Y), (1, Pauli::Z), (2, Pauli::Y)],
        vec![(0, Pauli::Y), (1, Pauli::X), (2, Pauli::Y), (3, Pauli::Y)],
    ];
    for n in 1..=11 {
        for &batch in &[1usize, 7, 8, 9, 37] {
            let mut circ = random_circuit(&mut rng, n, 30);
            let usable: Vec<&Vec<(usize, Pauli)>> = words.iter().filter(|w| w.len() <= n).collect();
            for (k, w) in usable.iter().enumerate() {
                // Map the word onto shuffled wires so wire order is exercised too.
                let ww: Vec<(usize, Pauli)> = w.iter().map(|&(q, p)| ((q + k) % n, p)).collect();
                if k % 2 == 0 {
                    circ.expval(&ww);
                } else {
                    circ.var(&ww);
                }
            }
            if n > 1 {
                circ.probs(&[n - 1, 0]);
            }
            let x = random_inputs(&mut rng, n, batch);
            let wts: Vec<f64> = (0..4).map(|_| rng.angle()).collect();
            let out = execute::<f64>(&circ, batch, &x, &wts).unwrap();
            let psi = final_states::<f64>(&circ, batch, &x, &wts).unwrap();
            let n_out = circ.n_outputs();
            for b in 0..batch {
                let s = &psi[b * (1 << n)..(b + 1) * (1 << n)];
                let mut want = vec![];
                for (k, w) in usable.iter().enumerate() {
                    let ww: Vec<(usize, Pauli)> = w.iter().map(|&(q, p)| ((q + k) % n, p)).collect();
                    let e = pauli_expval_direct(n, s, &ww);
                    want.push(if k % 2 == 0 { e } else { 1.0 - e * e });
                }
                if n > 1 {
                    let mut pr = [0.0; 4];
                    for (i, a) in s.iter().enumerate() {
                        pr[2 * (i & 1) + (i >> (n - 1))] += a.norm_sqr();
                    }
                    want.extend(pr);
                }
                let got = &out[b * n_out..(b + 1) * n_out];
                for (o, (g, wv)) in got.iter().zip(&want).enumerate() {
                    assert!((g - wv).abs() < 1e-12, "n={n} batch={batch} row={b} output {o}: {g} vs {wv}");
                }
            }
        }
    }
}
