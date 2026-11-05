//! Lowering a `Circuit` plus the batch's shared weights into a flat list of stages.
//!
//! Done once per call, so everything that is the same for every circuit in the batch is
//! computed once: shared-angle gate matrices are evaluated, consecutive single-qubit gates on a
//! wire are fused into one 2x2 matrix, and runs of permutation gates (CNOT, SWAP) become one
//! precomposed gather. Only angles read from per-circuit inputs are evaluated per circuit.

use crate::circuit::{Circuit, Gate, Measurement, Pauli, PauliWord, StatePrep};
use crate::gates::{is_diag2, mat1, mul2, C, K1, M2};
use crate::{err, Param, Real, Result};

/// An angle after binding the weights: either known now, or read from this circuit's inputs.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Val<T> {
    Fixed(T),
    Input { col: usize, scale: T },
}

impl<T: Real> Val<T> {
    #[inline(always)]
    fn get(&self, x: &[T]) -> T {
        match *self {
            Val::Fixed(v) => v,
            Val::Input { col, scale } => scale * x[col],
        }
    }
}

#[derive(Clone, Debug)]
enum Factor<T> {
    Fixed(M2<T>),
    Dyn { k: K1, p: [Val<T>; 3] },
}

/// Product of single-qubit factors, applied first to last.
#[derive(Clone, Debug)]
struct Fused<T> {
    factors: Vec<Factor<T>>,
    diag: bool,
}

impl<T: Real> Fused<T> {
    fn new() -> Self {
        Fused { factors: vec![], diag: true }
    }

    fn push(&mut self, f: Factor<T>) {
        match (&f, self.factors.last_mut()) {
            (Factor::Fixed(m), Some(Factor::Fixed(prev))) => {
                *prev = mul2(m, prev);
                self.diag = self.diag && is_diag2(prev);
                return;
            }
            (Factor::Fixed(m), _) => self.diag = self.diag && is_diag2(m),
            (Factor::Dyn { k, .. }, _) => self.diag = self.diag && k.is_diagonal(),
        }
        self.factors.push(f);
    }

    #[inline(always)]
    fn eval(&self, x: &[T]) -> M2<T> {
        let mut u: Option<M2<T>> = None;
        for f in &self.factors {
            let m = match f {
                Factor::Fixed(m) => *m,
                Factor::Dyn { k, p } => mat1(*k, [p[0].get(x), p[1].get(x), p[2].get(x)]),
            };
            u = Some(match u {
                None => m,
                Some(prev) => mul2(&m, &prev),
            });
        }
        u.expect("fused group is never empty")
    }
}

#[derive(Clone, Debug)]
enum Stage<T> {
    /// Single-qubit unitary on the amplitude pairs `stride` apart.
    U1 { s: usize, f: Fused<T> },
    /// Controlled single-qubit unitary: control mask `cm`, target stride `ts`.
    Ctrl1 { cm: usize, ts: usize, f: Fused<T> },
    /// psi_new[i] = psi_old[p[i]].
    Perm(Box<[u32]>),
    /// exp(-i theta/2 P) for a Pauli word with flip mask, Z-or-Y mask, and Y count.
    PRot { flip: usize, zy: usize, ny: u8, theta: Val<T> },
    /// Constant two-qubit unitary on wires with masks m0 (first wire) and m1.
    Dense2 { m0: usize, m1: usize, m: Box<[C<T>; 16]> },
}

#[derive(Clone, Debug)]
enum MeasPlan {
    Expval { flip: usize, zy: usize, ny: u8 },
    Var { flip: usize, zy: usize, ny: u8 },
    /// Output slot for each basis index.
    Probs { map: Box<[u32]>, width: usize },
}

pub(crate) struct Program<T> {
    pub(crate) dim: usize,
    pub(crate) n_out: usize,
    prep: Option<StatePrep>,
    stages: Vec<Stage<T>>,
    meas: Vec<MeasPlan>,
}

/// Per-thread buffers, reused for every circuit in a chunk.
pub(crate) struct Scratch<T> {
    pub(crate) psi: Vec<C<T>>,
    tmp: Vec<C<T>>,
}

impl<T: Real> Scratch<T> {
    pub(crate) fn new(dim: usize) -> Self {
        let z = C::new(T::zero(), T::zero());
        Scratch { psi: vec![z; dim], tmp: vec![z; dim] }
    }
}

fn i_pow<T: Real>(k: u8) -> C<T> {
    let (z, o) = (T::zero(), T::one());
    match k % 4 {
        0 => C::new(o, z),
        1 => C::new(z, o),
        2 => C::new(-o, z),
        _ => C::new(z, -o),
    }
}

#[inline(always)]
fn sign_of<T: Real>(i: usize, zy: usize) -> T {
    if (i & zy).count_ones() & 1 == 1 {
        -T::one()
    } else {
        T::one()
    }
}

impl<T: Real> Program<T> {
    pub(crate) fn new(c: &Circuit, d_x: usize, weights: &[T]) -> Result<Self> {
        c.validate()?;
        let (need_dx, need_w) = c.min_shapes();
        if d_x < need_dx {
            return err(format!("circuit reads {need_dx} input columns, inputs have {d_x}"));
        }
        if weights.len() < need_w {
            return err(format!("circuit reads {need_w} weights, got {}", weights.len()));
        }
        let n = c.n_qubits;
        let dim = 1usize << n;
        let mask = |w: usize| 1usize << (n - 1 - w);
        let bind = |p: Param| -> Val<T> {
            match p {
                Param::Const(v) => Val::Fixed(T::of(v)),
                Param::Weight { idx, scale } => Val::Fixed(T::of(scale) * weights[idx]),
                Param::Input { col, scale } => Val::Input { col, scale: T::of(scale) },
            }
        };
        let factor = |k: K1, ps: &[Param]| -> Factor<T> {
            let mut p = [Val::Fixed(T::zero()); 3];
            for (i, &q) in ps.iter().enumerate() {
                p[i] = bind(q);
            }
            if p.iter().all(|v| matches!(v, Val::Fixed(_))) {
                let f = |v: Val<T>| if let Val::Fixed(t) = v { t } else { unreachable!() };
                Factor::Fixed(mat1(k, [f(p[0]), f(p[1]), f(p[2])]))
            } else {
                Factor::Dyn { k, p }
            }
        };

        let mut stages: Vec<Stage<T>> = vec![];
        let mut pending: Vec<Option<Fused<T>>> = (0..n).map(|_| None).collect();
        let flush = |stages: &mut Vec<Stage<T>>, pending: &mut Vec<Option<Fused<T>>>| {
            for (w, slot) in pending.iter_mut().enumerate() {
                if let Some(f) = slot.take() {
                    stages.push(Stage::U1 { s: mask(w), f });
                }
            }
        };
        let push_perm = |stages: &mut Vec<Stage<T>>, p: Vec<u32>| {
            if let Some(Stage::Perm(prev)) = stages.last_mut() {
                let q: Vec<u32> = p.iter().map(|&j| prev[j as usize]).collect();
                *prev = q.into_boxed_slice();
            } else {
                stages.push(Stage::Perm(p.into_boxed_slice()));
            }
        };

        for op in &c.ops {
            let w = &op.wires;
            let one = |k: K1| Some(factor(k, &op.gate.params()));
            let single = match &op.gate {
                Gate::PauliX => one(K1::X),
                Gate::PauliY => one(K1::Y),
                Gate::PauliZ => one(K1::Z),
                Gate::Hadamard => one(K1::H),
                Gate::S => one(K1::S),
                Gate::Sdg => one(K1::Sdg),
                Gate::T => one(K1::T),
                Gate::Tdg => one(K1::Tdg),
                Gate::SX => one(K1::SX),
                Gate::RX(_) => one(K1::RX),
                Gate::RY(_) => one(K1::RY),
                Gate::RZ(_) => one(K1::RZ),
                Gate::PhaseShift(_) => one(K1::Phase),
                Gate::Rot(..) => one(K1::Rot),
                Gate::U3(..) => one(K1::U3),
                Gate::QubitUnitary(m) if w.len() == 1 => {
                    let t = |z: num_complex::Complex<f64>| C::new(T::of(z.re), T::of(z.im));
                    Some(Factor::Fixed([t(m[0]), t(m[1]), t(m[2]), t(m[3])]))
                }
                _ => None,
            };
            if let Some(f) = single {
                pending[w[0]].get_or_insert_with(Fused::new).push(f);
                continue;
            }
            flush(&mut stages, &mut pending);
            let ctrl = |k: K1| -> Stage<T> {
                let mut f = Fused::new();
                f.push(factor(k, &op.gate.params()));
                Stage::Ctrl1 { cm: mask(w[0]), ts: mask(w[1]), f }
            };
            let prot = |word: &[(usize, Pauli)], p: Param| -> Stage<T> {
                let (flip, zy, ny) = masks(n, word);
                Stage::PRot { flip, zy, ny, theta: bind(p) }
            };
            match &op.gate {
                Gate::CNOT => {
                    let (cm, tm) = (mask(w[0]), mask(w[1]));
                    push_perm(&mut stages, (0..dim).map(|i| (if i & cm != 0 { i ^ tm } else { i }) as u32).collect());
                }
                Gate::SWAP => {
                    let (a, b) = (mask(w[0]), mask(w[1]));
                    let p = (0..dim)
                        .map(|i| {
                            let (x, y) = (i & a != 0, i & b != 0);
                            (if x != y { i ^ a ^ b } else { i }) as u32
                        })
                        .collect();
                    push_perm(&mut stages, p);
                }
                Gate::CY => stages.push(ctrl(K1::Y)),
                Gate::CZ => stages.push(ctrl(K1::Z)),
                Gate::CRX(_) => stages.push(ctrl(K1::RX)),
                Gate::CRY(_) => stages.push(ctrl(K1::RY)),
                Gate::CRZ(_) => stages.push(ctrl(K1::RZ)),
                Gate::ControlledPhaseShift(_) => stages.push(ctrl(K1::Phase)),
                Gate::IsingXX(p) => stages.push(prot(&[(w[0], Pauli::X), (w[1], Pauli::X)], *p)),
                Gate::IsingYY(p) => stages.push(prot(&[(w[0], Pauli::Y), (w[1], Pauli::Y)], *p)),
                Gate::IsingZZ(p) => stages.push(prot(&[(w[0], Pauli::Z), (w[1], Pauli::Z)], *p)),
                Gate::MultiRZ(p) => {
                    let word: Vec<_> = w.iter().map(|&q| (q, Pauli::Z)).collect();
                    stages.push(prot(&word, *p));
                }
                Gate::PauliRot(p, ps) => {
                    let word: Vec<_> = w.iter().copied().zip(ps.iter().copied()).collect();
                    stages.push(prot(&word, *p));
                }
                Gate::QubitUnitary(m) => {
                    let mut b = Box::new([C::new(T::zero(), T::zero()); 16]);
                    for (dst, z) in b.iter_mut().zip(m.iter()) {
                        *dst = C::new(T::of(z.re), T::of(z.im));
                    }
                    stages.push(Stage::Dense2 { m0: mask(w[0]), m1: mask(w[1]), m: b });
                }
                g => unreachable!("single-qubit gate {g:?} not lowered"),
            }
        }
        flush(&mut stages, &mut pending);

        let meas = c
            .measurements
            .iter()
            .map(|m| match m {
                Measurement::Expval(PauliWord(word)) => {
                    let (flip, zy, ny) = masks(n, word);
                    MeasPlan::Expval { flip, zy, ny }
                }
                Measurement::Var(PauliWord(word)) => {
                    let (flip, zy, ny) = masks(n, word);
                    MeasPlan::Var { flip, zy, ny }
                }
                Measurement::Probs(wires) => MeasPlan::Probs { map: probs_map(n, wires), width: 1 << wires.len() },
            })
            .collect();

        Ok(Program { dim, n_out: c.n_outputs(), prep: c.state_prep, stages, meas })
    }

    /// Run one circuit; the final state is left in `sc.psi`.
    #[inline]
    pub(crate) fn evolve(&self, x: &[T], sc: &mut Scratch<T>) {
        let zero = C::new(T::zero(), T::zero());
        let psi = &mut sc.psi;
        match self.prep {
            None => {
                psi.fill(zero);
                psi[0] = C::new(T::one(), T::zero());
            }
            Some(StatePrep::Real { start }) => {
                for (a, &v) in psi.iter_mut().zip(&x[start..start + self.dim]) {
                    *a = C::new(v, T::zero());
                }
            }
            Some(StatePrep::Complex { start }) => {
                for (i, a) in psi.iter_mut().enumerate() {
                    *a = C::new(x[start + 2 * i], x[start + 2 * i + 1]);
                }
            }
        }
        for st in &self.stages {
            match st {
                Stage::U1 { s, f } => {
                    let m = f.eval(x);
                    if f.diag {
                        apply_diag1(&mut sc.psi, *s, m[0], m[3]);
                    } else {
                        apply1(&mut sc.psi, *s, &m);
                    }
                }
                Stage::Ctrl1 { cm, ts, f } => {
                    let m = f.eval(x);
                    let psi = &mut sc.psi;
                    for i in 0..self.dim {
                        if i & cm != 0 && i & ts == 0 {
                            let (a, b) = (psi[i], psi[i + ts]);
                            psi[i] = m[0] * a + m[1] * b;
                            psi[i + ts] = m[2] * a + m[3] * b;
                        }
                    }
                }
                Stage::Perm(p) => {
                    for (dst, &j) in sc.tmp.iter_mut().zip(p.iter()) {
                        *dst = sc.psi[j as usize];
                    }
                    std::mem::swap(&mut sc.psi, &mut sc.tmp);
                }
                Stage::PRot { flip, zy, ny, theta } => {
                    let (s, co) = (theta.get(x) * T::of(0.5)).sin_cos();
                    apply_pauli_rot(&mut sc.psi, *flip, *zy, *ny, co, s);
                }
                Stage::Dense2 { m0, m1, m } => {
                    let psi = &mut sc.psi;
                    for i in 0..self.dim {
                        if i & (m0 | m1) == 0 {
                            let idx = [i, i | m1, i | m0, i | m0 | m1];
                            let v = idx.map(|k| psi[k]);
                            for r in 0..4 {
                                psi[idx[r]] = m[4 * r] * v[0] + m[4 * r + 1] * v[1] + m[4 * r + 2] * v[2] + m[4 * r + 3] * v[3];
                            }
                        }
                    }
                }
            }
        }
    }

    /// Write this circuit's measurement row from the final state.
    #[inline]
    pub(crate) fn measure(&self, psi: &[C<T>], out: &mut [T]) {
        let mut o = 0;
        for m in &self.meas {
            match m {
                MeasPlan::Expval { flip, zy, ny } => {
                    out[o] = pauli_expval(psi, *flip, *zy, *ny);
                    o += 1;
                }
                MeasPlan::Var { flip, zy, ny } => {
                    let e = pauli_expval(psi, *flip, *zy, *ny);
                    out[o] = T::one() - e * e; // P^2 = I for a Pauli word
                    o += 1;
                }
                MeasPlan::Probs { map, width } => {
                    let dst = &mut out[o..o + width];
                    dst.fill(T::zero());
                    for (a, &k) in psi.iter().zip(map.iter()) {
                        dst[k as usize] = dst[k as usize] + a.norm_sqr();
                    }
                    o += width;
                }
            }
        }
    }

    /// Marginal probabilities over `map` (see `probs_map`), for sampling.
    pub(crate) fn marginal(psi: &[C<T>], map: &[u32], dst: &mut [f64]) {
        dst.fill(0.0);
        for (a, &k) in psi.iter().zip(map) {
            dst[k as usize] += a.norm_sqr().as_f64();
        }
    }
}

/// Flip mask (X or Y), sign mask (Z or Y), and Y count for a Pauli word on n qubits.
fn masks(n: usize, word: &[(usize, Pauli)]) -> (usize, usize, u8) {
    let (mut flip, mut zy, mut ny) = (0, 0, 0u8);
    for &(w, p) in word {
        let b = 1usize << (n - 1 - w);
        match p {
            Pauli::I => {}
            Pauli::X => flip |= b,
            Pauli::Y => {
                flip |= b;
                zy |= b;
                ny += 1;
            }
            Pauli::Z => zy |= b,
        }
    }
    (flip, zy, ny)
}

/// For each basis index, its position among the 2^k outcomes of `wires` (first wire most significant).
pub(crate) fn probs_map(n: usize, wires: &[usize]) -> Box<[u32]> {
    let k = wires.len();
    (0..1usize << n)
        .map(|i| {
            let mut j = 0u32;
            for (pos, &w) in wires.iter().enumerate() {
                if i >> (n - 1 - w) & 1 == 1 {
                    j |= 1 << (k - 1 - pos);
                }
            }
            j
        })
        .collect()
}

#[inline(always)]
fn apply1<T: Real>(psi: &mut [C<T>], s: usize, m: &M2<T>) {
    let d = psi.len();
    let mut base = 0;
    while base < d {
        for i in base..base + s {
            let (a, b) = (psi[i], psi[i + s]);
            psi[i] = m[0] * a + m[1] * b;
            psi[i + s] = m[2] * a + m[3] * b;
        }
        base += 2 * s;
    }
}

#[inline(always)]
fn apply_diag1<T: Real>(psi: &mut [C<T>], s: usize, d0: C<T>, d1: C<T>) {
    let d = psi.len();
    let mut base = 0;
    while base < d {
        for i in base..base + s {
            psi[i] = psi[i] * d0;
            psi[i + s] = psi[i + s] * d1;
        }
        base += 2 * s;
    }
}

/// psi <- (cos(t/2) I - i sin(t/2) P) psi, with P|i> = i^ny (-1)^popcount(i & zy) |i ^ flip>.
fn apply_pauli_rot<T: Real>(psi: &mut [C<T>], flip: usize, zy: usize, ny: u8, co: T, s: T) {
    let z = T::zero();
    if flip == 0 {
        for (i, a) in psi.iter_mut().enumerate() {
            *a = *a * C::new(co, -s * sign_of::<T>(i, zy));
        }
        return;
    }
    // -i * i^ny * s
    let g = C::new(z, -s) * i_pow::<T>(ny);
    for i in 0..psi.len() {
        let j = i ^ flip;
        if i < j {
            let (a, b) = (psi[i], psi[j]);
            psi[i] = a * co + g * b * sign_of::<T>(j, zy);
            psi[j] = b * co + g * a * sign_of::<T>(i, zy);
        }
    }
}

/// <psi| P |psi> = Re( i^ny * sum_i conj(psi[i ^ flip]) (-1)^popcount(i & zy) psi[i] ).
#[inline]
fn pauli_expval<T: Real>(psi: &[C<T>], flip: usize, zy: usize, ny: u8) -> T {
    if flip == 0 {
        let mut acc = T::zero();
        for (i, a) in psi.iter().enumerate() {
            acc = acc + a.norm_sqr() * sign_of::<T>(i, zy);
        }
        return acc;
    }
    let mut acc = C::new(T::zero(), T::zero());
    for (i, a) in psi.iter().enumerate() {
        acc = acc + psi[i ^ flip].conj() * *a * sign_of::<T>(i, zy);
    }
    (i_pow::<T>(ny) * acc).re
}
