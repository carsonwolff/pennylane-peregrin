//! Lane-parallel execution: `W` circuits of the batch at once.
//!
//! The scalar path in `program.rs` runs one circuit at a time on a state of 4 to 256
//! amplitudes, which is too short for SIMD. Here the state of a tile of `W` circuits is stored
//! struct-of-arrays, one `[T; W]` row per amplitude (real and imaginary parts separately), so
//! every gate is a loop over lanes that the compiler vectorizes.
//!
//! Two more things are settled once per call, when the tile program is built:
//! - Permutation stages (runs of CNOT and SWAP) move no data. The program tracks which row
//!   holds each logical amplitude and every later stage works on precomputed row lists.
//! - While the state is known to be real (real or no state prep, then only real gates such as
//!   H and RY), the imaginary half of the work is skipped.

use crate::circuit::StatePrep;
use crate::gates::{C, M2};
use crate::program::{sign_of, Fused, MeasPlan, Program, Stage, Val};
use crate::Real;

/// Circuits per tile.
pub(crate) const W: usize = 8;
/// Largest register the tile path takes; beyond it the scalar path is used.
pub(crate) const MAX_TILE_QUBITS: usize = 10;

type L<T> = [T; W];

/// A 2x2 matrix per lane, entries [m00, m01, m10, m11].
#[derive(Clone, Copy)]
struct LaneMat<T> {
    re: [L<T>; 4],
    im: [L<T>; 4],
}

impl<T: Real> LaneMat<T> {
    fn zero() -> Self {
        LaneMat { re: [[T::zero(); W]; 4], im: [[T::zero(); W]; 4] }
    }

    fn broadcast(m: &M2<T>) -> Self {
        let mut s = Self::zero();
        for k in 0..4 {
            s.re[k] = [m[k].re; W];
            s.im[k] = [m[k].im; W];
        }
        s
    }

    #[inline(always)]
    fn set_lane(&mut self, l: usize, m: &M2<T>) {
        for k in 0..4 {
            self.re[k][l] = m[k].re;
            self.im[k][l] = m[k].im;
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Diag,
    /// Real matrix on a state known to be real: only the real parts are touched.
    RealOnReal,
    /// Real matrix on a complex state.
    Real,
    General,
}

enum Mat<T> {
    Static(Box<LaneMat<T>>),
    Dyn(Fused<T>),
}

enum TStage<T> {
    /// A 2x2 unitary on each row pair (i, j); covers single-qubit and controlled gates.
    Pairs { pairs: Box<[(u32, u32)]>, mat: Mat<T>, kind: Kind },
    /// Diagonal Pauli rotation: row r times (cos, -sin * sign).
    PRotDiag { rows: Box<[(u32, T)]>, theta: Val<T> },
    /// Off-diagonal Pauli rotation on row pairs with the signs of both members.
    PRotOff { pairs: Box<[(u32, u32, T, T)]>, ny: u8, theta: Val<T> },
    /// Constant 4x4 unitary on row quads.
    Dense2 { quads: Box<[[u32; 4]]>, m: Box<[C<T>; 16]> },
}

enum TMeas<T> {
    /// sum over rows of coef * |psi|^2 (Z-type Pauli words).
    Diag { coef: Box<[T]>, var: bool },
    /// sum over row pairs of coef * Re or Im of conj(psi_j) psi_i (words with X or Y).
    Off { pairs: Box<[(u32, u32, T)]>, use_im: bool, var: bool },
    Probs { slot: Box<[u32]>, width: usize },
}

pub(crate) struct TileProgram<'p, T> {
    prog: &'p Program<T>,
    stages: Vec<TStage<T>>,
    meas: Vec<TMeas<T>>,
    /// The state is real when the circuit ends, so measurements can ignore imaginary parts.
    real_end: bool,
    needs_probs: bool,
}

impl<'p, T: Real> TileProgram<'p, T> {
    pub(crate) fn new(prog: &'p Program<T>) -> Self {
        let dim = prog.dim;
        let mut loc: Vec<u32> = (0..dim as u32).collect();
        let mut real = !matches!(prog.prep, Some(StatePrep::Complex { .. }));
        let mut stages = vec![];
        let pairs_where = |loc: &[u32], keep: &dyn Fn(usize) -> bool, partner: usize| -> Box<[(u32, u32)]> {
            (0..dim).filter(|&i| keep(i)).map(|i| (loc[i], loc[i | partner])).collect()
        };
        let push_pairs = |stages: &mut Vec<TStage<T>>, real: &mut bool, pairs: Box<[(u32, u32)]>, f: &Fused<T>| {
            let kind = if f.diag {
                Kind::Diag
            } else if f.is_real() {
                if *real {
                    Kind::RealOnReal
                } else {
                    Kind::Real
                }
            } else {
                Kind::General
            };
            if !f.is_real() {
                *real = false;
            }
            let mat = match f.fixed() {
                Some(m) => Mat::Static(Box::new(LaneMat::broadcast(&m))),
                None => Mat::Dyn(f.clone()),
            };
            stages.push(TStage::Pairs { pairs, mat, kind });
        };
        for st in &prog.stages {
            match st {
                Stage::U1 { s, f } => {
                    let s = *s;
                    let pairs = pairs_where(&loc, &|i| i & s == 0, s);
                    push_pairs(&mut stages, &mut real, pairs, f);
                }
                Stage::Ctrl1 { cm, ts, f } => {
                    let (cm, ts) = (*cm, *ts);
                    let pairs = pairs_where(&loc, &|i| i & cm != 0 && i & ts == 0, ts);
                    push_pairs(&mut stages, &mut real, pairs, f);
                }
                Stage::Perm(p) => loc = p.iter().map(|&j| loc[j as usize]).collect(),
                Stage::PRot { flip, zy, ny, theta } => {
                    real = false;
                    if *flip == 0 {
                        let rows = (0..dim).map(|i| (loc[i], sign_of::<T>(i, *zy))).collect();
                        stages.push(TStage::PRotDiag { rows, theta: *theta });
                    } else {
                        let pairs = (0..dim)
                            .filter(|&i| i < i ^ flip)
                            .map(|i| {
                                let j = i ^ flip;
                                (loc[i], loc[j], sign_of::<T>(i, *zy), sign_of::<T>(j, *zy))
                            })
                            .collect();
                        stages.push(TStage::PRotOff { pairs, ny: *ny, theta: *theta });
                    }
                }
                Stage::Dense2 { m0, m1, m } => {
                    if m.iter().any(|z| z.im != T::zero()) {
                        real = false;
                    }
                    let (m0, m1) = (*m0, *m1);
                    let quads = (0..dim)
                        .filter(|&i| i & (m0 | m1) == 0)
                        .map(|i| [loc[i], loc[i | m1], loc[i | m0], loc[i | m0 | m1]])
                        .collect();
                    stages.push(TStage::Dense2 { quads, m: m.clone() });
                }
            }
        }

        let mut needs_probs = false;
        let meas = prog
            .meas
            .iter()
            .map(|m| {
                let (flip, zy, ny, var) = match m {
                    MeasPlan::Expval { flip, zy, ny } => (*flip, *zy, *ny, false),
                    MeasPlan::Var { flip, zy, ny } => (*flip, *zy, *ny, true),
                    MeasPlan::Probs { map, width } => {
                        needs_probs = true;
                        let mut slot = vec![0u32; dim];
                        for i in 0..dim {
                            slot[loc[i] as usize] = map[i];
                        }
                        return TMeas::Probs { slot: slot.into(), width: *width };
                    }
                };
                if flip == 0 {
                    needs_probs = true;
                    let mut coef = vec![T::zero(); dim];
                    for i in 0..dim {
                        coef[loc[i] as usize] = sign_of::<T>(i, zy);
                    }
                    return TMeas::Diag { coef: coef.into(), var };
                }
                // <P> = Re(i^ny sum_i conj(psi[i^f]) s(i) psi[i]). Pairing i with j = i^f, and
                // using s(j) = (-1)^ny s(i), each pair contributes 2 s(i) Re(conj(psi_j) psi_i)
                // for even ny, and 2i s(i) Im(conj(psi_j) psi_i) for odd ny.
                let (use_im, f) = match ny % 4 {
                    0 => (false, 2.0),
                    1 => (true, -2.0),
                    2 => (false, -2.0),
                    _ => (true, 2.0),
                };
                let pairs = (0..dim)
                    .filter(|&i| i < i ^ flip)
                    .map(|i| (loc[i], loc[i ^ flip], T::of(f) * sign_of::<T>(i, zy)))
                    .collect();
                TMeas::Off { pairs, use_im, var }
            })
            .collect();

        TileProgram { prog, stages, meas, real_end: real, needs_probs }
    }

    /// Run every circuit in `xs` (row-major, `stride` values per row) into `out`
    /// (row-major, `n_out` per row).
    pub(crate) fn run(&self, xs: &[T], stride: usize, out: &mut [T]) {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                // SAFETY: the features were just detected on this CPU.
                return unsafe { self.run_avx2(xs, stride, out) };
            }
        }
        self.run_any(xs, stride, out)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn run_avx2(&self, xs: &[T], stride: usize, out: &mut [T]) {
        self.run_any(xs, stride, out)
    }

    #[inline(always)]
    fn run_any(&self, xs: &[T], stride: usize, out: &mut [T]) {
        let dim = self.prog.dim;
        let n_out = self.prog.n_out;
        let rows = out.len() / n_out;
        let mut re = vec![[T::zero(); W]; dim];
        let mut im = vec![[T::zero(); W]; dim];
        let mut p = vec![[T::zero(); W]; if self.needs_probs { dim } else { 0 }];
        let mut lm = LaneMat::zero();
        let mut r0 = 0;
        while r0 < rows {
            let valid = (rows - r0).min(W);
            // Lanes past the end of the batch repeat the last valid row; their results are dropped.
            let row = |l: usize| &xs[(r0 + l.min(valid - 1)) * stride..][..stride];
            self.prepare(&mut re, &mut im, &row);
            for st in &self.stages {
                match st {
                    TStage::Pairs { pairs, mat, kind } => {
                        let m: &LaneMat<T> = match mat {
                            Mat::Static(m) => m,
                            Mat::Dyn(f) => {
                                for l in 0..W {
                                    lm.set_lane(l, &f.eval(row(l)));
                                }
                                &lm
                            }
                        };
                        match kind {
                            Kind::Diag => k_diag(&mut re, &mut im, pairs, m),
                            Kind::RealOnReal => k_real_on_real(&mut re, pairs, m),
                            Kind::Real => k_real(&mut re, &mut im, pairs, m),
                            Kind::General => k_general(&mut re, &mut im, pairs, m),
                        }
                    }
                    TStage::PRotDiag { rows: rs, theta } => {
                        let (s, c) = half_angle(theta, &row);
                        for &(r, sign) in rs.iter() {
                            let (a, b) = (&mut re[r as usize], &mut im[r as usize]);
                            for l in 0..W {
                                let (ar, ai, si) = (a[l], b[l], -s[l] * sign);
                                a[l] = ar * c[l] - ai * si;
                                b[l] = ar * si + ai * c[l];
                            }
                        }
                    }
                    TStage::PRotOff { pairs, ny, theta } => {
                        let (s, c) = half_angle(theta, &row);
                        // g = -i * i^ny * sin
                        let (gr, gi): (L<T>, L<T>) = match ny % 4 {
                            0 => ([T::zero(); W], s.map(|v| -v)),
                            1 => (s, [T::zero(); W]),
                            2 => ([T::zero(); W], s),
                            _ => (s.map(|v| -v), [T::zero(); W]),
                        };
                        for &(i, j, si, sj) in pairs.iter() {
                            let (i, j) = (i as usize, j as usize);
                            let (ar, ai, br, bi) = (re[i], im[i], re[j], im[j]);
                            for l in 0..W {
                                // psi_i = c a + g sj b; psi_j = c b + g si a
                                re[i][l] = c[l] * ar[l] + sj * (gr[l] * br[l] - gi[l] * bi[l]);
                                im[i][l] = c[l] * ai[l] + sj * (gr[l] * bi[l] + gi[l] * br[l]);
                                re[j][l] = c[l] * br[l] + si * (gr[l] * ar[l] - gi[l] * ai[l]);
                                im[j][l] = c[l] * bi[l] + si * (gr[l] * ai[l] + gi[l] * ar[l]);
                            }
                        }
                    }
                    TStage::Dense2 { quads, m } => {
                        for q in quads.iter() {
                            let vr = q.map(|k| re[k as usize]);
                            let vi = q.map(|k| im[k as usize]);
                            for (r, &dst) in q.iter().enumerate() {
                                let (mut nr, mut ni) = ([T::zero(); W], [T::zero(); W]);
                                for c in 0..4 {
                                    let z = m[4 * r + c];
                                    for l in 0..W {
                                        nr[l] = nr[l] + z.re * vr[c][l] - z.im * vi[c][l];
                                        ni[l] = ni[l] + z.re * vi[c][l] + z.im * vr[c][l];
                                    }
                                }
                                re[dst as usize] = nr;
                                im[dst as usize] = ni;
                            }
                        }
                    }
                }
            }
            self.measure(&re, &im, &mut p, &mut out[r0 * n_out..(r0 + valid) * n_out], valid);
            r0 += W;
        }
    }

    #[inline(always)]
    fn prepare<'x>(&self, re: &mut [L<T>], im: &mut [L<T>], row: &dyn Fn(usize) -> &'x [T]) {
        let dim = self.prog.dim;
        for r in im.iter_mut() {
            *r = [T::zero(); W];
        }
        match self.prog.prep {
            None => {
                for r in re.iter_mut() {
                    *r = [T::zero(); W];
                }
                re[0] = [T::one(); W];
            }
            Some(StatePrep::Real { start }) => {
                for l in 0..W {
                    let x = &row(l)[start..start + dim];
                    for (r, &v) in re.iter_mut().zip(x) {
                        r[l] = v;
                    }
                }
            }
            Some(StatePrep::Complex { start }) => {
                for l in 0..W {
                    let x = &row(l)[start..start + 2 * dim];
                    for i in 0..dim {
                        re[i][l] = x[2 * i];
                        im[i][l] = x[2 * i + 1];
                    }
                }
            }
        }
    }

    #[inline(always)]
    fn measure(&self, re: &[L<T>], im: &[L<T>], p: &mut [L<T>], out: &mut [T], valid: usize) {
        let n_out = self.prog.n_out;
        if self.needs_probs {
            for ((pr, a), b) in p.iter_mut().zip(re).zip(im) {
                for l in 0..W {
                    pr[l] = if self.real_end { a[l] * a[l] } else { a[l] * a[l] + b[l] * b[l] };
                }
            }
        }
        let mut o = 0;
        let mut put = |o: usize, acc: &L<T>, var: bool| {
            for l in 0..valid {
                out[l * n_out + o] = if var { T::one() - acc[l] * acc[l] } else { acc[l] };
            }
        };
        for m in &self.meas {
            match m {
                TMeas::Diag { coef, var } => {
                    let mut acc = [T::zero(); W];
                    for (pr, &c) in p.iter().zip(coef.iter()) {
                        for l in 0..W {
                            acc[l] = acc[l] + c * pr[l];
                        }
                    }
                    put(o, &acc, *var);
                    o += 1;
                }
                TMeas::Off { pairs, use_im, var } => {
                    let mut acc = [T::zero(); W];
                    for &(i, j, c) in pairs.iter() {
                        let (ar, ai, br, bi) = (&re[i as usize], &im[i as usize], &re[j as usize], &im[j as usize]);
                        if self.real_end && !use_im {
                            for l in 0..W {
                                acc[l] = acc[l] + c * (br[l] * ar[l]);
                            }
                        } else if *use_im {
                            for l in 0..W {
                                acc[l] = acc[l] + c * (br[l] * ai[l] - bi[l] * ar[l]);
                            }
                        } else {
                            for l in 0..W {
                                acc[l] = acc[l] + c * (br[l] * ar[l] + bi[l] * ai[l]);
                            }
                        }
                    }
                    put(o, &acc, *var);
                    o += 1;
                }
                TMeas::Probs { slot, width } => {
                    let mut acc = vec![[T::zero(); W]; *width];
                    for (pr, &k) in p.iter().zip(slot.iter()) {
                        for l in 0..W {
                            acc[k as usize][l] = acc[k as usize][l] + pr[l];
                        }
                    }
                    for (k, a) in acc.iter().enumerate() {
                        put(o + k, a, false);
                    }
                    o += width;
                }
            }
        }
    }
}

#[inline(always)]
fn half_angle<'x, T: Real>(theta: &Val<T>, row: &dyn Fn(usize) -> &'x [T]) -> (L<T>, L<T>) {
    let (mut s, mut c) = ([T::zero(); W], [T::zero(); W]);
    for l in 0..W {
        let (a, b) = (theta.get(row(l)) * T::of(0.5)).sin_cos();
        s[l] = a;
        c[l] = b;
    }
    (s, c)
}

#[inline(always)]
fn k_general<T: Real>(re: &mut [L<T>], im: &mut [L<T>], pairs: &[(u32, u32)], m: &LaneMat<T>) {
    for &(i, j) in pairs {
        let (i, j) = (i as usize, j as usize);
        let (ar, ai, br, bi) = (re[i], im[i], re[j], im[j]);
        let (mut nar, mut nai, mut nbr, mut nbi) = ([T::zero(); W], [T::zero(); W], [T::zero(); W], [T::zero(); W]);
        for l in 0..W {
            let (m0r, m0i, m1r, m1i) = (m.re[0][l], m.im[0][l], m.re[1][l], m.im[1][l]);
            let (m2r, m2i, m3r, m3i) = (m.re[2][l], m.im[2][l], m.re[3][l], m.im[3][l]);
            nar[l] = m0r * ar[l] - m0i * ai[l] + m1r * br[l] - m1i * bi[l];
            nai[l] = m0r * ai[l] + m0i * ar[l] + m1r * bi[l] + m1i * br[l];
            nbr[l] = m2r * ar[l] - m2i * ai[l] + m3r * br[l] - m3i * bi[l];
            nbi[l] = m2r * ai[l] + m2i * ar[l] + m3r * bi[l] + m3i * br[l];
        }
        re[i] = nar;
        im[i] = nai;
        re[j] = nbr;
        im[j] = nbi;
    }
}

#[inline(always)]
fn k_real<T: Real>(re: &mut [L<T>], im: &mut [L<T>], pairs: &[(u32, u32)], m: &LaneMat<T>) {
    k_real_on_real(re, pairs, m);
    k_real_on_real(im, pairs, m);
}

#[inline(always)]
fn k_real_on_real<T: Real>(re: &mut [L<T>], pairs: &[(u32, u32)], m: &LaneMat<T>) {
    for &(i, j) in pairs {
        let (i, j) = (i as usize, j as usize);
        let (a, b) = (re[i], re[j]);
        for l in 0..W {
            re[i][l] = m.re[0][l] * a[l] + m.re[1][l] * b[l];
            re[j][l] = m.re[2][l] * a[l] + m.re[3][l] * b[l];
        }
    }
}

#[inline(always)]
fn k_diag<T: Real>(re: &mut [L<T>], im: &mut [L<T>], pairs: &[(u32, u32)], m: &LaneMat<T>) {
    for &(i, j) in pairs {
        for (r, k) in [(i as usize, 0), (j as usize, 3)] {
            let (a, b) = (re[r], im[r]);
            for l in 0..W {
                let (dr, di) = (m.re[k][l], m.im[k][l]);
                re[r][l] = dr * a[l] - di * b[l];
                im[r][l] = dr * b[l] + di * a[l];
            }
        }
    }
}
