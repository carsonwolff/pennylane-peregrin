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
use crate::gates::{mat1, C, K1, M2};
use crate::program::{sign_of, Factor, Fused, MeasPlan, Program, Stage, Val};
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
    Diag {
        coef: Box<[T]>,
        var: bool,
    },
    /// sum over row pairs of coef * Re or Im of conj(psi_j) psi_i (words with X or Y).
    Off {
        pairs: Box<[(u32, u32, T)]>,
        use_im: bool,
        var: bool,
    },
    Probs {
        slot: Box<[u32]>,
        width: usize,
    },
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
                                eval_lanes(f, &row, &mut lm);
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

/// sin and cos of theta/2 per lane.
#[inline(always)]
fn half_angle<'x, T: Real>(theta: &Val<T>, row: &dyn Fn(usize) -> &'x [T]) -> (L<T>, L<T>) {
    let mut h = [T::zero(); W];
    for (l, v) in h.iter_mut().enumerate() {
        *v = theta.get(row(l)) * T::of(0.5);
    }
    sin_cos_lanes(&h)
}

// sin/cos on lanes: reduce by multiples of pi/2 (three-part constant, from fdlibm), then the
// Cephes minimax polynomials on [-pi/4, pi/4]. About 1 ulp in f64. Written without branches
// or integer conversions so the loop vectorizes; lanes with |x| >= 1e5 (or NaN) are redone
// with the library functions. The constants are the published digits, kept as written.
#[allow(clippy::excessive_precision)]
mod trig_consts {
    pub(super) const PIO2_1: f64 = 1.570_796_326_734_125_614_17e+00;
    pub(super) const PIO2_2: f64 = 6.077_100_506_303_965_976_60e-11;
    pub(super) const PIO2_3: f64 = 2.022_266_248_711_166_455_80e-21;
    pub(super) const SIN_P: [f64; 6] = [
        1.589_623_015_765_465_680_60e-10,
        -2.505_074_776_285_780_728_66e-8,
        2.755_731_362_138_572_452_13e-6,
        -1.984_126_982_958_953_859_96e-4,
        8.333_333_333_322_118_588_78e-3,
        -1.666_666_666_666_663_072_95e-1,
    ];
    pub(super) const COS_P: [f64; 6] = [
        -1.135_853_652_138_768_173_00e-11,
        2.087_570_084_197_473_167_78e-9,
        -2.755_731_417_929_673_881_12e-7,
        2.480_158_728_885_170_453_48e-5,
        -1.388_888_888_887_305_641_16e-3,
        4.166_666_666_666_659_292_18e-2,
    ];
}
use trig_consts::{COS_P, PIO2_1, PIO2_2, PIO2_3, SIN_P};

#[inline(always)]
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub(crate) fn sin_cos_lanes<T: Real>(x: &L<T>) -> (L<T>, L<T>) {
    const MAGIC: f64 = 6_755_399_441_055_744.0; // 1.5 * 2^52: adding it rounds to an integer
    let (mut s, mut c) = ([T::zero(); W], [T::zero(); W]);
    let mut big = false;
    for l in 0..W {
        let v = x[l].as_f64();
        big |= !(v.abs() < 1e5); // also true for NaN; one compare, so the loop stays vectorized
        let kf = v * std::f64::consts::FRAC_2_PI + MAGIC;
        let q = kf.to_bits(); // low bits hold k mod 2^32
        let k = kf - MAGIC;
        let r = ((v - k * PIO2_1) - k * PIO2_2) - k * PIO2_3;
        let z = r * r;
        let ps = ((((SIN_P[0] * z + SIN_P[1]) * z + SIN_P[2]) * z + SIN_P[3]) * z + SIN_P[4]) * z + SIN_P[5];
        let pc = ((((COS_P[0] * z + COS_P[1]) * z + COS_P[2]) * z + COS_P[3]) * z + COS_P[4]) * z + COS_P[5];
        let sr = r + r * z * ps;
        let cr = 1.0 - 0.5 * z + z * z * pc;
        let (a, b) = if q & 1 == 1 { (cr, sr) } else { (sr, cr) };
        let sa = if q & 2 != 0 { -a } else { a };
        let cb = if q.wrapping_add(1) & 2 != 0 { -b } else { b };
        s[l] = T::of(sa);
        c[l] = T::of(cb);
    }
    if big {
        for l in 0..W {
            if !(x[l].as_f64().abs() < 1e5) {
                let (a, b) = x[l].sin_cos();
                s[l] = a;
                c[l] = b;
            }
        }
    }
    (s, c)
}

/// One single-qubit gate matrix per lane, for the gates that take parameters.
#[inline(always)]
fn mat1_lanes<T: Real>(k: K1, p: &[L<T>; 3], m: &mut LaneMat<T>) {
    let z = [T::zero(); W];
    let o = [T::one(); W];
    let half = T::of(0.5);
    let scaled = |a: &L<T>, f: T| a.map(|v| v * f);
    let neg = |a: &L<T>| a.map(|v| -v);
    match k {
        K1::RX => {
            let (s, c) = sin_cos_lanes(&scaled(&p[0], half));
            m.re = [c, z, z, c];
            m.im = [z, neg(&s), neg(&s), z];
        }
        K1::RY => {
            let (s, c) = sin_cos_lanes(&scaled(&p[0], half));
            m.re = [c, neg(&s), s, c];
            m.im = [z; 4];
        }
        K1::RZ => {
            let (s, c) = sin_cos_lanes(&scaled(&p[0], half));
            m.re = [c, z, z, c];
            m.im = [neg(&s), z, z, s];
        }
        K1::Phase => {
            let (s, c) = sin_cos_lanes(&p[0]);
            m.re = [o, z, z, c];
            m.im = [z, z, z, s];
        }
        _ => {
            for l in 0..W {
                m.set_lane(l, &mat1(k, [p[0][l], p[1][l], p[2][l]]));
            }
        }
    }
}

/// g * u per lane (apply u first).
#[inline(always)]
fn mul_lanes<T: Real>(g: &LaneMat<T>, u: &LaneMat<T>) -> LaneMat<T> {
    let mut out = LaneMat::zero();
    for (r, c) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
        let (a, b) = (2 * r, 2 * r + 1); // g[r][0], g[r][1]
        let (x, y) = (c, 2 + c); // u[0][c], u[1][c]
        for l in 0..W {
            out.re[2 * r + c][l] =
                g.re[a][l] * u.re[x][l] - g.im[a][l] * u.im[x][l] + g.re[b][l] * u.re[y][l] - g.im[b][l] * u.im[y][l];
            out.im[2 * r + c][l] =
                g.re[a][l] * u.im[x][l] + g.im[a][l] * u.re[x][l] + g.re[b][l] * u.im[y][l] + g.im[b][l] * u.re[y][l];
        }
    }
    out
}

/// The fused matrix of every lane: factors evaluated and multiplied lane-wise.
#[inline(always)]
fn eval_lanes<'x, T: Real>(f: &Fused<T>, row: &dyn Fn(usize) -> &'x [T], out: &mut LaneMat<T>) {
    let mut g = LaneMat::zero();
    for (n, fac) in f.factors.iter().enumerate() {
        match fac {
            Factor::Fixed(m) => g = LaneMat::broadcast(m),
            Factor::Dyn { k, p } => {
                let mut pl = [[T::zero(); W]; 3];
                for (dst, v) in pl.iter_mut().zip(p) {
                    for (l, d) in dst.iter_mut().enumerate() {
                        *d = v.get(row(l));
                    }
                }
                mat1_lanes(*k, &pl, &mut g);
            }
        }
        *out = if n == 0 { g } else { mul_lanes(&g, out) };
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lane_sin_cos_matches_std() {
        let mut worst = 0.0f64;
        let mut x = -2000.0f64;
        while x < 2000.0 {
            let v: L<f64> = std::array::from_fn(|l| x + l as f64 * 0.013 + 1e-3 * (x * 7.1).sin());
            let (s, c) = sin_cos_lanes(&v);
            for l in 0..W {
                worst = worst.max((s[l] - v[l].sin()).abs()).max((c[l] - v[l].cos()).abs());
            }
            x += 0.0917;
        }
        for &v in &[0.0, -0.0, 1e-300, 1e5, -3e7, 1e300, f64::NAN, f64::INFINITY, std::f64::consts::FRAC_PI_4] {
            let (s, c) = sin_cos_lanes(&[v; W]);
            let (es, ec) = v.sin_cos();
            assert!((s[0] - es).abs() < 1e-15 || (s[0].is_nan() && es.is_nan()), "sin {v}");
            assert!((c[0] - ec).abs() < 1e-15 || (c[0].is_nan() && ec.is_nan()), "cos {v}");
        }
        assert!(worst < 4e-16, "worst {worst:e}");
    }
}
