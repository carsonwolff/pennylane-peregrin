//! Single-qubit gate matrices (PennyLane conventions) and 2x2 helpers.

use num_complex::Complex;

use crate::Real;

pub(crate) type C<T> = Complex<T>;
/// Row-major 2x2 matrix: [m00, m01, m10, m11].
pub(crate) type M2<T> = [C<T>; 4];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum K1 {
    X,
    Y,
    Z,
    H,
    S,
    Sdg,
    T,
    Tdg,
    SX,
    RX,
    RY,
    RZ,
    Phase,
    Rot,
    U3,
}

impl K1 {
    /// True when the matrix is diagonal for every parameter value.
    pub(crate) fn is_diagonal(self) -> bool {
        matches!(self, K1::Z | K1::S | K1::Sdg | K1::T | K1::Tdg | K1::RZ | K1::Phase)
    }
}

#[inline(always)]
fn c<T: Real>(re: T, im: T) -> C<T> {
    Complex::new(re, im)
}

/// e^{i a}
#[inline(always)]
fn cis<T: Real>(a: T) -> C<T> {
    let (s, co) = a.sin_cos();
    c(co, s)
}

#[inline]
pub(crate) fn mat1<T: Real>(k: K1, p: [T; 3]) -> M2<T> {
    let z = T::zero();
    let o = T::one();
    let half = T::of(0.5);
    match k {
        K1::X => [c(z, z), c(o, z), c(o, z), c(z, z)],
        K1::Y => [c(z, z), c(z, -o), c(z, o), c(z, z)],
        K1::Z => [c(o, z), c(z, z), c(z, z), c(-o, z)],
        K1::H => {
            let h = T::FRAC_1_SQRT_2();
            [c(h, z), c(h, z), c(h, z), c(-h, z)]
        }
        K1::S => [c(o, z), c(z, z), c(z, z), c(z, o)],
        K1::Sdg => [c(o, z), c(z, z), c(z, z), c(z, -o)],
        K1::T => [c(o, z), c(z, z), c(z, z), cis(T::FRAC_PI_4())],
        K1::Tdg => [c(o, z), c(z, z), c(z, z), cis(-T::FRAC_PI_4())],
        K1::SX => [c(half, half), c(half, -half), c(half, -half), c(half, half)],
        K1::RX => {
            let (s, co) = (p[0] * half).sin_cos();
            [c(co, z), c(z, -s), c(z, -s), c(co, z)]
        }
        K1::RY => {
            let (s, co) = (p[0] * half).sin_cos();
            [c(co, z), c(-s, z), c(s, z), c(co, z)]
        }
        K1::RZ => {
            let e = cis(-p[0] * half);
            [e, c(z, z), c(z, z), e.conj()]
        }
        K1::Phase => [c(o, z), c(z, z), c(z, z), cis(p[0])],
        K1::Rot => {
            let (phi, theta, omega) = (p[0], p[1], p[2]);
            let (s, co) = (theta * half).sin_cos();
            let a = cis(-(phi + omega) * half);
            let b = cis((phi - omega) * half);
            [a * co, -b * s, b.conj() * s, a.conj() * co]
        }
        K1::U3 => {
            let (theta, phi, delta) = (p[0], p[1], p[2]);
            let (s, co) = (theta * half).sin_cos();
            [c(co, z), -cis(delta) * s, cis(phi) * s, cis(phi + delta) * co]
        }
    }
}

/// a * b (apply b first, then a).
#[inline(always)]
pub(crate) fn mul2<T: Real>(a: &M2<T>, b: &M2<T>) -> M2<T> {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
    ]
}

pub(crate) fn is_diag2<T: Real>(m: &M2<T>) -> bool {
    m[1] == C::new(T::zero(), T::zero()) && m[2] == C::new(T::zero(), T::zero())
}
