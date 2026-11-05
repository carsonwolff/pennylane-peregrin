use std::fmt::Debug;

use num_traits::{Float, FloatConst};

/// Floating point type a simulation runs in: `f32` or `f64`.
pub trait Real: Float + FloatConst + Send + Sync + Debug + Default + 'static {
    fn of(x: f64) -> Self;
    fn as_f64(self) -> f64;
}

impl Real for f64 {
    #[inline(always)]
    fn of(x: f64) -> Self {
        x
    }
    #[inline(always)]
    fn as_f64(self) -> f64 {
        self
    }
}

impl Real for f32 {
    #[inline(always)]
    fn of(x: f64) -> Self {
        x as f32
    }
    #[inline(always)]
    fn as_f64(self) -> f64 {
        self as f64
    }
}
