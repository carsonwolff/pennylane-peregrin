/// SplitMix64: tiny, fast, and good enough for drawing measurement samples.
/// Each circuit gets its own stream derived from (seed, circuit index), so samples do not
/// depend on how the batch is split across threads.
pub(crate) struct SplitMix64(u64);

impl SplitMix64 {
    pub(crate) fn for_circuit(seed: u64, index: u64) -> Self {
        let mut s = SplitMix64(seed ^ index.wrapping_mul(0xD1B5_4A32_D192_ED03));
        s.next_u64();
        s
    }

    #[inline]
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1) with 53 random bits.
    #[inline]
    pub(crate) fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}
