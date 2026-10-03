use rand::Rng;
use rand::rngs::StdRng;

pub const SEED: u64 = 42;
pub const SIZES: [usize; 4] = [32, 1_024, 32_768, 1_048_576];
pub const CASES: [(&str, f64); 2] = [("random", 0.0), ("large_offset", 1e12)];

/// Generates bounded random observations, optionally shifted by `offset`.
///
/// Call outside the timed loop. Reuse the RNG to generate successive series.
pub fn generate_samples(n: usize, offset: f64, rng: &mut StdRng) -> Vec<f64> {
    (0..n)
        .map(|_| rng.random_range(-5.0..5.0) + offset)
        .collect()
}
