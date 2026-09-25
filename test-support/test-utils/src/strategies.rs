//! Proptest strategies for bounded numerical inputs.
//!
//! Provides finite values, samples, constant and nonconstant samples, and
//! scaling factors for numerical property tests.
//!
//! Most strategies use moderate magnitudes so that tests can exercise
//! mathematical properties without unrelated overflow or extreme-range effects.
//! Powers-of-two scales help limit additional rounding during scaling
//! transformations.
//!
//! These strategies do not cover every `f64` value. Test NaN, infinities,
//! empty samples, and extreme magnitudes explicitly where the API requires
//! defined behavior for them.

use proptest::prelude::prop;
use proptest::strategy::Strategy;

/// Maximum number of observations generated for a sample.
pub const MAX_SAMPLE_LEN: usize = 64;

/// Generates a non-empty sample of bounded finite `f64` values.
///
/// Observations are generated in the range `[-10_000, 10_000)`, avoiding
/// non-finite values and extreme magnitudes that could introduce unrelated
/// overflow or precision issues into property tests.
///
/// The generated sample contains between `min_len` and [`MAX_SAMPLE_LEN`]
/// observations, inclusive.
///
/// # Panics
///
/// Panics unless `min_len` is between one and [`MAX_SAMPLE_LEN`], inclusive.
pub fn finite_sample(min_len: usize) -> impl Strategy<Value = Vec<f64>> {
    assert!(
        (1..=MAX_SAMPLE_LEN).contains(&min_len),
        "minimum length must be between one and {MAX_SAMPLE_LEN}"
    );

    prop::collection::vec(-10_000.0..10_000.0, min_len..=MAX_SAMPLE_LEN)
}

/// Generates a nonconstant sample of bounded finite `f64` values.
///
/// Observations are generated in the range `[-10_000, 10_000)`, avoiding
/// non-finite values and extreme magnitudes that could introduce unrelated
/// overflow or precision issues into property tests.
///
/// The generated sample contains at least two distinct observations and between
/// `min_len` and [`MAX_SAMPLE_LEN`] observations, inclusive.
///
/// # Panics
///
/// Panics unless `min_len` is between two and [`MAX_SAMPLE_LEN`], inclusive.
#[expect(clippy::float_cmp, reason = "detect exactly constant observations")]
pub fn finite_nonconstant_sample(min_len: usize) -> impl Strategy<Value = Vec<f64>> {
    assert!(
        (2..=MAX_SAMPLE_LEN).contains(&min_len),
        "minimum length must be between two and {MAX_SAMPLE_LEN}",
    );

    finite_sample(min_len).prop_map(|mut xs| {
        if xs.iter().all(|&x| x == xs[0]) {
            xs[1] = if xs[0] == 0.0 { 1.0 } else { 0.0 };
        }

        xs
    })
}

/// Generates a non-empty, constant sample of bounded finite `f64` values.
///
/// The repeated observation is generated in the range `[-10_000, 10_000)`,
/// avoiding non-finite values and extreme magnitudes that could introduce
/// unrelated overflow or precision issues into property tests.
///
/// The generated sample contains between `min_len` and [`MAX_SAMPLE_LEN`]
/// observations, inclusive.
///
/// # Panics
///
/// Panics unless `min_len` is between one and [`MAX_SAMPLE_LEN`], inclusive.
pub fn constant_sample(min_len: usize) -> impl Strategy<Value = Vec<f64>> {
    assert!(
        (1..=MAX_SAMPLE_LEN).contains(&min_len),
        "minimum length must be between one and {MAX_SAMPLE_LEN}",
    );

    (finite_value(), min_len..=MAX_SAMPLE_LEN).prop_map(|(value, len)| vec![value; len])
}

/// Generates a nonconstant sample and an exactly representable translation.
///
/// Sample values and offsets are integers in `[-10_000, 10_000)`, converted
/// to `f64`. Adding the offset preserves all differences between observations.
///
/// # Panics
///
/// Panics unless `min_len` is between two and [`MAX_SAMPLE_LEN`], inclusive.
pub fn exact_translation_case(min_len: usize) -> impl Strategy<Value = (Vec<f64>, f64)> {
    assert!(
        (2..=MAX_SAMPLE_LEN).contains(&min_len),
        "minimum length must be between two and MAX_SAMPLE_LEN"
    );

    let sample_strat = prop::collection::vec(-10_000_i32..10_000, min_len..=MAX_SAMPLE_LEN);
    let offset_strat = -10_000_i32..10_000;

    (sample_strat, offset_strat).prop_map(|(mut sample, offset)| {
        // Enforce nonconstancy while still working with integers.
        if sample.iter().all(|&x| x == sample[0]) {
            sample[1] = i32::from(sample[0] == 0);
        }

        let sample = sample.into_iter().map(f64::from).collect();
        let offset = f64::from(offset);

        (sample, offset)
    })
}

/// Generates bounded finite `f64` values.
///
/// Values are generated in the range `[-10_000, 10_000)`.
pub fn finite_value() -> impl Strategy<Value = f64> {
    -10_000.0..10_000.0
}

/// Generates bounded finite additive offsets.
///
/// Values are generated in the range `[-100, 100)`, keeping translated test
/// values within a moderate magnitude for numerical property tests.
pub fn finite_offset() -> impl Strategy<Value = f64> {
    -100.0..100.0
}

/// Generates exact positive powers-of-two scaling factors.
///
/// Exponents are generated from `-10` through `10`, producing scales from
/// `2^-10` through `2^10`.
///
/// Powers of two minimize additional rounding when scaling binary
/// floating-point values, making them useful for numerical property tests.
pub fn positive_power_of_two_scale() -> impl Strategy<Value = f64> {
    (-10i32..=10).prop_map(|exponent| 2.0_f64.powi(exponent))
}

/// Generates a single probability in the closed interval `[0.0, 1.0]`.
///
/// Values are generated in one-basis-point increments, including both
/// boundaries.
pub fn probability() -> impl Strategy<Value = f64> {
    (0_u16..=10_000).prop_map(|basis_points| f64::from(basis_points) / 10_000.0)
}

/// Generates a non-empty vector of 1 to 15 probabilities.
///
/// Each probability lies in the closed interval `[0.0, 1.0]` and is generated
/// in one-basis-point increments, including both boundaries.
pub fn probabilities() -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(probability(), 1..=15)
}
