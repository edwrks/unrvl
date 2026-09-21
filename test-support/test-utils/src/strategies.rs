//! Proptest strategy generators for numerical tests.

use proptest::prelude::prop;
use proptest::strategy::Strategy;

/// Generates non-empty samples of bounded finite `f64` values.
///
/// Values are generated in the range `[-10_000, 10_000)`, avoiding non-finite
/// values and extreme magnitudes that could introduce unrelated overflow or
/// precision issues into property tests.
///
/// # Panics
///
/// Panics if the min length is zero.
pub fn finite_samples(min_len: usize) -> impl Strategy<Value = Vec<f64>> {
    assert!(min_len > 0, "minimum sample length must be nonzero");

    prop::collection::vec(-10_000.0..10_000.0, min_len..64)
}

/// Generates nonconstant samples of bounded finite `f64` values.
///
/// Values are generated in the range `[-10_000, 10_000)`. Samples contain at
/// least two distinct values, making them suitable for statistics that are
/// undefined for zero-variance samples.
///
/// # Panics
///
/// Panics if the min length is less than `2`.
#[expect(clippy::float_cmp, reason = "detect exactly constant observations")]
pub fn finite_nonconstant_samples(min_len: usize) -> impl Strategy<Value = Vec<f64>> {
    assert!(
        min_len >= 2,
        "minimum length for a nonconstant sample must be at least two"
    );

    finite_samples(min_len).prop_map(|mut xs| {
        if xs.iter().all(|&x| x == xs[0]) {
            xs[1] = if xs[0] == 0.0 { 1.0 } else { 0.0 };
        }

        xs
    })
}

/// Generates non-empty, constant samples of bounded finite `f64` values.
///
/// The constant value is generated in the range `[-10_000, 10_000)`, avoiding
/// non-finite values and extreme magnitudes that could introduce unrelated
/// overflow or precision issues into property tests.
///
/// # Panics
///
/// Panics if the min length is zero.
pub fn constant_samples(min_len: usize) -> impl Strategy<Value = Vec<f64>> {
    assert!(min_len > 0, "minimum sample length must be nonzero");

    (finite_value(), min_len..64).prop_map(|(value, len)| vec![value; len])
}

/// Generates bounded finite `f64` values.
///
/// Values are generated in the range `[-10_000, 10_000)`.
pub fn finite_value() -> impl Strategy<Value = f64> {
    -10_000.0..10_000.0
}

/// Generates bounded finite scaling factors.
///
/// Values are generated in the range `[-100, 100)`, keeping scaled test values
/// within a moderate magnitude for numerical property tests.
pub fn finite_scale() -> impl Strategy<Value = f64> {
    -100.0..100.0
}

/// Generates exact nonzero powers-of-two scales.
///
/// Powers of two minimize additional rounding when scaling binary
/// floating-point values, making them useful for numerical property tests.
pub fn power_of_two_scale() -> impl Strategy<Value = f64> {
    (-10i32..=10).prop_map(|exponent| 2.0_f64.powi(exponent))
}
