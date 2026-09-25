use unrvl_numerics::convert::usize_to_f64;
use unrvl_numerics::summation::NeumaierSumExt;

use crate::internal::deviations::centered_scaled;

/// Returns the sample variance of a slice using Bessel's correction.
///
/// The statistic is defined as:
///
/// ```text
///         ∑ (xᵢ - x̄)²
/// s² = -----------------
///           n - 1
/// ```
///
/// where `x̄` is the sample mean.
///
/// Observations are divided by a power-of-two unit before centering. Centering
/// retains a correction for mean rounding to better preserve small variations
/// around a large or nearly constant level.
///
/// Squared deviations are accumulated with Neumaier compensated summation in
/// scaled units. The result is divided by `n - 1`, then converted back to the
/// original squared units without first squaring the scaling unit.
///
/// Returns `NaN` if fewer than two observations are provided or when the input
/// contains non-finite values. A finite constant sample returns zero.
///
/// Scaling reduces avoidable intermediate overflow and underflow but does not
/// eliminate rounding or extend the representable result range. Restoring the
/// original units can overflow to positive infinity, and very small nonzero
/// variances can underflow to zero.
///
/// # Panics
///
/// Panics if a finite sample contains more than `2^53` observations.
#[must_use]
pub fn var(xs: &[f64]) -> f64 {
    let n = xs.len();

    if n < 2 {
        return f64::NAN;
    }

    let Some((unit, deviations)) = centered_scaled(xs) else {
        return f64::NAN;
    };

    let sum_u2 = deviations.map(|d| d * d).neumaier_sum();
    let scaled_variance = sum_u2 / usize_to_f64(n - 1);
    (scaled_variance * unit) * unit
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_utils::approx::{Tolerance, approx_eq, assert_approx_eq};
    use test_utils::strategies::{
        constant_sample, exact_translation_case, finite_sample, positive_power_of_two_scale,
    };

    use super::*;

    #[test]
    fn returns_nan_for_empty_input() {
        let xs = &[];
        let result = var(xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_too_few_inputs() {
        let xs = &[1.0];
        let result = var(xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_nan_input() {
        let xs = &[1.0, 2.0, f64::NAN];
        let result = var(xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_positive_infinity() {
        let xs = &[1.0, 2.0, f64::INFINITY];
        let result = var(xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_negative_infinity() {
        let xs = &[1.0, 2.0, f64::NEG_INFINITY];
        let result = var(xs);

        assert!(result.is_nan());
    }

    #[test]
    fn calculates_variance() {
        let xs = &[2.0, 5.0, 9.0, 12.0];
        let result = var(xs);

        assert_approx_eq(result, 58.0 / 3.0, Tolerance::DEFAULT);
    }

    #[test]
    fn preserves_variation_around_large_offset() {
        let xs = &[1e16, 1e16, 1e16 + 2.0];
        let result = var(xs);

        assert_approx_eq(result, 4.0 / 3.0, Tolerance::DEFAULT);
    }

    #[test]
    fn returns_finite_variance_when_unscaled_sum_of_squares_overflows() {
        let xs = [-1e154, 0.0, 1e154];
        let result = var(&xs);
        let expected = 1e308;

        assert_approx_eq(result, expected, Tolerance::STRICT);
    }

    #[test]
    fn returns_finite_variance_when_squared_input_scale_overflows() {
        let base = 2.0_f64.powi(520);
        let step = 2.0_f64.powi(468);
        let xs = [base, base, base + step];

        // For [a, a, a + h], sample variance is h² / 3.
        let expected = 2.0_f64.powi(936) / 3.0;
        let actual = var(&xs);

        assert_approx_eq(actual, expected, Tolerance::STRICT);
    }

    #[test]
    fn preserves_an_exactly_representable_subnormal_variance() {
        let x = 2.0_f64.powi(-530);

        // For [-x, 0, x], sample variance is x² = 2^-1060.
        // A subnormal bit pattern of 2^14 represents 2^-1060.
        let result = var(&[-x, 0.0, x]);
        let expected = f64::from_bits(1_u64 << 14);

        assert_eq!(result, expected);
    }

    #[test]
    fn unrepresentably_small_variance_underflows_to_zero() {
        let d = f64::from_bits(1);
        let xs = [0.0, 0.0, d];
        let result = var(&xs);

        assert_eq!(result, 0.0);
    }

    #[test]
    fn unrepresentably_large_variance_overflows_to_positive_infinity() {
        let xs = [-f64::MAX, 0.0, f64::MAX];
        let result = var(&xs);

        assert_eq!(result, f64::INFINITY,);
    }

    #[test]
    fn constant_extreme_samples_have_zero_variance() {
        let values = [f64::from_bits(1), -f64::from_bits(1), f64::MAX, -f64::MAX];
        for value in values {
            assert_eq!(var(&[value; 3]), 0.0);
        }
    }

    proptest! {
        #[test]
        #[expect(clippy::float_cmp, reason = "exactly constant observations")]
        fn constant_sample_has_zero_variance(sample in constant_sample(2)) {
            let expected = 0.0;
            let result = var(&sample);
            prop_assert_eq!(result, expected);
        }

        #[test]
        fn variance_is_non_negative(sample in finite_sample(2)) {
            let actual = var(&sample);

            prop_assert!(actual.is_finite());
            prop_assert!(actual >= 0.0);
        }

        #[test]
        fn variance_is_translation_invariant(
            (sample, offset) in exact_translation_case(2)
        ) {
            let translated: Vec<_> = sample.iter().map(|&x| x + offset).collect();

            let x = var(&sample);
            let y = var(&translated);

            prop_assert!(approx_eq(x, y, Tolerance::DEFAULT));
        }

        #[test]
        fn variance_is_scale_equivariant(
            sample in finite_sample(2),
            scale in positive_power_of_two_scale(),
        ) {
            let scaled: Vec<_> = sample.iter().map(|&x| x * scale).collect();

            let x = var(&sample) * scale * scale;
            let y = var(&scaled);

            prop_assert!(approx_eq(x, y, Tolerance::DEFAULT));
        }

        #[test]
        fn variance_is_reversal_invariant(sample in finite_sample(2)) {
            let mut reversed = sample.clone();
            reversed.reverse();

            let x = var(&sample);
            let y = var(&reversed);

            prop_assert!(approx_eq(x, y, Tolerance::DEFAULT));
        }

    }
}
