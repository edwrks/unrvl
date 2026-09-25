use unrvl_numerics::convert::usize_to_f64;
use unrvl_numerics::summation::NeumaierSumExt;

use crate::internal::deviations::centered_scaled;

/// Returns the signed sample coefficient of variation.
///
/// The statistic is defined as:
///
/// ```text
/// CV = s / x̄
/// ```
///
/// where `s` is the sample standard deviation using Bessel's correction and
/// `x̄` is the arithmetic mean. Returns a ratio, not a percentage; a negative
/// mean produces a negative ratio.
///
/// Returns `NaN` if fewer than two observations are provided, any observation
/// is non-finite, or the computed mean in scaled units is zero. A finite
/// constant sample with a nonzero value returns zero.
///
/// Computes the standard deviation and mean in common scaled units, avoiding
/// the need to represent either statistic in the original units. Uses
/// corrected centering and compensated summation to reduce numerical error.
/// Scaling can still lose very small observations, and cancellation can
/// round the mean to zero. A mean near zero makes the ratio sensitive to
/// rounding; the result can overflow to signed infinity.
///
/// # Panics
///
/// Panics if a finite sample contains more than `2^53` observations.
#[must_use]
pub fn cv(xs: &[f64]) -> f64 {
    let n = xs.len();

    if n < 2 {
        return f64::NAN;
    }

    let Some((unit, deviations)) = centered_scaled(xs) else {
        return f64::NAN;
    };

    let scaled_mean = xs.iter().map(|&x| x / unit).neumaier_sum() / usize_to_f64(n);
    if scaled_mean == 0.0 {
        return f64::NAN;
    }

    let sum_u2 = deviations.map(|d| d * d).neumaier_sum();
    let scaled_variance = sum_u2 / usize_to_f64(n - 1);

    scaled_variance.sqrt() / scaled_mean
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_utils::approx::{Tolerance, approx_eq, assert_approx_eq};
    use test_utils::strategies::{MAX_SAMPLE_LEN, positive_power_of_two_scale};

    use super::*;

    #[test]
    fn returns_nan_for_empty_input() {
        let xs = [];
        let result = cv(&xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_singleton() {
        let xs = [4.0];
        let result = cv(&xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_nonfinite_observations() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let xs = [1.0, value, 3.0];
            let result = cv(&xs);

            assert!(result.is_nan());
        }
    }

    #[test]
    fn returns_nan_for_zero_mean() {
        let xs = [-2.0, 0.0, 2.0];
        let result = cv(&xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_zero_samples() {
        let samples = [[0.0, 0.0], [-0.0, -0.0], [-0.0, 0.0]];

        for xs in samples {
            let result = cv(&xs);

            assert!(result.is_nan());
        }
    }

    #[test]
    fn returns_zero_for_nonzero_constant_extreme_samples() {
        let tiny = f64::from_bits(1);
        let values = [-f64::MAX, -tiny, tiny, f64::MAX];

        for value in values {
            let xs = [value; 3];
            let result = cv(&xs);

            assert_eq!(result, 0.0);
        }
    }

    #[test]
    fn returns_ratio_for_positive_mean() {
        let xs = [1.0, 2.0, 3.0];
        let expected = 0.5;
        let result = cv(&xs);

        assert_eq!(result, expected);
    }

    #[test]
    fn returns_negative_ratio_for_negative_mean() {
        let xs = [-1.0, -2.0, -3.0];
        let expected = -0.5;
        let result = cv(&xs);

        assert_eq!(result, expected);
    }

    #[test]
    fn uses_sample_standard_deviation() {
        let xs = [0.0, 2.0];
        let expected = 2.0_f64.sqrt();
        let result = cv(&xs);

        assert_approx_eq(result, expected, Tolerance::STRICT);
    }

    #[test]
    fn handles_mixed_sign_observations() {
        let xs = [-1.0, 0.0, 4.0];
        let expected = 7.0_f64.sqrt();
        let result = cv(&xs);

        assert_approx_eq(result, expected, Tolerance::STRICT);
    }

    #[test]
    fn remains_finite_when_unscaled_mean_sum_overflows() {
        let xs = [f64::MAX / 2.0, f64::MAX];
        let expected = 2.0_f64.sqrt() / 3.0;
        let result = cv(&xs);

        assert_approx_eq(result, expected, Tolerance::STRICT);
    }

    #[test]
    fn remains_finite_when_standard_deviation_overflows() {
        let xs = [-f64::MAX, f64::MAX, f64::MAX];
        let expected = 2.0 * 3.0_f64.sqrt();
        let result = cv(&xs);

        assert_approx_eq(result, expected, Tolerance::STRICT);
    }

    #[test]
    fn remains_finite_when_unscaled_mean_underflows() {
        let tiny = f64::from_bits(1);
        let xs = [0.0, 0.0, tiny];
        let expected = 3.0_f64.sqrt();
        let result = cv(&xs);

        assert_approx_eq(result, expected, Tolerance::STRICT);
    }

    #[test]
    fn remains_finite_when_standard_deviation_underflows() {
        let tiny = f64::from_bits(1);
        let xs = [0.0, tiny, tiny, tiny, tiny];
        let expected = 5.0_f64.sqrt() / 4.0;
        let result = cv(&xs);

        assert_approx_eq(result, expected, Tolerance::STRICT);
    }

    #[test]
    fn preserves_variation_around_large_offset() {
        let xs = [1e16, 1e16, 1e16 + 2.0];
        let expected = (4.0_f64 / 3.0).sqrt() / 1e16;
        let tolerance = Tolerance::new(0.0, 1e-14);
        let result = cv(&xs);

        assert_approx_eq(result, expected, tolerance);
    }

    #[test]
    fn preserves_mean_after_cancellation() {
        let magnitude = 2.0_f64.powi(60);
        let xs = [magnitude, 1.0, -magnitude];
        let expected = 3.0 * magnitude;
        let result = cv(&xs);

        assert_approx_eq(result, expected, Tolerance::STRICT);
    }

    #[test]
    fn overflowing_ratio_retains_mean_sign() {
        let samples = [[-f64::MAX, 1.0, f64::MAX], [-f64::MAX, -1.0, f64::MAX]];
        let expected = [f64::INFINITY, f64::NEG_INFINITY];

        for (xs, expected) in samples.iter().zip(expected) {
            let result = cv(xs);

            assert_eq!(result, expected);
        }
    }

    #[test]
    fn returns_nan_when_scaled_mean_underflows_to_zero() {
        let tiny = f64::from_bits(1);
        let xs = [-1.0, tiny, 1.0];
        let result = cv(&xs);

        assert!(result.is_nan());
    }

    proptest! {
        // Positive bounded observations keep the mean away from zero so these
        // properties test the ratio without ill-conditioned cancellation.
        #[test]
        fn positive_sample_has_finite_nonnegative_cv(
            sample in prop::collection::vec(1.0_f64..10_000.0, 2..=MAX_SAMPLE_LEN),
        ) {
            let result = cv(&sample);

            prop_assert!(result.is_finite());
            prop_assert!(result >= 0.0);
        }

        #[test]
        fn positive_scaling_preserves_cv(
            sample in prop::collection::vec(1.0_f64..10_000.0, 2..=MAX_SAMPLE_LEN),
            scale in positive_power_of_two_scale(),
        ) {
            let scaled: Vec<f64> = sample.iter().map(|&x| x * scale).collect();
            let expected = cv(&sample);
            let result = cv(&scaled);

            prop_assert!(approx_eq(result, expected, Tolerance::STRICT));
        }

        #[test]
        fn negating_observations_negates_cv(
            sample in prop::collection::vec(1.0_f64..10_000.0, 2..=MAX_SAMPLE_LEN),
        ) {
            let negated: Vec<f64> = sample.iter().map(|&x| -x).collect();
            let expected = -cv(&sample);
            let result = cv(&negated);

            prop_assert!(approx_eq(result, expected, Tolerance::STRICT));
        }

        #[test]
        fn reversing_observations_preserves_cv(
            sample in prop::collection::vec(1.0_f64..10_000.0, 2..=MAX_SAMPLE_LEN),
        ) {
            let expected = cv(&sample);
            let mut reversed = sample;
            reversed.reverse();
            let result = cv(&reversed);

            prop_assert!(approx_eq(result, expected, Tolerance::STRICT));
        }
    }
}
