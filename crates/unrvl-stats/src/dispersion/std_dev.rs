use unrvl_numerics::convert::usize_to_f64;
use unrvl_numerics::summation::NeumaierSumExt;

use crate::internal::deviations::centered_scaled;

/// Returns the sample standard deviation using Bessel's correction.
///
/// The statistic is defined as:
///
/// ```text
/// s = sqrt(∑ (xᵢ - x̄)² / (n - 1))
/// ```
///
/// where `n` is the number of observations and `x̄` is their arithmetic mean.
///
/// Returns `NaN` if fewer than two observations are provided or any observation
/// is non-finite. A finite constant sample returns zero.
///
/// Uses scaling, corrected centering, and compensated summation to reduce
/// numerical error. Takes the square root before restoring the scale, so the
/// result can remain finite and nonzero when the variance overflows or
/// underflows. These techniques do not eliminate rounding error; the result can
/// still overflow to positive infinity or underflow to zero.
///
/// # Panics
///
/// Panics if a finite sample contains more than `2^53` observations.
#[must_use]
pub fn std_dev(xs: &[f64]) -> f64 {
    let n = xs.len();

    if n < 2 {
        return f64::NAN;
    }

    let Some((unit, deviations)) = centered_scaled(xs) else {
        return f64::NAN;
    };

    let sum_u2 = deviations.map(|d| d * d).neumaier_sum();
    let scaled_variance = sum_u2 / usize_to_f64(n - 1);

    scaled_variance.sqrt() * unit
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
        let xs = [];
        let result = std_dev(&xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_singleton() {
        let xs = [4.0];
        let result = std_dev(&xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_nan_input() {
        let xs = [1.0, f64::NAN, 3.0];
        let result = std_dev(&xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_positive_infinity() {
        let xs = [1.0, f64::INFINITY, 3.0];
        let result = std_dev(&xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_negative_infinity() {
        let xs = [1.0, f64::NEG_INFINITY, 3.0];
        let result = std_dev(&xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_zero_for_constant_extreme_samples() {
        let tiny = f64::from_bits(1);
        let values = [-f64::MAX, -tiny, 0.0, tiny, f64::MAX];

        for value in values {
            let xs = [value; 3];
            let result = std_dev(&xs);

            assert_eq!(result, 0.0);
        }
    }

    #[test]
    fn returns_zero_for_signed_zeros() {
        let xs = [-0.0, 0.0, -0.0, 0.0];
        let result = std_dev(&xs);

        assert_eq!(result, 0.0);
    }

    #[test]
    fn handles_two_observations() {
        let xs = [2.0, 4.0];
        let expected = 2.0_f64.sqrt();
        let result = std_dev(&xs);

        assert_approx_eq(result, expected, Tolerance::STRICT);
    }

    #[test]
    fn handles_negative_observations() {
        let xs = [-5.0, -3.0, -1.0];
        let expected = 2.0;
        let result = std_dev(&xs);

        assert_eq!(result, expected);
    }

    #[test]
    fn preserves_variation_around_large_offset() {
        let xs = [1e16, 1e16, 1e16 + 2.0];
        let expected = (4.0_f64 / 3.0).sqrt();
        let result = std_dev(&xs);

        assert_approx_eq(result, expected, Tolerance::STRICT);
    }

    #[test]
    fn remains_finite_when_variance_overflows() {
        let magnitude = 2.0_f64.powi(600);
        let xs = [-magnitude, 0.0, magnitude];
        let expected = magnitude;
        let result = std_dev(&xs);

        assert_eq!(result, expected);
    }

    #[test]
    fn remains_nonzero_when_variance_underflows() {
        let magnitude = 2.0_f64.powi(-600);
        let xs = [-magnitude, 0.0, magnitude];
        let expected = magnitude;
        let result = std_dev(&xs);

        assert_eq!(result, expected);
    }

    #[test]
    fn preserves_variation_when_squared_input_scale_overflows() {
        let base = 2.0_f64.powi(600);
        let step = 2.0_f64.powi(548);
        let xs = [base, base, base + step];
        let expected = step / 3.0_f64.sqrt();
        let result = std_dev(&xs);

        assert_approx_eq(result, expected, Tolerance::STRICT);
    }

    #[test]
    fn handles_largest_representable_standard_deviation() {
        let xs = [-f64::MAX, 0.0, f64::MAX];
        let expected = f64::MAX;
        let result = std_dev(&xs);

        assert_approx_eq(result, expected, Tolerance::STRICT);
    }

    #[test]
    fn returns_positive_infinity_when_standard_deviation_overflows() {
        let xs = [-f64::MAX, f64::MAX];
        let result = std_dev(&xs);

        assert_eq!(result, f64::INFINITY);
    }

    #[test]
    fn preserves_smallest_subnormal_standard_deviation() {
        let tiny = f64::from_bits(1);
        let xs = [-tiny, 0.0, tiny];
        let expected = tiny;
        let result = std_dev(&xs);

        assert_eq!(result, expected);
    }

    #[test]
    fn retains_centering_correction_for_subnormal_observations() {
        let tiny = f64::from_bits(1);
        let xs = [0.0, 0.0, tiny];

        // tiny / sqrt(3) rounds to tiny, not zero.
        let expected = tiny;
        let result = std_dev(&xs);

        assert_eq!(result, expected);
    }

    #[test]
    fn unrepresentably_small_standard_deviation_underflows_to_zero() {
        let tiny = f64::from_bits(1);
        let xs = [0.0, 0.0, 0.0, 0.0, tiny];

        // tiny / sqrt(5) rounds to zero.
        let result = std_dev(&xs);

        assert_eq!(result, 0.0);
    }

    proptest! {
        #[test]
        #[expect(clippy::float_cmp, reason = "exactly constant observations")]
        fn constant_sample_has_zero_std_dev(sample in constant_sample(2)) {
            let expected = 0.0;
            let result = std_dev(&sample);

            prop_assert_eq!(result, expected);
        }

        #[test]
        fn bounded_sample_has_finite_nonnegative_std_dev(sample in finite_sample(2)) {
            let result = std_dev(&sample);

            prop_assert!(result.is_finite());
            prop_assert!(result >= 0.0);
        }

        #[test]
        fn reversing_observations_preserves_std_dev(sample in finite_sample(2)) {
            let expected = std_dev(&sample);
            let mut reversed = sample;
            reversed.reverse();
            let result = std_dev(&reversed);

            prop_assert!(approx_eq(result, expected, Tolerance::STRICT));
        }

        #[test]
        fn exact_translation_preserves_std_dev(
            (sample, offset) in exact_translation_case(2),
        ) {
            let translated: Vec<f64> = sample.iter().map(|&x| x + offset).collect();
            let expected = std_dev(&sample);
            let result = std_dev(&translated);

            prop_assert!(approx_eq(result, expected, Tolerance::STRICT));
        }

        #[test]
        fn positive_scaling_scales_std_dev(
            sample in finite_sample(2),
            scale in positive_power_of_two_scale(),
        ) {
            let scaled: Vec<f64> = sample.iter().map(|&x| x * scale).collect();
            let expected = std_dev(&sample) * scale;
            let result = std_dev(&scaled);

            prop_assert!(approx_eq(result, expected, Tolerance::STRICT));
        }

        #[test]
        fn negating_observations_preserves_std_dev(sample in finite_sample(2)) {
            let negated: Vec<f64> = sample.iter().map(|&x| -x).collect();
            let expected = std_dev(&sample);
            let result = std_dev(&negated);

            prop_assert!(approx_eq(result, expected, Tolerance::STRICT));
        }

        #[test]
        fn nonfinite_observation_invalidates_sample(
            mut sample in finite_sample(2),
            value in prop::sample::select(vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY]),
            position in any::<usize>(),
        ) {
            let index = position % sample.len();
            sample[index] = value;
            let result = std_dev(&sample);

            prop_assert!(result.is_nan());
        }
    }
}
