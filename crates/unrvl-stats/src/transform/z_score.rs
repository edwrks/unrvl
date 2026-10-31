use unrvl_numerics::convert::usize_to_f64;
use unrvl_numerics::summation::NeumaierSumExt;

use crate::internal::deviations::CenteredSample;

/// Returns sample z-scores in the original observation order.
///
/// Each score is mathematically `(x - mean) / s`, where `s` is the sample
/// standard deviation with denominator `n - 1`. Scores are dimensionless.
///
/// An empty slice returns an empty vector. A singleton, a constant sample, or
/// input containing any non-finite observation produces one `NaN` per input
/// observation.
///
/// Uses scaled, corrected centering and compensated summation. The common
/// scaling unit cancels in the ratio, so scores can remain finite even
/// when the standard deviation in original units overflows or rounds to zero.
/// Results may differ from calculations using a separately rounded mean or
/// standard deviation. Rounding and underflow remain possible, and very small
/// contributions relative to the input scale can be lost.
///
/// Uses `O(n)` time and `O(n)` memory for the returned vector. Does not modify
/// the input observations.
///
/// # Panics
///
/// Panics if a finite sample contains more than `2^53` observations.
///
/// # Examples
///
/// ```
/// use unrvl_stats::transform::z_score;
///
/// let xs = [3.0, 1.0, 2.0];
/// assert_eq!(z_score(&xs), vec![1.0, -1.0, 0.0]);
/// ```
#[must_use]
pub fn z_score(xs: &[f64]) -> Vec<f64> {
    let n = xs.len();

    if n < 2 {
        return vec![f64::NAN; n];
    }

    let Some(sample) = CenteredSample::new(xs) else {
        return vec![f64::NAN; n];
    };

    let sum_squares = sample
        .deviations()
        .map(|deviation| deviation * deviation)
        .neumaier_sum();

    let scaled_variance = sum_squares / usize_to_f64(n - 1);
    let scaled_std_dev = scaled_variance.sqrt();

    if scaled_std_dev == 0.0 {
        return vec![f64::NAN; n];
    }

    // Deviations and standard deviation share a unit, which cancels here.
    sample
        .deviations()
        .map(|deviation| deviation / scaled_std_dev)
        .collect()
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_1_SQRT_2;

    use proptest::prelude::*;
    use test_utils::approx::{Tolerance, approx_eq, assert_all_approx_eq};
    use test_utils::strategies::{
        exact_translation_case, finite_nonconstant_sample, positive_power_of_two_scale,
    };

    use super::*;
    use crate::dispersion::std_dev;

    #[test]
    fn standardizes_in_original_observation_order() {
        let xs = [3.0, 1.0, 2.0];

        let actual = z_score(&xs);
        let expected = [1.0, -1.0, 0.0];

        assert_eq!(actual, expected);
    }

    #[test]
    fn uses_sample_standard_deviation_for_two_observations() {
        let xs = [10.0, 4.0];

        let actual = z_score(&xs);
        let expected = [FRAC_1_SQRT_2, -FRAC_1_SQRT_2];

        assert_all_approx_eq(&actual, &expected, Tolerance::VERY_STRICT);
    }

    #[test]
    fn empty_input_returns_an_empty_vector() {
        let actual = z_score(&[]);

        assert_eq!(actual, [] as [f64; 0]);
    }

    #[test]
    fn singleton_returns_nan() {
        let actual = z_score(&[42.0]);

        assert_eq!(actual.len(), 1);
        assert!(actual[0].is_nan());
    }

    #[test]
    fn constant_samples_return_nan() {
        let values = [0.0, -0.0, 42.0, f64::from_bits(1), f64::MAX, -f64::MAX];

        for value in values {
            let xs = [value; 3];

            let actual = z_score(&xs);

            assert_eq!(actual.len(), xs.len());
            assert!(actual.into_iter().all(f64::is_nan));
        }

        let xs = [-0.0, 0.0, -0.0, 0.0];

        let actual = z_score(&xs);

        assert_eq!(actual.len(), xs.len());
        assert!(actual.into_iter().all(f64::is_nan));
    }

    #[test]
    fn nonfinite_observations_make_all_results_nan() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let xs = [1.0, value, 3.0];

            let actual = z_score(&xs);

            assert_eq!(actual.len(), xs.len());
            assert!(actual.into_iter().all(f64::is_nan));
        }
    }

    #[test]
    fn preserves_small_changes_around_a_large_offset() {
        let xs = [1e16, 1e16, 1e16 + 2.0];
        let step = 1.0 / 3.0_f64.sqrt();

        let actual = z_score(&xs);
        let expected = [-step, -step, 2.0 * step];

        assert_all_approx_eq(&actual, &expected, Tolerance::VERY_STRICT);
    }

    #[test]
    fn remains_finite_when_original_standard_deviation_overflows() {
        let xs = [-f64::MAX, f64::MAX];

        let actual = z_score(&xs);
        let expected = [-FRAC_1_SQRT_2, FRAC_1_SQRT_2];

        assert_eq!(std_dev(&xs), f64::INFINITY);
        assert_all_approx_eq(&actual, &expected, Tolerance::VERY_STRICT);
    }

    #[test]
    fn remains_finite_when_original_standard_deviation_rounds_to_zero() {
        let tiny = f64::from_bits(1);
        let xs = [0.0, 0.0, 0.0, tiny];

        let actual = z_score(&xs);
        let expected = [-0.5, -0.5, -0.5, 1.5];

        assert_eq!(std_dev(&xs), 0.0);
        assert_eq!(actual, expected);
    }

    proptest! {
        #[test]
        fn scores_have_zero_mean_and_unit_sample_variance(
            xs in finite_nonconstant_sample(2),
        ) {
            let actual = z_score(&xs);

            let n = usize_to_f64(xs.len());
            let mean = actual.iter().neumaier_sum() / n;
            let variance = actual.iter()
                .map(|score| score * score)
                .neumaier_sum() / (n - 1.0);

            prop_assert_eq!(actual.len(), xs.len());
            prop_assert!(approx_eq(mean, 0.0, Tolerance::STRICT));
            prop_assert!(approx_eq(variance, 1.0, Tolerance::STRICT));
        }

        #[test]
        fn exact_affine_changes_preserve_or_negate_scores(
            (xs, offset) in exact_translation_case(2),
            magnitude in positive_power_of_two_scale(),
        ) {
            let expected = z_score(&xs);

            for scale in [magnitude, -magnitude] {
                let transformed: Vec<_> = xs.iter()
                    .map(|&value| scale.mul_add(value, offset))
                    .collect();

                let actual = z_score(&transformed);
                let sign = scale.signum();

                prop_assert_eq!(actual.len(), expected.len());

                for (&actual, &expected) in actual.iter().zip(&expected) {
                    prop_assert!(approx_eq(actual, sign * expected, Tolerance::STRICT));
                }
            }
        }
    }
}
