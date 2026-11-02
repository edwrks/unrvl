use unrvl_numerics::convert::usize_to_f64;

use crate::Normalization;
use crate::internal::deviations::CenteredSample;
use crate::internal::second_moment::SecondMoment;

/// Returns the signed coefficient of variation with the selected normalization.
///
/// The statistic is defined as:
///
/// ```text
/// CV = sqrt(∑ (xᵢ - x̄)² / (n - ddof)) / x̄
/// ```
///
/// where `n` is the number of observations and `x̄` is their arithmetic mean.
/// [`Normalization::Population`] uses `ddof = 0`, [`Normalization::Sample`]
/// uses `ddof = 1` (Bessel's correction), and [`Normalization::Ddof`] uses the
/// supplied nonnegative integer. Returns a ratio, not a percentage; a negative
/// mean produces a negative ratio.
///
/// Returns `NaN` for an empty slice, `n <= ddof`, any non-finite observation,
/// or a zero computed mean in scaled units. A finite constant sample with a
/// positive divisor and a nonzero value returns zero, including a singleton
/// with population normalization. A zero singleton returns `NaN`.
///
/// Computes the standard deviation and mean in common scaled units, avoiding
/// the need to represent either statistic in the original units. Uses
/// corrected centering and compensated summation to reduce numerical error.
/// Scaling can still lose very small observations, and cancellation can
/// round the mean to zero. A mean near zero makes the ratio sensitive to
/// rounding; the result can overflow to signed infinity.
///
/// For `n` observations, uses `O(n)` time and `O(1)` additional memory.
///
/// # Panics
///
/// Panics if the divisor is positive and a finite sample contains more than
/// `2^53` observations. The divisor requirement is checked before validating
/// the observations.
///
/// # Examples
///
/// ```
/// # use unrvl_stats::Normalization;
/// # use unrvl_stats::dispersion::cv;
/// let xs = [1.0, 3.0];
/// assert_eq!(cv(&xs, Normalization::Population), 0.5);
/// assert_eq!(cv(&xs, Normalization::Sample), 2.0_f64.sqrt() / 2.0);
///
/// let xs = [1.0, 2.0, 3.0, 4.0];
/// assert_eq!(cv(&xs, Normalization::Ddof(2)), 2.5_f64.sqrt() / 2.5);
/// ```
#[must_use]
pub fn cv(xs: &[f64], normalization: Normalization) -> f64 {
    let Some(divisor) = normalization.divisor(xs.len()) else {
        return f64::NAN;
    };

    let Some(sample) = CenteredSample::new(xs) else {
        return f64::NAN;
    };

    let scaled_mean = sample.centering().scaled_center();
    if scaled_mean == 0.0 {
        return f64::NAN;
    }

    let moment = SecondMoment::from_centered(&sample);
    let scaled_variance = moment.scaled_sum_squares() / usize_to_f64(divisor);

    scaled_variance.sqrt() / scaled_mean
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_utils::approx::{Tolerance, approx_eq, assert_approx_eq};
    use test_utils::strategies::{
        MAX_SAMPLE_LEN, finite_sample, nonfinite_value, positive_power_of_two_scale,
    };

    use super::*;

    const NORMALIZATIONS: [Normalization; 3] = [
        Normalization::Population,
        Normalization::Sample,
        Normalization::Ddof(2),
    ];

    #[test]
    fn returns_nan_for_empty_input() {
        let xs = [];

        for normalization in NORMALIZATIONS {
            let result = cv(&xs, normalization);

            assert!(result.is_nan());
        }
    }

    #[test]
    fn returns_nan_for_sample_singleton() {
        let xs = [4.0];
        let result = cv(&xs, Normalization::Sample);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_nonfinite_observations() {
        for normalization in NORMALIZATIONS {
            for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                for xs in [[value, 1.0, 2.0], [1.0, value, 2.0], [1.0, 2.0, value]] {
                    let result = cv(&xs, normalization);

                    assert!(result.is_nan());
                }
            }
        }
    }

    #[test]
    fn population_singletons_reject_nonfinite_observations() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let result = cv(&[value], Normalization::Population);

            assert!(result.is_nan());
        }
    }

    #[test]
    fn returns_nan_for_invalid_divisors() {
        let xs = [1.0, 3.0];

        for ddof in [2, 3, usize::MAX] {
            let result = cv(&xs, Normalization::Ddof(ddof));

            assert!(result.is_nan());
        }
    }

    #[test]
    fn returns_nan_for_zero_population_singletons() {
        for value in [0.0, -0.0] {
            let result = cv(&[value], Normalization::Population);

            assert!(result.is_nan());
        }
    }

    #[test]
    fn returns_nan_for_zero_mean() {
        let xs = [-2.0, 0.0, 2.0];

        for normalization in NORMALIZATIONS {
            let result = cv(&xs, normalization);

            assert!(result.is_nan());
        }
    }

    #[test]
    fn returns_nan_for_zero_samples() {
        let samples = [[0.0, 0.0], [-0.0, -0.0], [-0.0, 0.0]];

        for normalization in NORMALIZATIONS {
            for xs in samples {
                let result = cv(&xs, normalization);

                assert!(result.is_nan());
            }
        }
    }

    #[test]
    fn returns_nan_when_scaled_mean_underflows_to_zero() {
        let tiny = f64::from_bits(1);
        let xs = [-1.0, tiny, 1.0];

        for normalization in NORMALIZATIONS {
            let result = cv(&xs, normalization);

            assert!(result.is_nan());
        }
    }

    #[test]
    fn accepts_a_finite_nonzero_population_singleton() {
        let values = [
            -f64::MAX,
            -42.0,
            -f64::from_bits(1),
            f64::from_bits(1),
            42.0,
            f64::MAX,
        ];

        for value in values {
            let result = cv(&[value], Normalization::Population);

            assert_eq!(result, 0.0);
        }
    }

    #[test]
    fn returns_zero_for_nonzero_constant_extreme_samples() {
        let tiny = f64::from_bits(1);
        let values = [-f64::MAX, -tiny, tiny, f64::MAX];

        for normalization in NORMALIZATIONS {
            for value in values {
                let xs = [value; 3];
                let result = cv(&xs, normalization);

                assert_eq!(result, 0.0);
            }
        }
    }

    #[test]
    fn returns_ratio_for_positive_mean() {
        let xs = [1.0, 2.0, 3.0];

        let cases = [
            (Normalization::Population, (2.0_f64 / 3.0).sqrt() / 2.0),
            (Normalization::Sample, 0.5),
            (Normalization::Ddof(2), 2.0_f64.sqrt() / 2.0),
        ];

        for (normalization, expected) in cases {
            let result = cv(&xs, normalization);

            assert_approx_eq(result, expected, Tolerance::STRICT);
        }
    }

    #[test]
    fn returns_negative_ratio_for_negative_mean() {
        let xs = [-1.0, -2.0, -3.0];

        let cases = [
            (Normalization::Population, -(2.0_f64 / 3.0).sqrt() / 2.0),
            (Normalization::Sample, -0.5),
            (Normalization::Ddof(2), -2.0_f64.sqrt() / 2.0),
        ];

        for (normalization, expected) in cases {
            let result = cv(&xs, normalization);

            assert_approx_eq(result, expected, Tolerance::STRICT);
        }
    }

    #[test]
    fn calculates_each_normalization_for_a_known_sample() {
        let xs = [1.0, 2.0, 3.0, 4.0];

        // Mean = 2.5; centered sum of squares = 5.
        let cases = [
            (Normalization::Population, 1.25_f64),
            (Normalization::Sample, 5.0 / 3.0),
            (Normalization::Ddof(2), 2.5),
            (Normalization::Ddof(3), 5.0),
        ];

        for (normalization, variance) in cases {
            let expected = variance.sqrt() / 2.5;
            let result = cv(&xs, normalization);

            assert_approx_eq(result, expected, Tolerance::STRICT);
        }
    }

    #[test]
    fn checks_observation_boundaries_without_count_overflow() {
        let xs = [7.0; 4];

        for n in 0..=xs.len() {
            for ddof in [0, 1, 2, n, n + 1, usize::MAX] {
                let result = cv(&xs[..n], Normalization::Ddof(ddof));

                if n > ddof {
                    assert_eq!(result, 0.0);
                } else {
                    assert!(result.is_nan(), "n={n}, ddof={ddof}");
                }
            }
        }
    }

    #[test]
    fn presets_match_explicit_ddof_for_boundary_and_nonfinite_inputs() {
        let samples: &[&[f64]] = &[
            &[],
            &[1.0],
            &[-1.0],
            &[0.0],
            &[-0.0],
            &[1.0, 2.0],
            &[7.0; 4],
            &[-1.0, 0.0, 1.0],
            &[f64::NAN],
            &[f64::INFINITY],
            &[f64::NEG_INFINITY],
            &[1.0, 2.0, f64::NAN],
            &[1.0, 2.0, f64::INFINITY],
            &[1.0, 2.0, f64::NEG_INFINITY],
        ];

        for xs in samples {
            for (preset, ddof) in [(Normalization::Population, 0), (Normalization::Sample, 1)] {
                assert_eq!(
                    cv(xs, preset).to_bits(),
                    cv(xs, Normalization::Ddof(ddof)).to_bits(),
                );
            }
        }
    }

    #[test]
    fn uses_sample_standard_deviation() {
        let xs = [0.0, 2.0];
        let expected = 2.0_f64.sqrt();
        let result = cv(&xs, Normalization::Sample);

        assert_approx_eq(result, expected, Tolerance::STRICT);
    }

    #[test]
    fn uses_population_standard_deviation_for_two_observations() {
        let xs = [1.0, 3.0];
        let result = cv(&xs, Normalization::Population);

        assert_eq!(result, 0.5);
    }

    #[test]
    fn handles_mixed_sign_observations() {
        let xs = [-1.0, 0.0, 4.0];
        let expected = 7.0_f64.sqrt();
        let result = cv(&xs, Normalization::Sample);

        assert_approx_eq(result, expected, Tolerance::STRICT);
    }

    #[test]
    fn remains_finite_when_unscaled_mean_sum_overflows() {
        let xs = [f64::MAX / 2.0, f64::MAX];

        let cases = [
            (Normalization::Population, 1.0 / 3.0),
            (Normalization::Sample, 2.0_f64.sqrt() / 3.0),
        ];

        for (normalization, expected) in cases {
            let result = cv(&xs, normalization);

            assert_approx_eq(result, expected, Tolerance::STRICT);
        }
    }

    #[test]
    fn remains_finite_when_standard_deviation_overflows() {
        let xs = [-f64::MAX, f64::MAX, f64::MAX];

        let cases = [
            (Normalization::Population, 2.0 * 2.0_f64.sqrt()),
            (Normalization::Sample, 2.0 * 3.0_f64.sqrt()),
            (Normalization::Ddof(2), 2.0 * 6.0_f64.sqrt()),
        ];

        for (normalization, expected) in cases {
            let result = cv(&xs, normalization);

            assert_approx_eq(result, expected, Tolerance::STRICT);
        }
    }

    #[test]
    fn remains_finite_when_unscaled_mean_underflows() {
        let tiny = f64::from_bits(1);
        let xs = [0.0, 0.0, tiny];

        let cases = [
            (Normalization::Population, 2.0_f64.sqrt()),
            (Normalization::Sample, 3.0_f64.sqrt()),
            (Normalization::Ddof(2), 6.0_f64.sqrt()),
        ];

        for (normalization, expected) in cases {
            let result = cv(&xs, normalization);

            assert_approx_eq(result, expected, Tolerance::STRICT);
        }
    }

    #[test]
    fn remains_finite_when_standard_deviation_underflows() {
        let tiny = f64::from_bits(1);
        let xs = [0.0, tiny, tiny, tiny, tiny];

        let cases = [
            (Normalization::Population, 0.5),
            (Normalization::Sample, 5.0_f64.sqrt() / 4.0),
            (Normalization::Ddof(2), (5.0_f64 / 12.0).sqrt()),
        ];

        for (normalization, expected) in cases {
            let result = cv(&xs, normalization);

            assert_approx_eq(result, expected, Tolerance::STRICT);
        }
    }

    #[test]
    fn preserves_variation_around_large_offset() {
        let xs = [1e16, 1e16, 1e16 + 2.0];
        let tolerance = Tolerance::new(0.0, 1e-14);

        let cases = [
            (Normalization::Population, 8.0_f64 / 9.0),
            (Normalization::Sample, 4.0 / 3.0),
            (Normalization::Ddof(2), 8.0 / 3.0),
        ];

        for (normalization, variance) in cases {
            let expected = variance.sqrt() / 1e16;
            let result = cv(&xs, normalization);

            assert_approx_eq(result, expected, tolerance);
        }
    }

    #[test]
    fn preserves_mean_after_cancellation() {
        let magnitude = 2.0_f64.powi(60);
        let xs = [magnitude, 1.0, -magnitude];

        let cases = [
            (Normalization::Population, 6.0_f64.sqrt()),
            (Normalization::Sample, 3.0),
            (Normalization::Ddof(2), 3.0 * 2.0_f64.sqrt()),
        ];

        for (normalization, factor) in cases {
            let expected = factor * magnitude;
            let result = cv(&xs, normalization);

            assert_approx_eq(result, expected, Tolerance::STRICT);
        }
    }

    #[test]
    fn overflowing_ratio_retains_mean_sign() {
        let samples = [[-f64::MAX, 1.0, f64::MAX], [-f64::MAX, -1.0, f64::MAX]];
        let expected = [f64::INFINITY, f64::NEG_INFINITY];

        for normalization in NORMALIZATIONS {
            for (xs, expected) in samples.iter().zip(expected) {
                let result = cv(xs, normalization);

                assert_eq!(result, expected);
            }
        }
    }

    proptest! {
        // Positive bounded observations keep the mean away from zero so these
        // properties test the ratio without ill-conditioned cancellation.
        #[test]
        fn positive_sample_has_finite_nonnegative_cv(
            sample in prop::collection::vec(1.0_f64..10_000.0, 1..=MAX_SAMPLE_LEN),
            adjustment in 0..MAX_SAMPLE_LEN,
        ) {
            let normalization = Normalization::Ddof(adjustment % sample.len());
            let result = cv(&sample, normalization);

            prop_assert!(result.is_finite());
            prop_assert!(result >= 0.0);
        }

        #[test]
        fn positive_scaling_preserves_cv(
            sample in prop::collection::vec(1.0_f64..10_000.0, 1..=MAX_SAMPLE_LEN),
            scale in positive_power_of_two_scale(),
            adjustment in 0..MAX_SAMPLE_LEN,
        ) {
            let normalization = Normalization::Ddof(adjustment % sample.len());
            let scaled: Vec<f64> = sample.iter().map(|&x| x * scale).collect();
            let expected = cv(&sample, normalization);
            let result = cv(&scaled, normalization);

            prop_assert!(approx_eq(result, expected, Tolerance::STRICT));
        }

        #[test]
        fn negating_observations_negates_cv(
            sample in prop::collection::vec(1.0_f64..10_000.0, 1..=MAX_SAMPLE_LEN),
            adjustment in 0..MAX_SAMPLE_LEN,
        ) {
            let normalization = Normalization::Ddof(adjustment % sample.len());
            let negated: Vec<f64> = sample.iter().map(|&x| -x).collect();
            let expected = -cv(&sample, normalization);
            let result = cv(&negated, normalization);

            prop_assert!(approx_eq(result, expected, Tolerance::STRICT));
        }

        #[test]
        fn reversing_observations_preserves_cv(
            sample in prop::collection::vec(1.0_f64..10_000.0, 1..=MAX_SAMPLE_LEN),
            adjustment in 0..MAX_SAMPLE_LEN,
        ) {
            let normalization = Normalization::Ddof(adjustment % sample.len());
            let expected = cv(&sample, normalization);
            let mut reversed = sample;
            reversed.reverse();
            let result = cv(&reversed, normalization);

            prop_assert!(approx_eq(result, expected, Tolerance::STRICT));
        }

        #[test]
        fn presets_match_explicit_ddof(sample in finite_sample(1)) {
            for (preset, ddof) in [(Normalization::Population, 0), (Normalization::Sample, 1)] {
                prop_assert_eq!(
                    cv(&sample, preset).to_bits(),
                    cv(&sample, Normalization::Ddof(ddof)).to_bits(),
                );
            }
        }

        #[test]
        fn nonfinite_observation_invalidates_sample(
            mut sample in finite_sample(1),
            value in nonfinite_value(),
            position in any::<usize>(),
            adjustment in 0..MAX_SAMPLE_LEN,
        ) {
            let normalization = Normalization::Ddof(adjustment % sample.len());
            let index = position % sample.len();
            sample[index] = value;
            let result = cv(&sample, normalization);

            prop_assert!(result.is_nan());
        }

        #[test]
        fn sample_and_population_cvs_have_expected_ratio(
            sample in prop::collection::vec(1.0_f64..10_000.0, 2..=MAX_SAMPLE_LEN),
        ) {
            let n = usize_to_f64(sample.len());
            let sample_cv = cv(&sample, Normalization::Sample);
            let population_cv = cv(&sample, Normalization::Population);

            prop_assert!(approx_eq(
                population_cv,
                sample_cv * ((n - 1.0) / n).sqrt(),
                Tolerance::STRICT,
            ));
        }
    }
}
