use unrvl_numerics::convert::usize_to_f64;

use crate::Normalization;
use crate::internal::second_moment::SecondMoment;

/// Returns the standard deviation with the selected normalization.
///
/// The statistic is defined as:
///
/// ```text
/// s = sqrt(∑ (xᵢ - x̄)² / (n - ddof))
/// ```
///
/// where `n` is the number of observations and `x̄` is their arithmetic mean.
/// [`Normalization::Population`] uses `ddof = 0`, [`Normalization::Sample`]
/// uses `ddof = 1` (Bessel's correction), and [`Normalization::Ddof`] uses the
/// supplied nonnegative integer.
///
/// Returns `NaN` for an empty slice, `n <= ddof`, or any non-finite
/// observation. A finite constant sample with a positive divisor returns zero,
/// including a singleton with population normalization. Even `Ddof(usize::MAX)`
/// returns `NaN` for insufficient observations without overflowing the count
/// arithmetic.
///
/// Uses scaling, corrected centering, and compensated summation to reduce
/// numerical error. Takes the square root before restoring the scale, so the
/// result can remain finite and nonzero when the variance overflows or
/// underflows. These techniques do not eliminate rounding error; the result can
/// still overflow to positive infinity or underflow to zero.
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
/// use unrvl_stats::Normalization;
/// use unrvl_stats::dispersion::std_dev;
///
/// let xs = [1.0, 3.0];
/// assert_eq!(std_dev(&xs, Normalization::Population), 1.0);
/// assert_eq!(std_dev(&xs, Normalization::Sample), 2.0_f64.sqrt());
///
/// let xs = [1.0, 2.0, 3.0, 4.0];
/// assert_eq!(std_dev(&xs, Normalization::Ddof(2)), 2.5_f64.sqrt());
/// ```
#[must_use]
pub fn std_dev(xs: &[f64], normalization: Normalization) -> f64 {
    let Some(divisor) = normalization.divisor(xs.len()) else {
        return f64::NAN;
    };

    let Some(moment) = SecondMoment::new(xs) else {
        return f64::NAN;
    };

    let unit = moment.unit();
    let scaled_variance = moment.scaled_sum_squares() / usize_to_f64(divisor);

    scaled_variance.sqrt() * unit
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_utils::approx::{Tolerance, approx_eq, assert_approx_eq};
    use test_utils::strategies::{
        MAX_SAMPLE_LEN, constant_sample, exact_translation_case, finite_sample, nonfinite_value,
        positive_power_of_two_scale,
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
            let result = std_dev(&xs, normalization);

            assert!(result.is_nan());
        }
    }

    #[test]
    fn returns_nan_for_sample_singleton() {
        let xs = [4.0];
        let result = std_dev(&xs, Normalization::Sample);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_nan_input() {
        let xs = [1.0, f64::NAN, 3.0];

        for normalization in NORMALIZATIONS {
            let result = std_dev(&xs, normalization);

            assert!(result.is_nan());
        }
    }

    #[test]
    fn returns_nan_for_positive_infinity() {
        let xs = [1.0, f64::INFINITY, 3.0];

        for normalization in NORMALIZATIONS {
            let result = std_dev(&xs, normalization);

            assert!(result.is_nan());
        }
    }

    #[test]
    fn returns_nan_for_negative_infinity() {
        let xs = [1.0, f64::NEG_INFINITY, 3.0];

        for normalization in NORMALIZATIONS {
            let result = std_dev(&xs, normalization);

            assert!(result.is_nan());
        }
    }

    #[test]
    fn returns_nan_for_invalid_divisors() {
        let xs = [1.0, 3.0];

        for ddof in [2, 3, usize::MAX] {
            let result = std_dev(&xs, Normalization::Ddof(ddof));

            assert!(result.is_nan());
        }
    }

    #[test]
    fn population_singletons_reject_nonfinite_observations() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let result = std_dev(&[value], Normalization::Population);

            assert!(result.is_nan());
        }
    }

    #[test]
    fn accepts_a_finite_population_singleton() {
        for value in [0.0, -0.0, 42.0, f64::from_bits(1), f64::MAX, -f64::MAX] {
            let result = std_dev(&[value], Normalization::Population);

            assert_eq!(result, 0.0);
        }
    }

    #[test]
    fn returns_zero_for_constant_extreme_samples() {
        let tiny = f64::from_bits(1);
        let values = [-f64::MAX, -tiny, -0.0, 0.0, tiny, f64::MAX];

        for normalization in NORMALIZATIONS {
            for value in values {
                let xs = [value; 3];
                let result = std_dev(&xs, normalization);

                assert_eq!(result, 0.0);
            }
        }
    }

    #[test]
    fn returns_zero_for_signed_zeros() {
        let xs = [-0.0, 0.0, -0.0, 0.0];

        for normalization in NORMALIZATIONS {
            let result = std_dev(&xs, normalization);

            assert_eq!(result, 0.0);
        }
    }

    #[test]
    fn handles_two_observations() {
        let xs = [2.0, 4.0];

        assert_eq!(std_dev(&xs, Normalization::Population), 1.0);
        assert_approx_eq(
            std_dev(&xs, Normalization::Sample),
            2.0_f64.sqrt(),
            Tolerance::STRICT,
        );
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
            let result = std_dev(&xs, normalization);

            assert_approx_eq(result, variance.sqrt(), Tolerance::STRICT);
        }
    }

    #[test]
    fn checks_observation_boundaries_without_count_overflow() {
        let xs = [7.0; 4];

        for n in 0..=xs.len() {
            for ddof in [0, 1, 2, n, n + 1, usize::MAX] {
                let result = std_dev(&xs[..n], Normalization::Ddof(ddof));

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
            &[1.0, 2.0],
            &[7.0; 4],
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
                    std_dev(xs, preset).to_bits(),
                    std_dev(xs, Normalization::Ddof(ddof)).to_bits(),
                );
            }
        }
    }

    #[test]
    fn handles_negative_observations() {
        let xs = [-5.0, -3.0, -1.0];
        let expected = 2.0;
        let result = std_dev(&xs, Normalization::Sample);

        assert_eq!(result, expected);
    }

    #[test]
    fn preserves_variation_around_large_offset() {
        let xs = [1e16, 1e16, 1e16 + 2.0];

        let cases = [
            (Normalization::Population, 8.0_f64 / 9.0),
            (Normalization::Sample, 4.0 / 3.0),
            (Normalization::Ddof(2), 8.0 / 3.0),
        ];

        for (normalization, variance) in cases {
            let result = std_dev(&xs, normalization);

            assert_approx_eq(result, variance.sqrt(), Tolerance::STRICT);
        }
    }

    #[test]
    fn remains_finite_when_variance_overflows() {
        let magnitude = 2.0_f64.powi(600);
        let xs = [-magnitude, 0.0, magnitude];

        let cases = [
            (Normalization::Population, (2.0_f64 / 3.0).sqrt()),
            (Normalization::Sample, 1.0),
            (Normalization::Ddof(2), 2.0_f64.sqrt()),
        ];

        for (normalization, factor) in cases {
            let expected = magnitude * factor;
            let result = std_dev(&xs, normalization);

            assert_eq!(result, expected);
        }
    }

    #[test]
    fn remains_nonzero_when_variance_underflows() {
        let magnitude = 2.0_f64.powi(-600);
        let xs = [-magnitude, 0.0, magnitude];

        let cases = [
            (Normalization::Population, (2.0_f64 / 3.0).sqrt()),
            (Normalization::Sample, 1.0),
            (Normalization::Ddof(2), 2.0_f64.sqrt()),
        ];

        for (normalization, factor) in cases {
            let expected = magnitude * factor;
            let result = std_dev(&xs, normalization);

            assert_eq!(result, expected);
        }
    }

    #[test]
    fn preserves_variation_when_squared_input_scale_overflows() {
        let base = 2.0_f64.powi(600);
        let step = 2.0_f64.powi(548);
        let xs = [base, base, base + step];

        let cases = [
            (Normalization::Population, 2.0_f64.sqrt() / 3.0),
            (Normalization::Sample, 1.0 / 3.0_f64.sqrt()),
            (Normalization::Ddof(2), (2.0_f64 / 3.0).sqrt()),
        ];

        for (normalization, factor) in cases {
            let expected = step * factor;
            let result = std_dev(&xs, normalization);

            assert_approx_eq(result, expected, Tolerance::STRICT);
        }
    }

    #[test]
    fn handles_largest_representable_standard_deviation() {
        let xs = [-f64::MAX, 0.0, f64::MAX];
        let expected = f64::MAX;
        let result = std_dev(&xs, Normalization::Sample);

        assert_approx_eq(result, expected, Tolerance::STRICT);
    }

    #[test]
    fn returns_positive_infinity_when_standard_deviation_overflows() {
        let xs = [-f64::MAX, f64::MAX];
        let result = std_dev(&xs, Normalization::Sample);

        assert_eq!(result, f64::INFINITY);

        let xs = [-f64::MAX, 0.0, f64::MAX];
        let result = std_dev(&xs, Normalization::Ddof(2));

        assert_eq!(result, f64::INFINITY);
    }

    #[test]
    fn population_std_dev_remains_finite_when_sample_std_dev_overflows() {
        let xs = [-f64::MAX, f64::MAX];

        assert_eq!(std_dev(&xs, Normalization::Sample), f64::INFINITY);
        assert_eq!(std_dev(&xs, Normalization::Population), f64::MAX);
    }

    #[test]
    fn preserves_smallest_subnormal_standard_deviation() {
        let tiny = f64::from_bits(1);
        let xs = [-tiny, 0.0, tiny];
        let expected = tiny;

        for normalization in NORMALIZATIONS {
            let result = std_dev(&xs, normalization);

            assert_eq!(result, expected);
        }
    }

    #[test]
    fn retains_centering_correction_for_subnormal_observations() {
        let tiny = f64::from_bits(1);
        let xs = [0.0, 0.0, tiny];

        // Population SD is tiny * sqrt(2) / 3, which rounds to zero.
        // Sample SD is tiny / sqrt(3); custom SD is tiny * sqrt(2/3).
        // Both of the latter round to tiny.
        let cases = [
            (Normalization::Population, 0.0),
            (Normalization::Sample, tiny),
            (Normalization::Ddof(2), tiny),
        ];

        for (normalization, expected) in cases {
            let result = std_dev(&xs, normalization);

            assert_eq!(result, expected);
        }
    }

    #[test]
    fn unrepresentably_small_standard_deviation_underflows_to_zero() {
        let tiny = f64::from_bits(1);
        let xs = [0.0, 0.0, 0.0, 0.0, tiny];

        // Population SD is tiny * 2/5; sample SD is tiny / sqrt(5).
        // Both round to zero.
        for normalization in [Normalization::Population, Normalization::Sample] {
            let result = std_dev(&xs, normalization);

            assert_eq!(result, 0.0);
        }
    }

    proptest! {
        #[test]
        #[expect(clippy::float_cmp, reason = "exactly constant observations")]
        fn constant_sample_has_zero_std_dev(
            sample in constant_sample(1),
            adjustment in 0..MAX_SAMPLE_LEN,
        ) {
            let normalization = Normalization::Ddof(adjustment % sample.len());
            let expected = 0.0;
            let result = std_dev(&sample, normalization);

            prop_assert_eq!(result, expected);
        }

        #[test]
        fn bounded_sample_has_finite_nonnegative_std_dev(
            sample in finite_sample(1),
            adjustment in 0..MAX_SAMPLE_LEN,
        ) {
            let normalization = Normalization::Ddof(adjustment % sample.len());
            let result = std_dev(&sample, normalization);

            prop_assert!(result.is_finite());
            prop_assert!(result >= 0.0);
        }

        #[test]
        fn reversing_observations_preserves_std_dev(
            sample in finite_sample(1),
            adjustment in 0..MAX_SAMPLE_LEN,
        ) {
            let normalization = Normalization::Ddof(adjustment % sample.len());
            let expected = std_dev(&sample, normalization);
            let mut reversed = sample;
            reversed.reverse();
            let result = std_dev(&reversed, normalization);

            prop_assert!(approx_eq(result, expected, Tolerance::STRICT));
        }

        #[test]
        fn exact_translation_preserves_std_dev(
            (sample, offset) in exact_translation_case(2),
            adjustment in 0..MAX_SAMPLE_LEN,
        ) {
            let normalization = Normalization::Ddof(adjustment % sample.len());
            let translated: Vec<f64> = sample.iter().map(|&x| x + offset).collect();
            let expected = std_dev(&sample, normalization);
            let result = std_dev(&translated, normalization);

            prop_assert!(approx_eq(result, expected, Tolerance::STRICT));
        }

        #[test]
        fn positive_scaling_scales_std_dev(
            sample in finite_sample(1),
            scale in positive_power_of_two_scale(),
            adjustment in 0..MAX_SAMPLE_LEN,
        ) {
            let normalization = Normalization::Ddof(adjustment % sample.len());
            let scaled: Vec<f64> = sample.iter().map(|&x| x * scale).collect();
            let expected = std_dev(&sample, normalization) * scale;
            let result = std_dev(&scaled, normalization);

            prop_assert!(approx_eq(result, expected, Tolerance::STRICT));
        }

        #[test]
        fn negating_observations_preserves_std_dev(
            sample in finite_sample(1),
            adjustment in 0..MAX_SAMPLE_LEN,
        ) {
            let normalization = Normalization::Ddof(adjustment % sample.len());
            let negated: Vec<f64> = sample.iter().map(|&x| -x).collect();
            let expected = std_dev(&sample, normalization);
            let result = std_dev(&negated, normalization);

            prop_assert!(approx_eq(result, expected, Tolerance::STRICT));
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
            let result = std_dev(&sample, normalization);

            prop_assert!(result.is_nan());
        }

        #[test]
        fn presets_match_explicit_ddof(sample in finite_sample(1)) {
            for (preset, ddof) in [(Normalization::Population, 0), (Normalization::Sample, 1)] {
                prop_assert_eq!(
                    std_dev(&sample, preset).to_bits(),
                    std_dev(&sample, Normalization::Ddof(ddof)).to_bits(),
                );
            }
        }

        #[test]
        fn sample_and_population_std_devs_have_expected_ratio(sample in finite_sample(2)) {
            let n = usize_to_f64(sample.len());
            let sample_std_dev = std_dev(&sample, Normalization::Sample);
            let population_std_dev = std_dev(&sample, Normalization::Population);

            prop_assert!(approx_eq(
                population_std_dev,
                sample_std_dev * ((n - 1.0) / n).sqrt(),
                Tolerance::STRICT,
            ));
        }
    }
}
