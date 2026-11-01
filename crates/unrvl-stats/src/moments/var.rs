use unrvl_numerics::convert::usize_to_f64;

use crate::Normalization;
use crate::internal::second_moment::SecondMoment;

/// Returns the variance with the selected normalization.
///
/// The statistic is defined as:
///
/// ```text
///         ∑ (xᵢ - x̄)²
/// v = -----------------
///         n - ddof
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
/// numerical error. These do not eliminate rounding or underflow. The result
/// can overflow to positive infinity or underflow to zero.
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
/// use unrvl_stats::moments::var;
///
/// let xs = [1.0, 2.0, 3.0];
/// assert_eq!(var(&xs, Normalization::Sample), 1.0);
/// assert_eq!(var(&xs, Normalization::Population), 2.0 / 3.0);
/// assert_eq!(var(&xs, Normalization::Ddof(2)), 2.0);
/// ```
#[must_use]
pub fn var(xs: &[f64], normalization: Normalization) -> f64 {
    let Some(divisor) = normalization.divisor(xs.len()) else {
        return f64::NAN;
    };

    let Some(moment) = SecondMoment::new(xs) else {
        return f64::NAN;
    };

    let unit = moment.unit();
    let scaled_variance = moment.scaled_sum_squares() / usize_to_f64(divisor);

    (scaled_variance * unit) * unit
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_utils::approx::{Tolerance, approx_eq, assert_approx_eq};
    use test_utils::strategies::{
        MAX_SAMPLE_LEN, constant_sample, exact_translation_case, finite_sample,
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
        let xs = &[];
        let result = var(xs, Normalization::Sample);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_too_few_inputs() {
        let xs = &[1.0];
        let result = var(xs, Normalization::Sample);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_nan_input() {
        let xs = &[1.0, 2.0, f64::NAN];
        let result = var(xs, Normalization::Sample);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_positive_infinity() {
        let xs = &[1.0, 2.0, f64::INFINITY];
        let result = var(xs, Normalization::Sample);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_negative_infinity() {
        let xs = &[1.0, 2.0, f64::NEG_INFINITY];
        let result = var(xs, Normalization::Sample);

        assert!(result.is_nan());
    }

    #[test]
    fn all_normalizations_reject_nonfinite_observations() {
        for normalization in NORMALIZATIONS {
            for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                for xs in [[value, 1.0, 2.0], [1.0, value, 2.0], [1.0, 2.0, value]] {
                    assert!(var(&xs, normalization).is_nan());
                }
            }
        }
    }

    #[test]
    fn calculates_variance() {
        let xs = &[2.0, 5.0, 9.0, 12.0];
        let result = var(xs, Normalization::Sample);

        assert_approx_eq(result, 58.0 / 3.0, Tolerance::DEFAULT);
    }

    #[test]
    fn preserves_variation_around_large_offset() {
        let xs = &[1e16, 1e16, 1e16 + 2.0];

        let cases = [
            (Normalization::Population, 8.0 / 9.0),
            (Normalization::Sample, 4.0 / 3.0),
            (Normalization::Ddof(2), 8.0 / 3.0),
        ];

        for (normalization, expected) in cases {
            assert_approx_eq(var(xs, normalization), expected, Tolerance::DEFAULT);
        }
    }

    #[test]
    fn returns_finite_variance_when_unscaled_sum_of_squares_overflows() {
        let xs = [-1e154, 0.0, 1e154];
        let result = var(&xs, Normalization::Sample);
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
        let actual = var(&xs, Normalization::Sample);

        assert_approx_eq(actual, expected, Tolerance::STRICT);
    }

    #[test]
    fn preserves_subnormal_variances_for_each_normalization() {
        let x = 2.0_f64.powi(-530);

        // x² = 2^-1060 has subnormal bit pattern 2^14 = 16384.
        // Population variance rounds (2/3) * x² to bit pattern 10923.
        let cases = [
            (Normalization::Population, 10_923),
            (Normalization::Sample, 16_384),
            (Normalization::Ddof(2), 32_768),
        ];

        for (normalization, expected_bits) in cases {
            assert_eq!(
                var(&[-x, 0.0, x], normalization),
                f64::from_bits(expected_bits)
            );
        }
    }

    #[test]
    fn unrepresentably_small_variance_underflows_to_zero() {
        let d = f64::from_bits(1);
        let xs = [0.0, 0.0, d];
        for normalization in NORMALIZATIONS {
            assert_eq!(var(&xs, normalization), 0.0);
        }
    }

    #[test]
    fn unrepresentably_large_variance_overflows_to_positive_infinity() {
        let xs = [-f64::MAX, 0.0, f64::MAX];
        for normalization in NORMALIZATIONS {
            assert_eq!(var(&xs, normalization), f64::INFINITY);
        }
    }

    #[test]
    fn constant_extreme_samples_have_zero_variance() {
        let values = [
            0.0,
            -0.0,
            f64::from_bits(1),
            -f64::from_bits(1),
            f64::MAX,
            -f64::MAX,
        ];
        for normalization in NORMALIZATIONS {
            for value in values {
                assert_eq!(var(&[value; 3], normalization), 0.0);
            }
        }
    }

    #[test]
    fn calculates_each_normalization_for_a_known_sample() {
        let xs = [1.0, 2.0, 3.0, 4.0];

        // Mean = 2.5; centered sum of squares = 5.
        let cases = [
            (Normalization::Population, 1.25),
            (Normalization::Sample, 5.0 / 3.0),
            (Normalization::Ddof(2), 2.5),
            (Normalization::Ddof(3), 5.0),
        ];

        for (normalization, expected) in cases {
            assert_approx_eq(var(&xs, normalization), expected, Tolerance::STRICT);
        }
    }

    #[test]
    fn accepts_a_finite_population_singleton() {
        for value in [0.0, -0.0, 42.0, f64::from_bits(1), f64::MAX, -f64::MAX] {
            assert_eq!(var(&[value], Normalization::Population), 0.0);
        }
    }

    #[test]
    fn calculates_two_point_variances() {
        let xs = [1.0, 2.0];

        assert_eq!(var(&xs, Normalization::Population), 0.25);
        assert_eq!(var(&xs, Normalization::Sample), 0.5);
    }

    #[test]
    fn checks_observation_boundaries_without_count_overflow() {
        let xs = [7.0; 4];

        for n in 0..=xs.len() {
            for ddof in [0, 1, 2, n, n + 1, usize::MAX] {
                let result = var(&xs[..n], Normalization::Ddof(ddof));
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
            &[1.0, 2.0, f64::NAN],
            &[1.0, 2.0, f64::INFINITY],
            &[1.0, 2.0, f64::NEG_INFINITY],
        ];

        for xs in samples {
            for (preset, ddof) in [(Normalization::Population, 0), (Normalization::Sample, 1)] {
                assert_eq!(
                    var(xs, preset).to_bits(),
                    var(xs, Normalization::Ddof(ddof)).to_bits()
                );
            }
        }
    }

    #[test]
    fn population_variance_remains_finite_when_sample_variance_overflows() {
        let xs = [-1e154, 1e154];

        assert_eq!(var(&xs, Normalization::Sample), f64::INFINITY);
        assert_approx_eq(
            var(&xs, Normalization::Population),
            1e308,
            Tolerance::STRICT,
        );
    }

    proptest! {
        #[test]
        #[expect(clippy::float_cmp, reason = "exactly constant observations")]
        fn constant_sample_has_zero_variance(
            sample in constant_sample(1),
            adjustment in 0..MAX_SAMPLE_LEN,
        ) {
            let normalization = Normalization::Ddof(adjustment % sample.len());
            let expected = 0.0;
            let result = var(&sample, normalization);
            prop_assert_eq!(result, expected);
        }

        #[test]
        fn variance_is_non_negative(
            sample in finite_sample(1),
            adjustment in 0..MAX_SAMPLE_LEN,
        ) {
            let normalization = Normalization::Ddof(adjustment % sample.len());
            let actual = var(&sample, normalization);

            prop_assert!(actual.is_finite());
            prop_assert!(actual >= 0.0);
        }

        #[test]
        fn variance_is_translation_invariant(
            (sample, offset) in exact_translation_case(2),
            adjustment in 0..MAX_SAMPLE_LEN,
        ) {
            let normalization = Normalization::Ddof(adjustment % sample.len());
            let translated: Vec<_> = sample.iter().map(|&x| x + offset).collect();

            let x = var(&sample, normalization);
            let y = var(&translated, normalization);

            prop_assert!(approx_eq(x, y, Tolerance::DEFAULT));
        }

        #[test]
        fn variance_is_scale_equivariant(
            sample in finite_sample(1),
            scale in positive_power_of_two_scale(),
            adjustment in 0..MAX_SAMPLE_LEN,
        ) {
            let normalization = Normalization::Ddof(adjustment % sample.len());
            let scaled: Vec<_> = sample.iter().map(|&x| x * scale).collect();

            let x = var(&sample, normalization) * scale * scale;
            let y = var(&scaled, normalization);

            prop_assert!(approx_eq(x, y, Tolerance::DEFAULT));
        }

        #[test]
        fn variance_is_reversal_invariant(
            sample in finite_sample(1),
            adjustment in 0..MAX_SAMPLE_LEN,
        ) {
            let normalization = Normalization::Ddof(adjustment % sample.len());
            let mut reversed = sample.clone();
            reversed.reverse();

            let x = var(&sample, normalization);
            let y = var(&reversed, normalization);

            prop_assert!(approx_eq(x, y, Tolerance::DEFAULT));
        }

        #[test]
        fn presets_match_explicit_ddof(sample in finite_sample(1)) {
            for (preset, ddof) in [(Normalization::Population, 0), (Normalization::Sample, 1)] {
                prop_assert_eq!(
                    var(&sample, preset).to_bits(),
                    var(&sample, Normalization::Ddof(ddof)).to_bits()
                );
            }
        }

        #[test]
        fn sample_and_population_variances_have_expected_ratio(sample in finite_sample(2)) {
            let n = usize_to_f64(sample.len());
            let sample_variance = var(&sample, Normalization::Sample);
            let population_variance = var(&sample, Normalization::Population);

            prop_assert!(approx_eq(
                population_variance,
                sample_variance * ((n - 1.0) / n),
                Tolerance::STRICT,
            ));
        }
    }
}
