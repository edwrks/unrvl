use unrvl_numerics::convert::usize_to_f64;

use crate::Normalization;
use crate::internal::deviations::CenteredSample;
use crate::internal::second_moment::SecondMoment;

/// Returns z-scores with the selected normalization, in original observation
/// order.
///
/// Each score is mathematically `(x - mean) / s`, where `s` is the standard
/// deviation with divisor `n - ddof`. [`Normalization::Population`] uses
/// `ddof = 0`, [`Normalization::Sample`] uses `ddof = 1` (Bessel's correction),
/// and [`Normalization::Ddof`] uses the supplied nonnegative integer. Scores
/// are dimensionless.
///
/// An empty slice returns an empty vector. An invalid divisor (`n <= ddof`),
/// zero spread, or any non-finite observation produces one `NaN` per input
/// observation. Singletons produce `NaN` with every normalization because their
/// spread is zero or the divisor is invalid.
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
/// Panics if the divisor is positive and a finite sample contains more than
/// `2^53` observations. The divisor requirement is checked before validating
/// the observations, including for `Ddof(usize::MAX)`.
///
/// # Examples
///
/// ```
/// # use unrvl_stats::transform::z_score;
/// use unrvl_stats::Normalization;
///
/// let xs = [3.0, 1.0, 2.0];
/// assert_eq!(z_score(&xs, Normalization::Sample), vec![1.0, -1.0, 0.0]);
/// assert_eq!(
///     z_score(&[10.0, 4.0], Normalization::Population),
///     vec![1.0, -1.0]
/// );
/// ```
#[must_use]
pub fn z_score(xs: &[f64], normalization: Normalization) -> Vec<f64> {
    let n = xs.len();

    let Some(divisor) = normalization.divisor(n) else {
        return vec![f64::NAN; n];
    };

    let Some(sample) = CenteredSample::new(xs) else {
        return vec![f64::NAN; n];
    };

    let moment = SecondMoment::from_centered(&sample);
    let scaled_variance = moment.scaled_sum_squares() / usize_to_f64(divisor);
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
        MAX_SAMPLE_LEN, exact_translation_case, finite_nonconstant_sample, finite_sample,
        nonfinite_value, positive_power_of_two_scale,
    };
    use unrvl_numerics::summation::NeumaierSumExt;

    use super::*;
    use crate::dispersion::std_dev;
    use crate::transform::demean;

    const NORMALIZATIONS: [Normalization; 3] = [
        Normalization::Population,
        Normalization::Sample,
        Normalization::Ddof(2),
    ];

    #[test]
    fn singleton_returns_nan() {
        let tiny = f64::from_bits(1);
        let values = [
            0.0,
            -0.0,
            42.0,
            tiny,
            -tiny,
            f64::MAX,
            -f64::MAX,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        let normalizations = [
            Normalization::Population,
            Normalization::Sample,
            Normalization::Ddof(2),
            Normalization::Ddof(usize::MAX),
        ];

        for normalization in normalizations {
            for value in values {
                let actual = z_score(&[value], normalization);

                assert_eq!(actual.len(), 1);
                assert!(actual[0].is_nan());
            }
        }
    }

    #[test]
    fn constant_samples_return_nan() {
        let tiny = f64::from_bits(1);
        let values = [0.0, -0.0, 42.0, tiny, -tiny, f64::MAX, -f64::MAX];

        for normalization in NORMALIZATIONS {
            for value in values {
                let xs = [value; 3];
                let actual = z_score(&xs, normalization);

                assert_eq!(actual.len(), xs.len());
                assert!(actual.into_iter().all(f64::is_nan));
            }

            let xs = [-0.0, 0.0, -0.0, 0.0];
            let actual = z_score(&xs, normalization);

            assert_eq!(actual.len(), xs.len());
            assert!(actual.into_iter().all(f64::is_nan));
        }
    }

    #[test]
    fn nonfinite_observations_make_all_results_nan() {
        for normalization in NORMALIZATIONS {
            for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                for index in 0..3 {
                    let mut xs = [1.0, 2.0, 3.0];
                    xs[index] = value;
                    let actual = z_score(&xs, normalization);

                    assert_eq!(actual.len(), xs.len());
                    assert!(actual.into_iter().all(f64::is_nan));
                }
            }
        }
    }

    #[test]
    fn translation_that_rounds_to_a_constant_returns_nan() {
        let e = f64::EPSILON;
        let xs = [
            1.0,
            1.0 + e,
            2.0_f64.mul_add(e, 1.0),
            3.0_f64.mul_add(e, 1.0),
        ];
        let translated = xs.map(|x| x + 100.0);

        // The translation erased the distinctions between observations.
        assert_eq!(translated, [101.0; 4]);

        for normalization in NORMALIZATIONS {
            assert!(z_score(&xs, normalization).into_iter().all(f64::is_finite));

            let actual = z_score(&translated, normalization);
            assert_eq!(actual.len(), xs.len());
            assert!(actual.into_iter().all(f64::is_nan));
        }
    }

    #[test]
    fn checks_observation_boundaries_without_count_overflow() {
        let xs = [1.0, 2.0, 3.0, 4.0];

        for n in 0..=xs.len() {
            let adjustments = [0, 1, 2, n, n + 1, usize::MAX];

            for ddof in adjustments {
                let actual = z_score(&xs[..n], Normalization::Ddof(ddof));
                assert_eq!(actual.len(), n);

                // A singleton has zero spread even when its divisor is valid.
                if n >= 2 && n > ddof {
                    assert!(actual.into_iter().all(f64::is_finite));
                } else {
                    assert!(actual.into_iter().all(f64::is_nan));
                }
            }
        }
    }

    #[test]
    fn empty_input_returns_an_empty_vector() {
        let normalizations = [
            Normalization::Population,
            Normalization::Sample,
            Normalization::Ddof(2),
            Normalization::Ddof(usize::MAX),
        ];

        for normalization in normalizations {
            assert_eq!(z_score(&[], normalization), [] as [f64; 0]);
        }
    }

    #[test]
    fn standardizes_in_original_observation_order() {
        let xs = [3.0, 1.0, 2.0];
        let actual = z_score(&xs, Normalization::Sample);
        let expected = [1.0, -1.0, 0.0];

        assert_eq!(actual, expected);
    }

    #[test]
    fn distinguishes_population_and_sample_for_two_observations() {
        let xs = [10.0, 4.0];
        let cases = [
            (Normalization::Population, [1.0, -1.0]),
            (Normalization::Sample, [FRAC_1_SQRT_2, -FRAC_1_SQRT_2]),
        ];

        for (normalization, expected) in cases {
            let actual = z_score(&xs, normalization);

            assert_all_approx_eq(&actual, &expected, Tolerance::VERY_STRICT);
        }
    }

    #[test]
    fn calculates_each_normalization_for_a_known_sample() {
        let xs = [1.0, 2.0, 3.0, 4.0];
        let deviations = [-1.5, -0.5, 0.5, 1.5];

        // Mean 2.5 gives a squared-deviation sum of 5.
        let cases = [
            (Normalization::Population, 1.25_f64.sqrt()),
            (Normalization::Sample, (5.0 / 3.0_f64).sqrt()),
            (Normalization::Ddof(2), 2.5_f64.sqrt()),
            (Normalization::Ddof(3), 5.0_f64.sqrt()),
        ];

        for (normalization, standard_deviation) in cases {
            let actual = z_score(&xs, normalization);
            let expected = deviations.map(|deviation| deviation / standard_deviation);

            assert_all_approx_eq(&actual, &expected, Tolerance::VERY_STRICT);
        }
    }

    #[test]
    fn presets_match_explicit_ddof_for_boundary_and_nonfinite_inputs() {
        let cases: &[&[f64]] = &[
            &[],
            &[42.0],
            &[1.0, 2.0],
            &[7.0; 3],
            &[-0.0, 0.0],
            &[1.0, f64::NAN, 3.0],
            &[f64::INFINITY, 2.0],
            &[1.0, f64::NEG_INFINITY],
        ];
        let aliases = [
            (Normalization::Population, Normalization::Ddof(0)),
            (Normalization::Sample, Normalization::Ddof(1)),
        ];

        for xs in cases {
            for (preset, explicit) in aliases {
                let a = z_score(xs, preset);
                let b = z_score(xs, explicit);

                assert_eq!(a.len(), b.len());
                for (a, b) in a.iter().zip(&b) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
            }
        }
    }

    #[test]
    fn preserves_small_changes_around_a_large_offset() {
        let xs = [1e16, 1e16, 1e16 + 2.0];
        let cases = [
            (Normalization::Population, FRAC_1_SQRT_2),
            (Normalization::Sample, 1.0 / 3.0_f64.sqrt()),
            (Normalization::Ddof(2), 1.0 / 6.0_f64.sqrt()),
        ];

        for (normalization, step) in cases {
            let actual = z_score(&xs, normalization);
            let expected = [-step, -step, 2.0 * step];

            assert_all_approx_eq(&actual, &expected, Tolerance::VERY_STRICT);
        }
    }

    #[test]
    fn preserves_signed_zeros_in_observation_order() {
        let xs = [-0.0, 1.0, 0.0, -1.0];
        let cases = [
            (Normalization::Population, 2.0_f64.sqrt()),
            (Normalization::Sample, 1.5_f64.sqrt()),
            (Normalization::Ddof(2), 1.0),
        ];

        for (normalization, magnitude) in cases {
            let actual = z_score(&xs, normalization);
            let expected = [-0.0, magnitude, 0.0, -magnitude];

            assert_all_approx_eq(&actual, &expected, Tolerance::VERY_STRICT);
            assert_eq!(actual[0].to_bits(), (-0.0_f64).to_bits());
            assert_eq!(actual[2].to_bits(), 0.0_f64.to_bits());
        }
    }

    #[test]
    fn remains_finite_when_original_standard_deviation_overflows() {
        let xs = [-f64::MAX, f64::MAX];
        let actual = z_score(&xs, Normalization::Sample);
        let expected = [-FRAC_1_SQRT_2, FRAC_1_SQRT_2];

        assert_eq!(std_dev(&xs, Normalization::Sample), f64::INFINITY);
        assert_all_approx_eq(&actual, &expected, Tolerance::VERY_STRICT);
    }

    #[test]
    fn remains_finite_when_unscaled_deviations_overflow() {
        let xs = [-f64::MAX, f64::MAX, f64::MAX];
        let cases = [
            (Normalization::Population, FRAC_1_SQRT_2),
            (Normalization::Sample, 1.0 / 3.0_f64.sqrt()),
            (Normalization::Ddof(2), 1.0 / 6.0_f64.sqrt()),
        ];

        assert_eq!(demean(&xs)[0], f64::NEG_INFINITY);

        for (normalization, step) in cases {
            let actual = z_score(&xs, normalization);
            let expected = [-2.0 * step, step, step];

            assert_all_approx_eq(&actual, &expected, Tolerance::VERY_STRICT);
        }
    }

    #[test]
    fn remains_finite_when_original_standard_deviation_rounds_to_zero() {
        let tiny = f64::from_bits(1);
        let xs = [0.0, 0.0, 0.0, tiny];
        let actual = z_score(&xs, Normalization::Sample);
        let expected = [-0.5, -0.5, -0.5, 1.5];

        assert_eq!(std_dev(&xs, Normalization::Sample), 0.0);
        assert_eq!(actual, expected);
    }

    #[test]
    fn population_scores_survive_unscaled_standard_deviation_underflow() {
        let tiny = f64::from_bits(1);
        let xs = [0.0, 0.0, 0.0, tiny];
        let step = 1.0 / 3.0_f64.sqrt();
        let actual = z_score(&xs, Normalization::Population);
        let expected = [-step, -step, -step, 3.0 * step];

        assert_eq!(std_dev(&xs, Normalization::Population), 0.0);
        assert_all_approx_eq(&actual, &expected, Tolerance::VERY_STRICT);
    }

    proptest! {
        #[test]
        fn scores_have_zero_mean_and_unit_selected_variance(
            xs in finite_nonconstant_sample(2),
            adjustment in 0..MAX_SAMPLE_LEN,
        ) {
            let count = xs.len();
            let custom_ddof = adjustment % count;
            let n = usize_to_f64(count);
            let cases = [
                (Normalization::Population, count),
                (Normalization::Sample, count - 1),
                (Normalization::Ddof(custom_ddof), count - custom_ddof),
            ];

            for (normalization, divisor) in cases {
                let actual = z_score(&xs, normalization);
                let mean = actual.iter().neumaier_sum() / n;
                let variance = actual.iter()
                    .map(|score| score * score)
                    .neumaier_sum() / usize_to_f64(divisor);

                prop_assert_eq!(actual.len(), xs.len());
                prop_assert!(approx_eq(mean, 0.0, Tolerance::STRICT));
                prop_assert!(approx_eq(variance, 1.0, Tolerance::STRICT));
            }
        }

        #[test]
        fn exact_affine_changes_preserve_or_negate_scores(
            (xs, offset) in exact_translation_case(2),
            magnitude in positive_power_of_two_scale(),
            adjustment in 0..MAX_SAMPLE_LEN,
        ) {
            let normalizations = [
                Normalization::Population,
                Normalization::Sample,
                Normalization::Ddof(adjustment % xs.len()),
            ];

            for scale in [magnitude, -magnitude] {
                let transformed: Vec<_> = xs.iter()
                    .map(|&value| scale.mul_add(value, offset))
                    .collect();
                let sign = scale.signum();

                for normalization in normalizations {
                    let expected = z_score(&xs, normalization);
                    let actual = z_score(&transformed, normalization);
                    prop_assert_eq!(actual.len(), expected.len());

                    for (&actual, &expected) in actual.iter().zip(&expected) {
                        prop_assert!(approx_eq(actual, sign * expected, Tolerance::STRICT));
                    }
                }
            }
        }

        #[test]
        fn presets_match_explicit_ddof(xs in finite_sample(1)) {
            let aliases = [
                (Normalization::Population, Normalization::Ddof(0)),
                (Normalization::Sample, Normalization::Ddof(1)),
            ];

            for (preset, explicit) in aliases {
                let a = z_score(&xs, preset);
                let b = z_score(&xs, explicit);

                prop_assert_eq!(a.len(), b.len());
                for (a, b) in a.iter().zip(&b) {
                    prop_assert_eq!(a.to_bits(), b.to_bits());
                }
            }
        }

        #[test]
        fn nonfinite_observation_invalidates_all_scores(
            mut xs in finite_sample(3),
            value in nonfinite_value(),
            position in any::<usize>(),
            adjustment in 0..MAX_SAMPLE_LEN,
        ) {
            let index = position % xs.len();
            xs[index] = value;
            let normalizations = [
                Normalization::Population,
                Normalization::Sample,
                Normalization::Ddof(adjustment % xs.len()),
            ];

            for normalization in normalizations {
                let actual = z_score(&xs, normalization);

                prop_assert_eq!(actual.len(), xs.len());
                prop_assert!(actual.into_iter().all(f64::is_nan));
            }
        }
    }
}
