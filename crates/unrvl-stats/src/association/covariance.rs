use unrvl_numerics::convert::usize_to_f64;
use unrvl_numerics::scaling::{binary_exponent, scale_power_of_two};
use unrvl_numerics::summation::NeumaierSum;

use crate::Normalization;
use crate::internal::deviations::CenteredSample;

/// Returns the covariance of paired observations with the selected
/// normalization.
///
/// The statistic is defined as:
///
/// ```text
///              ∑ (xᵢ - x̄)(yᵢ - ȳ)
/// cov(x, y) = ---------------------
///                   n - ddof
/// ```
///
/// Observations are paired by position. [`Normalization::Population`] uses
/// `ddof = 0`, [`Normalization::Sample`] uses `ddof = 1` (Bessel's correction),
/// and [`Normalization::Ddof`] uses the supplied nonnegative integer.
///
/// Returns `NaN` for empty input, `n <= ddof`, or any non-finite observation.
/// Finite samples with a positive divisor have zero covariance if either is
/// constant, including a single pair with population normalization. Both
/// samples are validated before calculating covariance; no pairs are discarded.
///
/// Uses separate power-of-two scales, corrected centering, and compensation
/// for product and summation rounding. These measures do not eliminate rounding
/// or underflow; small contributions relative to the input scales can be lost.
/// The result can overflow to signed infinity or underflow to zero.
///
/// Uses `O(n)` time and `O(1)` additional memory, where `n` is the number of
/// paired observations.
///
/// # Panics
///
/// Panics if the slices have different lengths. This is checked before the
/// divisor and observations. With a positive divisor, panics if a finite sample
/// contains more than `2^53` observations. Divisor validity is checked before
/// validating the observations, including for `Ddof(usize::MAX)`.
///
/// # Examples
///
/// ```
/// # use unrvl_stats::Normalization;
/// # use unrvl_stats::association::covariance;
/// let xs = [1.0, 2.0, 3.0];
/// let ys = [2.0, 4.0, 6.0];
/// assert_eq!(covariance(&xs, &ys, Normalization::Sample), 2.0);
/// assert_eq!(covariance(&xs, &ys, Normalization::Population), 4.0 / 3.0);
/// assert_eq!(covariance(&xs, &ys, Normalization::Ddof(2)), 4.0);
/// ```
#[must_use]
pub fn covariance(xs: &[f64], ys: &[f64], normalization: Normalization) -> f64 {
    assert_eq!(xs.len(), ys.len(), "inputs must have equal lengths");

    let Some(divisor) = normalization.divisor(xs.len()) else {
        return f64::NAN;
    };

    let Some(x_sample) = CenteredSample::new(xs) else {
        return f64::NAN;
    };

    let Some(y_sample) = CenteredSample::new(ys) else {
        return f64::NAN;
    };

    let mut sum = NeumaierSum::new();
    for (x, y) in x_sample.deviations().zip(y_sample.deviations()) {
        sum.add_product(x, y);
    }

    let scaled_covariance = sum.total() / usize_to_f64(divisor);
    restore_units(
        scaled_covariance,
        x_sample.centering().unit(),
        y_sample.centering().unit(),
    )
}

/// Restores covariance units from two positive, finite power-of-two scales.
/// Shared by scalar and matrix covariance to preserve extreme-scale handling.
pub(super) fn restore_units(scaled_covariance: f64, x_unit: f64, y_unit: f64) -> f64 {
    let x_exponent = binary_exponent(x_unit).expect("positive finite scaling unit");
    let y_exponent = binary_exponent(y_unit).expect("positive finite scaling unit");

    scale_power_of_two(scaled_covariance, x_exponent + y_exponent)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_utils::approx::{Tolerance, approx_eq, assert_approx_eq};
    use test_utils::strategies::{
        MAX_SAMPLE_LEN, finite_paired_samples, finite_sample, nonfinite_value,
        positive_power_of_two_scale,
    };

    use super::*;
    use crate::moments::var;

    const NORMALIZATIONS: [Normalization; 3] = [
        Normalization::Population,
        Normalization::Sample,
        Normalization::Ddof(2),
    ];

    #[test]
    fn returns_nan_for_empty_input() {
        for normalization in NORMALIZATIONS {
            assert!(covariance(&[], &[], normalization).is_nan());
        }
    }

    #[test]
    fn returns_nan_for_fewer_than_two_pairs() {
        assert!(covariance(&[], &[], Normalization::Sample).is_nan());
        assert!(covariance(&[1.0], &[2.0], Normalization::Sample).is_nan());
    }

    #[test]
    fn rejects_nonfinite_values_in_either_sample_even_if_other_is_constant() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for bad in [[invalid, 1.0], [1.0, invalid]] {
                for good in [[2.0, 3.0], [2.0, 2.0]] {
                    assert!(covariance(&bad, &good, Normalization::Sample).is_nan());
                    assert!(covariance(&good, &bad, Normalization::Sample).is_nan());
                }
            }
        }
    }

    #[test]
    fn returns_nan_for_invalid_divisors() {
        let xs = [1.0, 2.0, 3.0];
        let ys = [2.0, 4.0, 6.0];
        let cases = [
            (Normalization::Sample, 1),
            (Normalization::Ddof(2), 1),
            (Normalization::Ddof(2), 2),
            (Normalization::Ddof(usize::MAX), 3),
        ];

        for (normalization, n) in cases {
            assert!(covariance(&xs[..n], &ys[..n], normalization).is_nan());
        }
    }

    #[test]
    fn all_normalizations_reject_nonfinite_observations() {
        for normalization in NORMALIZATIONS {
            for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let bad_samples = [
                    [invalid, 1.0, 2.0],
                    [1.0, invalid, 2.0],
                    [1.0, 2.0, invalid],
                ];
                let good_samples = [[2.0, 3.0, 4.0], [2.0; 3]];

                for bad in bad_samples {
                    for good in good_samples {
                        assert!(covariance(&bad, &good, normalization).is_nan());
                        assert!(covariance(&good, &bad, normalization).is_nan());
                    }
                }
            }
        }
    }

    #[test]
    fn population_singletons_reject_nonfinite_observations() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for valid in [0.0, 7.0, f64::MAX] {
                assert!(covariance(&[invalid], &[valid], Normalization::Population).is_nan());
                assert!(covariance(&[valid], &[invalid], Normalization::Population).is_nan());
            }
        }
    }

    #[test]
    #[should_panic(expected = "inputs must have equal lengths")]
    fn panics_on_length_mismatch() {
        let _ = covariance(&[1.0, 2.0], &[3.0, 4.0, 5.0], Normalization::Sample);
    }

    #[test]
    #[should_panic(expected = "inputs must have equal lengths")]
    fn checks_lengths_before_minimum_sample_size() {
        let _ = covariance(&[], &[1.0], Normalization::Sample);
    }

    #[test]
    #[should_panic(expected = "inputs must have equal lengths")]
    fn checks_lengths_before_invalid_normalization() {
        let _ = covariance(&[], &[1.0], Normalization::Ddof(usize::MAX));
    }

    #[test]
    fn accepts_finite_population_singletons() {
        let tiny = f64::from_bits(1);
        let values = [0.0, -0.0, tiny, -tiny, f64::MAX, -f64::MAX];

        for x in values {
            for y in values {
                assert_eq!(covariance(&[x], &[y], Normalization::Population), 0.0);
            }
        }
    }

    #[test]
    fn calculates_each_normalization_for_a_known_sample() {
        let xs = [1.0, 2.0, 3.0, 4.0];
        let ys = [2.0, 4.0, 6.0, 8.0];

        // Means are 2.5 and 5; the centered cross-product sum is 10.
        let cases = [
            (Normalization::Population, 2.5),
            (Normalization::Sample, 10.0 / 3.0),
            (Normalization::Ddof(2), 5.0),
            (Normalization::Ddof(3), 10.0),
        ];

        for (normalization, expected) in cases {
            let result = covariance(&xs, &ys, normalization);

            assert_approx_eq(result, expected, Tolerance::STRICT);
        }
    }

    #[test]
    fn checks_observation_boundaries_without_count_overflow() {
        let xs = [7.0; 4];
        let ys = [-3.0; 4];

        for n in 0..=xs.len() {
            let adjustments = [0, 1, 2, n, n + 1, usize::MAX];

            for ddof in adjustments {
                let result = covariance(&xs[..n], &ys[..n], Normalization::Ddof(ddof));

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
        let cases: &[(&[f64], &[f64])] = &[
            (&[], &[]),
            (&[1.0], &[2.0]),
            (&[1.0, 3.0], &[2.0, 6.0]),
            (&[0.0, -0.0], &[-0.0, 0.0]),
            (&[1.0, f64::NAN], &[2.0, 2.0]),
            (&[1.0, 2.0], &[2.0, f64::INFINITY]),
            (&[f64::NEG_INFINITY, 1.0], &[2.0, 3.0]),
        ];
        let aliases = [
            (Normalization::Population, Normalization::Ddof(0)),
            (Normalization::Sample, Normalization::Ddof(1)),
        ];

        for &(xs, ys) in cases {
            for (preset, explicit) in aliases {
                assert_eq!(
                    covariance(xs, ys, preset).to_bits(),
                    covariance(xs, ys, explicit).to_bits()
                );
            }
        }
    }

    #[test]
    fn population_covariance_is_finite_when_sample_covariance_overflows() {
        let xs = [-f64::MAX, f64::MAX];
        let positive = [-1.0, 1.0];
        let negative = [1.0, -1.0];

        assert_eq!(
            covariance(&xs, &positive, Normalization::Population),
            f64::MAX
        );
        assert_eq!(
            covariance(&xs, &negative, Normalization::Population),
            -f64::MAX
        );
        assert_eq!(
            covariance(&xs, &positive, Normalization::Sample),
            f64::INFINITY
        );
        assert_eq!(
            covariance(&xs, &negative, Normalization::Sample),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn calculates_covariance_with_either_sign() {
        let xs = &[1.0, 3.0];
        let cases = [
            (Normalization::Population, 2.0),
            (Normalization::Sample, 4.0),
        ];

        for (normalization, expected) in cases {
            assert_eq!(covariance(xs, &[2.0, 6.0], normalization), expected);
            assert_eq!(covariance(xs, &[6.0, 2.0], normalization), -expected);
        }
    }

    #[test]
    fn calculates_zero_covariance() {
        let xs = [-1.0, -1.0, 1.0, 1.0];
        let ys = [-1.0, 1.0, -1.0, 1.0];

        for normalization in NORMALIZATIONS {
            assert_eq!(covariance(&xs, &ys, normalization), 0.0);
        }
    }

    #[test]
    fn finite_constant_samples_have_zero_covariance() {
        let varying = [-f64::MAX, 0.0, f64::MAX];
        let tiny = f64::from_bits(1);
        let values = [0.0, -0.0, tiny, -tiny, -f64::MAX, f64::MAX];

        for normalization in NORMALIZATIONS {
            for value in values {
                let constant = [value; 3];

                assert_eq!(covariance(&constant, &varying, normalization), 0.0);
                assert_eq!(covariance(&varying, &constant, normalization), 0.0);
                assert_eq!(covariance(&constant, &constant, normalization), 0.0);
            }
        }
    }

    #[test]
    fn preserves_variation_around_large_offsets() {
        let xs = [1e16, 1e16, 1e16 + 2.0];
        let ys = [1.0, 1.0, 4.0];
        let cases = [
            (Normalization::Population, 4.0 / 3.0),
            (Normalization::Sample, 2.0),
            (Normalization::Ddof(2), 4.0),
        ];

        for (normalization, expected) in cases {
            assert_approx_eq(
                covariance(&xs, &ys, normalization),
                expected,
                Tolerance::VERY_STRICT,
            );
        }
    }

    #[test]
    fn compensates_product_rounding_before_cancellation() {
        let delta = 2.0_f64.powi(-27);
        let xs = [1.0 + delta, -1.0, -(1.0 + delta), 1.0];
        let ys = [1.0 - delta, 1.0, -(1.0 - delta), -1.0];

        // Both means are zero. The cross-product sum is -2 * delta², whereas
        // rounding each product before summing would give zero.
        let cases = [
            (Normalization::Population, -2.0_f64.powi(-55)),
            (Normalization::Sample, -2.0_f64.powi(-53) / 3.0),
            (Normalization::Ddof(2), -2.0_f64.powi(-54)),
        ];

        for (normalization, expected) in cases {
            let result = covariance(&xs, &ys, normalization);

            assert_approx_eq(result, expected, Tolerance::new(0.0, 1e-15));
        }
    }

    #[test]
    fn avoids_overflow_when_unscaled_cross_product_sum_is_too_large() {
        // Both means are zero, so the paired deviation products are:
        // [-1e308, 0, -1e308].
        //
        // Their sum, approximately -2e308, is too large in magnitude for f64.
        // Dividing by n - 1 = 2 gives -1e308, which fits.
        // We should get this finite covariance, not negative infinity.
        let xs = [-1e154, 0.0, 1e154];
        let ys = [1e154, 0.0, -1e154];

        let result = covariance(&xs, &ys, Normalization::Sample);
        let expected = -1e308;

        assert_approx_eq(result, expected, Tolerance::STRICT);
    }

    #[test]
    fn preserves_small_variation_around_huge_offsets() {
        // Start with a simple pattern:
        // covariance([0, 0, h], [0, 0, k]) = h * k / 3.
        // Adding the same base to every value does not change covariance.
        //
        // These powers of two make both the base and the changes exact in f64.
        // Each change is tiny relative to the base, but still a huge number.
        let base = 2.0_f64.powi(520);
        let x_step = 2.0_f64.powi(468);
        let y_step = 2.0_f64.powi(469);
        let xs = [base, base, base + x_step];
        let ys = [base, base, base + y_step];

        let result = covariance(&xs, &ys, Normalization::Sample);

        // The answer is 2^468 * 2^469 / 3 = 2^937 / 3, which fits.
        // Internally, each input uses a scaling unit of 2^520. Multiplying
        // those units gives 2^1040, which overflows. Restoring the units in
        // stages must still recover the finite answer.
        let expected = 2.0_f64.powi(937) / 3.0;
        assert_approx_eq(result, expected, Tolerance::STRICT);
    }

    #[test]
    fn preserves_the_answer_when_one_sample_is_huge_and_the_other_is_tiny() {
        // Covariance([a, a, a + h], [b, b, b + k]) = h * k / 3.
        // One sample is huge and the other is tiny. Their starting values
        // do not affect the answer, only their changes matter.
        let x_base = 2.0_f64.powi(1000);
        let x_step = 2.0_f64.powi(948);
        let y_base = 2.0_f64.powi(-1000);

        // MIN_POSITIVE is 2^-1022, so dividing by 2^30 gives 2^-1052.
        let y_step = f64::MIN_POSITIVE / 2.0_f64.powi(30);
        let xs = [x_base, x_base, x_base + x_step];
        let ys = [y_base, y_base, y_base + y_step];

        // The answer is 2^948 * 2^-1052 / 3 = 2^-104 / 3.
        // The scaling units cancel: 2^1000 * 2^-1000 = 1. Applying the tiny
        // unit first instead could round an intermediate result to zero.
        let expected = 2.0_f64.powi(-104) / 3.0;
        let tolerance = Tolerance::new(0.0, 1e-14);

        // Swapping the samples must preserve the answer too.
        assert_approx_eq(
            covariance(&xs, &ys, Normalization::Sample),
            expected,
            tolerance,
        );
        assert_approx_eq(
            covariance(&ys, &xs, Normalization::Sample),
            expected,
            tolerance,
        );
    }

    #[test]
    fn preserves_subnormal_result_when_scale_product_underflows() {
        let x = 2.0_f64.powi(-537);
        let y = 2.0_f64.powi(-538);
        let tiny = f64::from_bits(1);

        // The cross-product sum is 2*x*y = 2^-1074. Sample covariance is
        // exactly tiny; population covariance is half that and rounds to zero.
        let cases = [
            (Normalization::Population, 0.0),
            (Normalization::Sample, tiny),
        ];

        for (normalization, expected) in cases {
            assert_eq!(covariance(&[-x, x], &[-y, y], normalization), expected);
        }
    }

    #[test]
    fn returns_signed_infinity_when_the_final_answer_is_too_large() {
        let xs = [-f64::MAX, 0.0, f64::MAX];
        let ys = [f64::MAX, 0.0, -f64::MAX];

        for normalization in NORMALIZATIONS {
            assert_eq!(covariance(&xs, &xs, normalization), f64::INFINITY);
            assert_eq!(covariance(&xs, &ys, normalization), f64::NEG_INFINITY);
        }
    }

    #[test]
    fn returns_zero_when_the_final_answer_is_too_small() {
        // For the smallest positive f64 t, even sample covariance 2*t² is
        // too small to represent; population covariance is smaller still.
        let tiny = f64::from_bits(1);
        let xs = [-tiny, tiny];

        for normalization in [Normalization::Population, Normalization::Sample] {
            assert_eq!(covariance(&xs, &xs, normalization), 0.0);
        }
    }

    proptest! {
        #[test]
        fn is_symmetric(
            (xs, ys) in finite_paired_samples(2),
            adjustment in 0..MAX_SAMPLE_LEN,
        ) {
            let normalizations = [
                Normalization::Population,
                Normalization::Sample,
                Normalization::Ddof(adjustment % xs.len()),
            ];

            for normalization in normalizations {
                let a = covariance(&xs, &ys, normalization).to_bits();
                let b = covariance(&ys, &xs, normalization).to_bits();

                prop_assert_eq!(a, b);
            }
        }

        #[test]
        fn agrees_with_variance_for_identical_samples(
            xs in finite_sample(2),
            adjustment in 0..MAX_SAMPLE_LEN,
        ) {
            let normalizations = [
                Normalization::Population,
                Normalization::Sample,
                Normalization::Ddof(adjustment % xs.len()),
            ];

            for normalization in normalizations {
                let c = covariance(&xs, &xs, normalization);
                let v = var(&xs, normalization);

                prop_assert!(approx_eq(c, v, Tolerance::STRICT));
            }
        }

        #[test]
        fn preserves_covariance_under_joint_reversal(
            (xs, ys) in finite_paired_samples(2),
            adjustment in 0..MAX_SAMPLE_LEN,
        ) {
            let normalizations = [
                Normalization::Population,
                Normalization::Sample,
                Normalization::Ddof(adjustment % xs.len()),
            ];
            let reversed_x: Vec<_> = xs.iter().rev().copied().collect();
            let reversed_y: Vec<_> = ys.iter().rev().copied().collect();

            for normalization in normalizations {
                let expected = covariance(&xs, &ys, normalization);
                let reversed = covariance(&reversed_x, &reversed_y, normalization);

                prop_assert!(approx_eq(reversed, expected, Tolerance::STRICT));
            }
        }

        #[test]
        fn scales_with_both_variables(
            (xs, ys) in finite_paired_samples(2),
            x_scale in positive_power_of_two_scale(),
            y_scale in positive_power_of_two_scale(),
            adjustment in 0..MAX_SAMPLE_LEN,
        ) {
            let scaled_x: Vec<_> = xs.iter().map(|x| x * x_scale).collect();
            let scaled_y: Vec<_> = ys.iter().map(|y| -y * y_scale).collect();
            let normalizations = [
                Normalization::Population,
                Normalization::Sample,
                Normalization::Ddof(adjustment % xs.len()),
            ];

            for normalization in normalizations {
                let scaled_result = covariance(&scaled_x, &scaled_y, normalization);
                let expected = -(covariance(&xs, &ys, normalization) * x_scale) * y_scale;

                prop_assert!(approx_eq(scaled_result, expected, Tolerance::STRICT));
            }
        }

        #[test]
        fn is_invariant_to_exact_translations(
            pairs in prop::collection::vec((-10_000_i32..10_000, -10_000_i32..10_000), 2..=MAX_SAMPLE_LEN),
            x_offset in -100_i32..100,
            y_offset in -100_i32..100,
            adjustment in 0..MAX_SAMPLE_LEN,
        ) {
            let (xs, ys): (Vec<_>, Vec<_>) = pairs.iter().map(|&(x, y)| (f64::from(x), f64::from(y))).unzip();
            let translated_x: Vec<_> = xs.iter().map(|x| x + f64::from(x_offset)).collect();
            let translated_y: Vec<_> = ys.iter().map(|y| y + f64::from(y_offset)).collect();
            let normalizations = [
                Normalization::Population,
                Normalization::Sample,
                Normalization::Ddof(adjustment % xs.len()),
            ];

            for normalization in normalizations {
                let a = covariance(&xs, &ys, normalization);
                let b = covariance(&translated_x, &translated_y, normalization);

                prop_assert!(approx_eq(a, b, Tolerance::DEFAULT));
            }
        }

        #[test]
        fn presets_match_explicit_ddof(
            (xs, ys) in finite_paired_samples(1),
        ) {
            let aliases = [
                (Normalization::Population, Normalization::Ddof(0)),
                (Normalization::Sample, Normalization::Ddof(1)),
            ];

            for (preset, explicit) in aliases {
                let a = covariance(&xs, &ys, preset).to_bits();
                let b = covariance(&xs, &ys, explicit).to_bits();

                prop_assert_eq!(a, b);
            }
        }

        #[test]
        fn nonfinite_observation_invalidates_either_sample(
            (mut xs, mut ys) in finite_paired_samples(1),
            value in nonfinite_value(),
            position in any::<usize>(),
            invalid_in_x in any::<bool>(),
            adjustment in 0..MAX_SAMPLE_LEN,
        ) {
            let index = position % xs.len();
            let normalizations = [
                Normalization::Population,
                Normalization::Sample,
                Normalization::Ddof(adjustment % xs.len()),
            ];

            if invalid_in_x {
                xs[index] = value;
            } else {
                ys[index] = value;
            }

            for normalization in normalizations {
                prop_assert!(covariance(&xs, &ys, normalization).is_nan());
            }
        }
    }
}
