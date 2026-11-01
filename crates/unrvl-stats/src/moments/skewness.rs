use unrvl_numerics::convert::usize_to_f64;
use unrvl_numerics::summation::NeumaierSum;

use crate::internal::deviations::CenteredSample;
use crate::moments::ShapeEstimator;

/// Returns skewness with the selected Joanes-Gill estimator.
///
/// The statistic is defined as:
///
/// ```text
/// mᵣ = ∑ (xᵢ - x̄)ʳ / n
/// s² = ∑ (xᵢ - x̄)² / (n - 1)
///
/// Type 1: g1 = m3 / m2^(3/2)
/// Type 2: G1 = g1 · sqrt(n(n - 1)) / (n - 2)
/// Type 3: b1 = m3 / s³
/// ```
///
/// Here `n` is the number of observations, `x̄` is their arithmetic mean, and
/// `s` is the sample standard deviation using Bessel's correction. Type 2 is
/// the adjusted Fisher-Pearson coefficient; Type 3 uses the sample standard
/// deviation to standardize the third central moment.
///
/// Types 1 and 3 require at least two observations; Type 2 requires at least
/// three. Returns `NaN` for fewer observations, a constant sample, or any
/// non-finite observation. Two distinct observations have zero skewness for
/// Types 1 and 3.
///
/// Uses scaling, corrected centering, and compensated summation to reduce
/// numerical error. Rounding remains possible, and observations that are very
/// small relative to the largest magnitude can underflow during scaling.
///
/// For `n` observations, uses `O(n)` time and `O(1)` additional memory.
///
/// # Panics
///
/// Panics if a finite sample contains more than `2^53` observations.
///
/// # Examples
///
/// ```
/// # use unrvl_stats::moments::{ShapeEstimator, skewness};
/// let xs = [1.0, 3.0];
/// assert_eq!(skewness(&xs, ShapeEstimator::Type1), 0.0);
/// assert_eq!(skewness(&xs, ShapeEstimator::Type3), 0.0);
/// assert!(skewness(&xs, ShapeEstimator::Type2).is_nan());
/// ```
#[must_use]
pub fn skewness(xs: &[f64], estimator: ShapeEstimator) -> f64 {
    let minimum = match estimator {
        ShapeEstimator::Type1 | ShapeEstimator::Type3 => 2,
        ShapeEstimator::Type2 => 3,
    };

    if xs.len() < minimum {
        return f64::NAN;
    }

    let Some(sample) = CenteredSample::new(xs) else {
        return f64::NAN;
    };

    let mut sum_u2 = NeumaierSum::new();
    let mut sum_u3 = NeumaierSum::new();

    for u in sample.deviations() {
        let u2 = u * u;

        sum_u2 += u2;
        sum_u3 += u2 * u;
    }

    let sum_u2 = sum_u2.total();
    let sum_u3 = sum_u3.total();

    if sum_u2 == 0.0 {
        return f64::NAN;
    }

    let n = usize_to_f64(xs.len());

    match estimator {
        ShapeEstimator::Type1 => {
            let scaled_std_dev = (sum_u2 / n).sqrt();

            (sum_u3 / n) / scaled_std_dev.powi(3)
        }
        ShapeEstimator::Type2 => {
            let correction = n / ((n - 1.0) * (n - 2.0));
            let scaled_std_dev = (sum_u2 / (n - 1.0)).sqrt();

            correction * sum_u3 / scaled_std_dev.powi(3)
        }
        ShapeEstimator::Type3 => {
            let scaled_std_dev = (sum_u2 / (n - 1.0)).sqrt();

            (sum_u3 / n) / scaled_std_dev.powi(3)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_1_SQRT_2;

    use proptest::prelude::*;
    use test_utils::approx::{Tolerance, approx_eq, assert_approx_eq};
    use test_utils::strategies::{
        exact_translation_case, finite_nonconstant_sample, nonfinite_value,
        positive_power_of_two_scale,
    };

    use super::*;

    const ESTIMATORS: [ShapeEstimator; 3] = [
        ShapeEstimator::Type1,
        ShapeEstimator::Type2,
        ShapeEstimator::Type3,
    ];

    // For [a, a, b] with b > a, the three coefficients have these exact forms.
    fn three_point_references() -> [(ShapeEstimator, f64); 3] {
        [
            (ShapeEstimator::Type1, FRAC_1_SQRT_2),
            (ShapeEstimator::Type2, 3.0_f64.sqrt()),
            (ShapeEstimator::Type3, 2.0 / (3.0 * 3.0_f64.sqrt())),
        ]
    }

    #[test]
    fn returns_nan_for_empty_input() {
        let xs = &[];

        for estimator in ESTIMATORS {
            let result = skewness(xs, estimator);

            assert!(result.is_nan());
        }
    }

    #[test]
    fn returns_nan_for_singleton() {
        let xs = &[1.0];

        for estimator in ESTIMATORS {
            let result = skewness(xs, estimator);

            assert!(result.is_nan());
        }
    }

    #[test]
    fn returns_nan_for_two_observations_with_type_2() {
        let xs = &[1.0, 2.0];
        let result = skewness(xs, ShapeEstimator::Type2);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_nan_input() {
        let xs = &[1.0, 2.0, f64::NAN];

        for estimator in ESTIMATORS {
            let result = skewness(xs, estimator);

            assert!(result.is_nan());
        }
    }

    #[test]
    fn returns_nan_for_positive_infinity() {
        let xs = &[1.0, 2.0, f64::INFINITY];

        for estimator in ESTIMATORS {
            let result = skewness(xs, estimator);

            assert!(result.is_nan());
        }
    }

    #[test]
    fn returns_nan_for_negative_infinity() {
        let xs = &[1.0, 2.0, f64::NEG_INFINITY];

        for estimator in ESTIMATORS {
            let result = skewness(xs, estimator);

            assert!(result.is_nan());
        }
    }

    #[test]
    fn returns_nan_for_nonfinite_two_observation_samples() {
        for estimator in ESTIMATORS {
            for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                for xs in [[value, 1.0], [1.0, value]] {
                    assert!(skewness(&xs, estimator).is_nan());
                }
            }
        }
    }

    #[test]
    fn returns_nan_for_zero_variance() {
        let xs = &[1.0, 1.0, 1.0];

        for estimator in ESTIMATORS {
            let result = skewness(xs, estimator);

            assert!(result.is_nan());
        }
    }

    #[test]
    fn returns_nan_for_constant_extreme_samples() {
        let values = [
            0.0,
            -0.0,
            f64::from_bits(1),
            -f64::from_bits(1),
            f64::MAX,
            -f64::MAX,
        ];

        for estimator in ESTIMATORS {
            for value in values {
                let xs = [value; 4];

                for n in 2..=xs.len() {
                    assert!(skewness(&xs[..n], estimator).is_nan());
                }
            }
        }
    }

    #[test]
    fn returns_nan_for_translation_that_rounds_to_a_constant() {
        let e = f64::EPSILON;
        let xs = [1.0, 1.0 + e, 2.0f64.mul_add(e, 1.0)];
        let translated = xs.map(|x| x + 100.0);

        // The translation erased the distinctions between observations.
        assert_eq!(translated, [101.0; 3]);

        for estimator in ESTIMATORS {
            assert_approx_eq(skewness(&xs, estimator), 0.0, Tolerance::STRICT);
            assert!(skewness(&translated, estimator).is_nan());
        }
    }

    #[test]
    fn accepts_two_observations_for_types_1_and_3() {
        let tiny = f64::from_bits(1);
        let samples = [
            [1.0, 3.0],
            [-f64::MAX, f64::MAX],
            [1e16, 1e16 + 2.0],
            [0.0, tiny],
        ];

        for estimator in [ShapeEstimator::Type1, ShapeEstimator::Type3] {
            for xs in samples {
                assert_eq!(skewness(&xs, estimator), 0.0);
            }
        }
    }

    #[test]
    fn checks_estimator_minimum_observations() {
        let xs = [-1.0, 0.0, 1.0, 2.0];

        let cases = [
            (ShapeEstimator::Type1, 2),
            (ShapeEstimator::Type2, 3),
            (ShapeEstimator::Type3, 2),
        ];

        for (estimator, minimum) in cases {
            for n in 0..=xs.len() {
                let result = skewness(&xs[..n], estimator);

                if n < minimum {
                    assert!(result.is_nan());
                } else {
                    assert_approx_eq(result, 0.0, Tolerance::VERY_STRICT);
                }
            }
        }
    }

    #[test]
    fn is_zero_for_symmetric_sample() {
        let xs = &[1.0, 2.0, 3.0, 4.0, 5.0];

        for estimator in ESTIMATORS {
            let result = skewness(xs, estimator);

            assert_approx_eq(result, 0.0, Tolerance::VERY_STRICT);
        }
    }

    #[test]
    fn matches_known_right_skewed_sample() {
        let xs = &[1.0, 1.0, 2.0];

        for (estimator, expected) in three_point_references() {
            let result = skewness(xs, estimator);

            assert_approx_eq(result, expected, Tolerance::VERY_STRICT);
        }
    }

    #[test]
    fn matches_known_left_skewed_sample() {
        let xs = &[1.0, 2.0, 2.0];

        for (estimator, expected) in three_point_references() {
            let result = skewness(xs, estimator);

            assert_approx_eq(result, -expected, Tolerance::VERY_STRICT);
        }
    }

    #[test]
    fn remains_finite_at_large_scale() {
        let xs = [1.0, 1.0, 2.0, 4.0, 10.0];
        let scaled = xs.map(|x| x * 1e110);

        for estimator in ESTIMATORS {
            let result = skewness(&xs, estimator);
            let result_scaled = skewness(&scaled, estimator);

            assert!(result_scaled.is_finite());
            assert_approx_eq(result_scaled, result, Tolerance::DEFAULT);
        }
    }

    #[test]
    fn remains_finite_at_small_scale() {
        let xs = [1.0, 1.0, 2.0, 4.0, 10.0];
        let scaled = xs.map(|x| x * 1e-110);

        for estimator in ESTIMATORS {
            let result = skewness(&xs, estimator);
            let result_scaled = skewness(&scaled, estimator);

            assert!(result_scaled.is_finite());
            assert_approx_eq(result_scaled, result, Tolerance::DEFAULT);
        }
    }

    #[test]
    fn preserves_shape_at_the_smallest_positive_subnormal_scale() {
        let d = f64::from_bits(1);
        let xs = [0.0, 0.0, d];

        for (estimator, expected) in three_point_references() {
            let result = skewness(&xs, estimator);

            assert_approx_eq(result, expected, Tolerance::STRICT);
        }
    }

    #[test]
    fn preserves_shape_at_the_smallest_negative_subnormal_scale() {
        let d = f64::from_bits(1);
        let xs = [0.0, 0.0, -d];

        for (estimator, expected) in three_point_references() {
            let result = skewness(&xs, estimator);

            assert_approx_eq(result, -expected, Tolerance::STRICT);
        }
    }

    #[test]
    fn remains_finite_when_the_unscaled_sum_would_overflow() {
        let half_max = f64::MAX / 2.0;
        let xs = [half_max, half_max, f64::MAX];

        for (estimator, expected) in three_point_references() {
            let result = skewness(&xs, estimator);

            assert_approx_eq(result, expected, Tolerance::STRICT);
        }
    }

    #[test]
    fn preserves_shape_around_a_large_offset() {
        let xs = [1e16, 1e16, 1e16 + 2.0];

        for (estimator, expected) in three_point_references() {
            let result = skewness(&xs, estimator);

            assert_approx_eq(result, expected, Tolerance::STRICT);
        }
    }

    proptest! {
        #[test]
        fn skewness_is_translation_invariant(
            (sample, offset) in exact_translation_case(3)
        ) {
            let translated: Vec<_> = sample.iter().map(|&x| x + offset).collect();

            for estimator in ESTIMATORS {
                let x = skewness(&sample, estimator);
                let y = skewness(&translated, estimator);

                prop_assert!(approx_eq(x, y, Tolerance::DEFAULT));
            }
        }

        #[test]
        fn skewness_is_positive_scale_invariant(
            sample in finite_nonconstant_sample(3),
            scale in positive_power_of_two_scale(),
        ) {
            let scaled: Vec<_> = sample.iter().map(|&x| x * scale).collect();

            for estimator in ESTIMATORS {
                let x = skewness(&sample, estimator);
                let y = skewness(&scaled, estimator);

                prop_assert!(approx_eq(x, y, Tolerance::DEFAULT));
            }
        }

        #[test]
        fn skewness_flips_sign_under_negative_scaling(
            sample in finite_nonconstant_sample(3),
            scale in positive_power_of_two_scale(),
        ) {
            let reflected: Vec<_> = sample.iter().map(|&x| x * -scale).collect();

            for estimator in ESTIMATORS {
                let x = skewness(&sample, estimator);
                let y = skewness(&reflected, estimator);

                prop_assert!(approx_eq(y, -x, Tolerance::DEFAULT));
            }
        }

        #[test]
        fn skewness_is_reversal_invariant(sample in finite_nonconstant_sample(3)) {
            let mut reversed = sample.clone();
            reversed.reverse();

            for estimator in ESTIMATORS {
                let x = skewness(&sample, estimator);
                let y = skewness(&reversed, estimator);

                prop_assert!(approx_eq(x, y, Tolerance::DEFAULT));
            }
        }

        #[test]
        fn skewness_is_finite_for_finite_nonconstant_samples(
            sample in finite_nonconstant_sample(3),
        ) {
            for estimator in ESTIMATORS {
                prop_assert!(skewness(&sample, estimator).is_finite());
            }
        }

        #[test]
        fn skewness_rejects_nonfinite_observations_at_any_position(
            mut sample in finite_nonconstant_sample(3),
            value in nonfinite_value(),
            position in any::<usize>(),
        ) {
            let index = position % sample.len();
            sample[index] = value;

            for estimator in ESTIMATORS {
                prop_assert!(skewness(&sample, estimator).is_nan());
            }
        }
    }
}
