use unrvl_numerics::convert::usize_to_f64;
use unrvl_numerics::summation::NeumaierSum;

use crate::internal::deviations::CenteredSample;
use crate::moments::ShapeEstimator;

/// Reporting convention for [`kurtosis`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kurtosis {
    /// Pearson convention, with a normal-distribution reference of three.
    Pearson,

    /// Excess convention, subtracting three for a normal-distribution reference
    /// of zero.
    Excess,
}

/// Returns kurtosis with the selected Joanes-Gill estimator and convention.
///
/// Let `n` be the number of observations and `x̄` their arithmetic mean:
///
/// ```text
/// mᵣ = ∑ (xᵢ - x̄)ʳ / n
/// s² = ∑ (xᵢ - x̄)² / (n - 1)
///
/// Type 1: g2 = m4 / m2² - 3
/// Type 2: G2 = ((n + 1) · g2 + 6) · (n - 1) / ((n - 2)(n - 3))
/// Type 3: b2 = m4 / s⁴ - 3
/// ```
///
/// These formulas define [`Kurtosis::Excess`]. [`Kurtosis::Pearson`] adds three
/// to the corresponding estimate, including the finite-sample adjustment for
/// Type 2. Their normal-distribution references are zero and three,
/// respectively; finite samples need not produce those values. Adjusted Type 2
/// Pearson kurtosis can be negative.
///
/// Types 1 and 3 require at least two observations; Type 2 requires at least
/// four. Both conventions return `NaN` for fewer observations, a constant
/// sample, or any non-finite observation. Two distinct observations give excess
/// kurtosis `-2` for Type 1 and `-2.75` for Type 3, or Pearson kurtosis `1` and
/// `0.25`, respectively.
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
/// use unrvl_stats::moments::{Kurtosis, ShapeEstimator, kurtosis};
///
/// let xs = [1.0, 2.0, 3.0, 6.0];
/// let excess = kurtosis(&xs, ShapeEstimator::Type1, Kurtosis::Excess);
/// let pearson = kurtosis(&xs, ShapeEstimator::Type1, Kurtosis::Pearson);
/// assert!((excess + 1.0).abs() < 1e-14);
/// assert!((pearson - 2.0).abs() < 1e-14);
/// ```
#[must_use]
pub fn kurtosis(xs: &[f64], estimator: ShapeEstimator, convention: Kurtosis) -> f64 {
    let minimum = match estimator {
        ShapeEstimator::Type1 | ShapeEstimator::Type3 => 2,
        ShapeEstimator::Type2 => 4,
    };

    if xs.len() < minimum {
        return f64::NAN;
    }

    let Some(sample) = CenteredSample::new(xs) else {
        return f64::NAN;
    };

    let mut sum_u2 = NeumaierSum::new();
    let mut sum_u4 = NeumaierSum::new();

    for u in sample.deviations() {
        let u2 = u * u;

        sum_u2 += u2;
        sum_u4 += u2 * u2;
    }

    let sum_u2 = sum_u2.total();
    let sum_u4 = sum_u4.total();

    if sum_u2 == 0.0 {
        return f64::NAN;
    }

    let n = usize_to_f64(xs.len());
    let n_minus_1 = n - 1.0;

    let scaled_var = match estimator {
        ShapeEstimator::Type1 => sum_u2 / n,
        ShapeEstimator::Type2 | ShapeEstimator::Type3 => sum_u2 / n_minus_1,
    };

    let sum_z4 = sum_u4 / scaled_var.powi(2);

    match estimator {
        ShapeEstimator::Type1 | ShapeEstimator::Type3 => {
            let pearson = sum_z4 / n;

            match convention {
                Kurtosis::Pearson => pearson,
                Kurtosis::Excess => pearson - 3.0,
            }
        }
        ShapeEstimator::Type2 => {
            let n_minus_2 = n - 2.0;
            let n_minus_3 = n - 3.0;

            let bias_factor = n * (n + 1.0) / (n_minus_1 * n_minus_2 * n_minus_3);
            let excess_correction = 3.0 * n_minus_1.powi(2) / (n_minus_2 * n_minus_3);
            let excess = bias_factor.mul_add(sum_z4, -excess_correction);

            match convention {
                Kurtosis::Pearson => excess + 3.0,
                Kurtosis::Excess => excess,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_utils::approx::{Tolerance, approx_eq, assert_approx_eq};
    use test_utils::strategies::{
        exact_translation_case, finite_nonconstant_sample, nonfinite_value,
        positive_power_of_two_scale,
    };

    use super::*;

    const SELECTIONS: [(ShapeEstimator, Kurtosis); 6] = [
        (ShapeEstimator::Type1, Kurtosis::Excess),
        (ShapeEstimator::Type2, Kurtosis::Excess),
        (ShapeEstimator::Type3, Kurtosis::Excess),
        (ShapeEstimator::Type1, Kurtosis::Pearson),
        (ShapeEstimator::Type2, Kurtosis::Pearson),
        (ShapeEstimator::Type3, Kurtosis::Pearson),
    ];

    // For [a, a, a, b] with distinct a and b, the coefficients have these
    // forms.
    fn four_point_references() -> [(ShapeEstimator, Kurtosis, f64); 6] {
        [
            (ShapeEstimator::Type1, Kurtosis::Excess, -2.0 / 3.0),
            (ShapeEstimator::Type2, Kurtosis::Excess, 4.0),
            (ShapeEstimator::Type3, Kurtosis::Excess, -27.0 / 16.0),
            (ShapeEstimator::Type1, Kurtosis::Pearson, 7.0 / 3.0),
            (ShapeEstimator::Type2, Kurtosis::Pearson, 7.0),
            (ShapeEstimator::Type3, Kurtosis::Pearson, 21.0 / 16.0),
        ]
    }

    #[test]
    fn returns_nan_for_empty_input() {
        let xs = &[];

        for (estimator, convention) in SELECTIONS {
            let result = kurtosis(xs, estimator, convention);

            assert!(result.is_nan());
        }
    }

    #[test]
    fn returns_nan_for_singleton() {
        let xs = &[1.0];

        for (estimator, convention) in SELECTIONS {
            let result = kurtosis(xs, estimator, convention);

            assert!(result.is_nan());
        }
    }

    #[test]
    fn returns_nan_for_two_observations_with_type_2() {
        let xs = &[1.0, 2.0];

        for convention in [Kurtosis::Pearson, Kurtosis::Excess] {
            let result = kurtosis(xs, ShapeEstimator::Type2, convention);

            assert!(result.is_nan());
        }
    }

    #[test]
    fn returns_nan_for_three_observations_with_type_2() {
        let xs = &[1.0, 2.0, 3.0];

        for convention in [Kurtosis::Pearson, Kurtosis::Excess] {
            let result = kurtosis(xs, ShapeEstimator::Type2, convention);

            assert!(result.is_nan());
        }
    }

    #[test]
    fn returns_nan_for_nan_input() {
        let xs = &[1.0, 2.0, 3.0, f64::NAN];

        for (estimator, convention) in SELECTIONS {
            let result = kurtosis(xs, estimator, convention);

            assert!(result.is_nan());
        }
    }

    #[test]
    fn returns_nan_for_positive_infinity() {
        let xs = &[1.0, 2.0, 3.0, f64::INFINITY];

        for (estimator, convention) in SELECTIONS {
            let result = kurtosis(xs, estimator, convention);

            assert!(result.is_nan());
        }
    }

    #[test]
    fn returns_nan_for_negative_infinity() {
        let xs = &[1.0, 2.0, 3.0, f64::NEG_INFINITY];

        for (estimator, convention) in SELECTIONS {
            let result = kurtosis(xs, estimator, convention);

            assert!(result.is_nan());
        }
    }

    #[test]
    fn returns_nan_for_nonfinite_short_samples() {
        for (estimator, convention) in SELECTIONS {
            for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                for n in 2..=3 {
                    for index in 0..n {
                        let mut xs = [1.0, 2.0, 3.0];
                        xs[index] = value;

                        assert!(kurtosis(&xs[..n], estimator, convention).is_nan());
                    }
                }
            }
        }
    }

    #[test]
    fn returns_nan_for_zero_variance() {
        let xs = &[1.0, 1.0, 1.0, 1.0];

        for (estimator, convention) in SELECTIONS {
            let result = kurtosis(xs, estimator, convention);

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

        for (estimator, convention) in SELECTIONS {
            for value in values {
                let xs = [value; 4];

                for n in 2..=xs.len() {
                    assert!(kurtosis(&xs[..n], estimator, convention).is_nan());
                }
            }
        }
    }

    #[test]
    fn returns_nan_for_translation_that_rounds_to_a_constant() {
        let e = f64::EPSILON;
        let xs = [1.0, 1.0 + e, 2.0f64.mul_add(e, 1.0), 3.0f64.mul_add(e, 1.0)];
        let translated = xs.map(|x| x + 100.0);

        // The translation erased the distinctions between observations.
        assert_eq!(translated, [101.0; 4]);

        let cases = [
            (ShapeEstimator::Type1, -34.0 / 25.0, 41.0 / 25.0),
            (ShapeEstimator::Type2, -1.2, 1.8),
            (ShapeEstimator::Type3, -831.0 / 400.0, 369.0 / 400.0),
        ];

        for (estimator, excess, pearson) in cases {
            let expectations = [(Kurtosis::Excess, excess), (Kurtosis::Pearson, pearson)];

            for (convention, expected) in expectations {
                assert_approx_eq(
                    kurtosis(&xs, estimator, convention),
                    expected,
                    Tolerance::STRICT,
                );
                assert!(kurtosis(&translated, estimator, convention).is_nan());
            }
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

        let cases = [
            (ShapeEstimator::Type1, Kurtosis::Excess, -2.0),
            (ShapeEstimator::Type3, Kurtosis::Excess, -2.75),
            (ShapeEstimator::Type1, Kurtosis::Pearson, 1.0),
            (ShapeEstimator::Type3, Kurtosis::Pearson, 0.25),
        ];

        for (estimator, convention, expected) in cases {
            for xs in samples {
                assert_eq!(kurtosis(&xs, estimator, convention), expected);
            }
        }
    }

    #[test]
    fn matches_known_three_observation_sample() {
        let xs = &[1.0, 2.0, 3.0];

        let cases = [
            (ShapeEstimator::Type1, Kurtosis::Excess, -1.5),
            (ShapeEstimator::Type3, Kurtosis::Excess, -7.0 / 3.0),
            (ShapeEstimator::Type1, Kurtosis::Pearson, 1.5),
            (ShapeEstimator::Type3, Kurtosis::Pearson, 2.0 / 3.0),
        ];

        for (estimator, convention, expected) in cases {
            let result = kurtosis(xs, estimator, convention);

            assert_approx_eq(result, expected, Tolerance::VERY_STRICT);
        }
    }

    #[test]
    fn checks_estimator_minimum_observations() {
        let xs = [-1.0, 0.0, 1.0, 2.0];

        let cases = [
            (ShapeEstimator::Type1, 2),
            (ShapeEstimator::Type2, 4),
            (ShapeEstimator::Type3, 2),
        ];

        for (estimator, minimum) in cases {
            for convention in [Kurtosis::Pearson, Kurtosis::Excess] {
                for n in 0..=xs.len() {
                    let result = kurtosis(&xs[..n], estimator, convention);

                    if n < minimum {
                        assert!(result.is_nan());
                    } else {
                        assert!(result.is_finite());
                    }
                }
            }
        }
    }

    #[test]
    fn matches_known_symmetric_sample() {
        let xs = &[1.0, 2.0, 3.0, 4.0, 5.0];

        let cases = [
            (ShapeEstimator::Type1, -1.3, 1.7),
            (ShapeEstimator::Type2, -1.2, 1.8),
            (ShapeEstimator::Type3, -1.912, 1.088),
        ];

        for (estimator, excess, pearson) in cases {
            let expectations = [(Kurtosis::Excess, excess), (Kurtosis::Pearson, pearson)];

            for (convention, expected) in expectations {
                let result = kurtosis(xs, estimator, convention);

                assert_approx_eq(result, expected, Tolerance::VERY_STRICT);
            }
        }
    }

    #[test]
    fn matches_known_high_kurtosis_sample() {
        let xs = &[1.0, 1.0, 1.0, 2.0];

        for (estimator, convention, expected) in four_point_references() {
            let result = kurtosis(xs, estimator, convention);

            assert_approx_eq(result, expected, Tolerance::VERY_STRICT);
        }
    }

    #[test]
    fn matches_known_low_kurtosis_sample() {
        let xs = &[1.0, 1.0, 2.0, 2.0];

        // The adjusted Type 2 Pearson estimate is valid even though it is
        // negative.
        let cases = [
            (ShapeEstimator::Type1, -2.0, 1.0),
            (ShapeEstimator::Type2, -6.0, -3.0),
            (ShapeEstimator::Type3, -39.0 / 16.0, 9.0 / 16.0),
        ];

        for (estimator, excess, pearson) in cases {
            let expectations = [(Kurtosis::Excess, excess), (Kurtosis::Pearson, pearson)];

            for (convention, expected) in expectations {
                let result = kurtosis(xs, estimator, convention);

                assert_approx_eq(result, expected, Tolerance::VERY_STRICT);
            }
        }
    }

    #[test]
    fn remains_finite_at_large_scale() {
        let xs = [1.0, 1.0, 2.0, 4.0, 10.0];
        let scaled = xs.map(|x| x * 1e110);

        for (estimator, convention) in SELECTIONS {
            let result = kurtosis(&xs, estimator, convention);
            let result_scaled = kurtosis(&scaled, estimator, convention);

            assert!(result_scaled.is_finite());
            assert_approx_eq(result_scaled, result, Tolerance::DEFAULT);
        }
    }

    #[test]
    fn remains_finite_at_small_scale() {
        let xs = [1.0, 1.0, 2.0, 4.0, 10.0];
        let scaled = xs.map(|x| x * 1e-110);

        for (estimator, convention) in SELECTIONS {
            let result = kurtosis(&xs, estimator, convention);
            let result_scaled = kurtosis(&scaled, estimator, convention);

            assert!(result_scaled.is_finite());
            assert_approx_eq(result_scaled, result, Tolerance::DEFAULT);
        }
    }

    #[test]
    fn preserves_shape_at_the_smallest_positive_subnormal_scale() {
        let d = f64::from_bits(1);
        let xs = [0.0, 0.0, 0.0, d];

        for (estimator, convention, expected) in four_point_references() {
            let result = kurtosis(&xs, estimator, convention);

            assert_approx_eq(result, expected, Tolerance::STRICT);
        }
    }

    #[test]
    fn preserves_shape_at_the_smallest_negative_subnormal_scale() {
        let d = f64::from_bits(1);
        let xs = [0.0, 0.0, 0.0, -d];

        for (estimator, convention, expected) in four_point_references() {
            let result = kurtosis(&xs, estimator, convention);

            assert_approx_eq(result, expected, Tolerance::STRICT);
        }
    }

    #[test]
    fn remains_finite_when_the_unscaled_sum_would_overflow() {
        let half_max = f64::MAX / 2.0;
        let xs = [half_max, half_max, half_max, f64::MAX];

        for (estimator, convention, expected) in four_point_references() {
            let result = kurtosis(&xs, estimator, convention);

            assert_approx_eq(result, expected, Tolerance::STRICT);
        }
    }

    #[test]
    fn preserves_shape_around_a_large_offset() {
        let xs = [1e16, 1e16, 1e16, 1e16 + 2.0];

        for (estimator, convention, expected) in four_point_references() {
            let result = kurtosis(&xs, estimator, convention);

            assert_approx_eq(result, expected, Tolerance::STRICT);
        }
    }

    proptest! {
        #[test]
        fn kurtosis_is_translation_invariant(
            (sample, offset) in exact_translation_case(4)
        ) {
            let translated: Vec<_> = sample.iter().map(|&x| x + offset).collect();

            for (estimator, convention) in SELECTIONS {
                let x = kurtosis(&sample, estimator, convention);
                let y = kurtosis(&translated, estimator, convention);

                prop_assert!(approx_eq(x, y, Tolerance::DEFAULT));
            }
        }

        #[test]
        fn kurtosis_is_positive_scale_invariant(
            sample in finite_nonconstant_sample(4),
            scale in positive_power_of_two_scale(),
        ) {
            let scaled: Vec<_> = sample.iter().map(|&x| x * scale).collect();

            for (estimator, convention) in SELECTIONS {
                let x = kurtosis(&sample, estimator, convention);
                let y = kurtosis(&scaled, estimator, convention);

                prop_assert!(approx_eq(x, y, Tolerance::DEFAULT));
            }
        }

        #[test]
        fn kurtosis_is_negative_scale_invariant(
            sample in finite_nonconstant_sample(4),
            scale in positive_power_of_two_scale(),
        ) {
            let reflected: Vec<_> = sample.iter().map(|&x| x * -scale).collect();

            for (estimator, convention) in SELECTIONS {
                let x = kurtosis(&sample, estimator, convention);
                let y = kurtosis(&reflected, estimator, convention);

                prop_assert!(approx_eq(x, y, Tolerance::DEFAULT));
            }
        }

        #[test]
        fn kurtosis_is_reversal_invariant(
            sample in finite_nonconstant_sample(4),
        ) {
            let mut reversed = sample.clone();
            reversed.reverse();

            for (estimator, convention) in SELECTIONS {
                let x = kurtosis(&sample, estimator, convention);
                let y = kurtosis(&reversed, estimator, convention);

                prop_assert!(approx_eq(x, y, Tolerance::DEFAULT));
            }
        }

        #[test]
        fn kurtosis_is_finite_for_finite_nonconstant_samples(
            sample in finite_nonconstant_sample(4),
        ) {
            for (estimator, convention) in SELECTIONS {
                prop_assert!(kurtosis(&sample, estimator, convention).is_finite());
            }
        }

        #[test]
        fn kurtosis_conventions_differ_by_three(
            sample in finite_nonconstant_sample(4),
        ) {
            for estimator in [ShapeEstimator::Type1, ShapeEstimator::Type2, ShapeEstimator::Type3] {
                let pearson = kurtosis(&sample, estimator, Kurtosis::Pearson);
                let excess = kurtosis(&sample, estimator, Kurtosis::Excess);

                prop_assert!(approx_eq(pearson, excess + 3.0, Tolerance::STRICT));
            }
        }

        #[test]
        fn kurtosis_rejects_nonfinite_observations_at_any_position(
            mut sample in finite_nonconstant_sample(4),
            value in nonfinite_value(),
            position in any::<usize>(),
        ) {
            let index = position % sample.len();
            sample[index] = value;

            for (estimator, convention) in SELECTIONS {
                prop_assert!(kurtosis(&sample, estimator, convention).is_nan());
            }
        }
    }
}
