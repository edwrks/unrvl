use unrvl_numerics::summation::NeumaierSum;

use crate::internal::deviations::CenteredSample;

/// Returns the Pearson correlation of paired observations.
///
/// Measures linear association, with values in `[-1, 1]`. The statistic is:
///
/// ```text
///                   ∑ (xᵢ - x̄)(yᵢ - ȳ)
/// r = -----------------------------------------------
///       sqrt(∑ (xᵢ - x̄)²) · sqrt(∑ (yᵢ - ȳ)²)
/// ```
///
/// Observations are paired by position. Returns `NaN` if fewer than two pairs
/// are provided, any observation is non-finite, or either sample is constant.
/// No pairs are discarded.
///
/// Uses separate power-of-two scales, corrected centering, and compensation
/// for product and summation rounding. The scales cancel in the ratio, so the
/// calculation can remain finite even when covariance overflows or underflows.
/// Rounding beyond `[-1, 1]` is clamped to the nearest endpoint. These measures
/// do not eliminate rounding or underflow; small contributions relative to the
/// input scales can be lost.
///
/// Uses `O(n)` time and `O(1)` additional memory, where `n` is the number of
/// paired observations.
///
/// # Panics
///
/// Panics if the slices have different lengths, or if a finite sample contains
/// more than `2^53` observations.
///
/// # Examples
///
/// ```
/// # use unrvl_stats::association::pearson;
/// let xs = [1.0, 2.0, 3.0];
/// let ys = [6.0, 4.0, 2.0];
/// assert!((pearson(&xs, &ys) + 1.0).abs() < 1e-14);
/// ```
#[must_use]
pub fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    assert_eq!(xs.len(), ys.len(), "inputs must have equal lengths");

    if xs.len() < 2 {
        return f64::NAN;
    }

    let Some(x_sample) = CenteredSample::new(xs) else {
        return f64::NAN;
    };

    let Some(y_sample) = CenteredSample::new(ys) else {
        return f64::NAN;
    };

    let mut dot = NeumaierSum::new();
    let mut sum_x2 = NeumaierSum::new();
    let mut sum_y2 = NeumaierSum::new();

    for (x, y) in x_sample.deviations().zip(y_sample.deviations()) {
        dot.add_product(x, y);
        sum_x2.add_product(x, x);
        sum_y2.add_product(y, y);
    }

    let sum_x2 = sum_x2.total();
    let sum_y2 = sum_y2.total();

    if sum_x2 == 0.0 || sum_y2 == 0.0 {
        return f64::NAN;
    }

    // Both the input scales and the sample divisor n - 1 cancel in this ratio.
    let denominator = sum_x2.sqrt() * sum_y2.sqrt();
    (dot.total() / denominator).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_utils::approx::{Tolerance, approx_eq, assert_approx_eq};
    use test_utils::strategies::{
        exact_translation_case, finite_nonconstant_sample, finite_paired_samples,
        positive_power_of_two_scale,
    };

    use super::*;
    use crate::association::covariance;
    use crate::dispersion::std_dev;

    #[test]
    #[should_panic(expected = "inputs must have equal lengths")]
    fn panics_on_length_mismatch() {
        let _ = pearson(&[1.0, 2.0], &[3.0, 4.0, 5.0]);
    }

    #[test]
    #[should_panic(expected = "inputs must have equal lengths")]
    fn checks_lengths_before_minimum_sample_size() {
        let _ = pearson(&[], &[1.0]);
    }

    #[test]
    fn returns_nan_for_fewer_than_two_pairs() {
        assert!(pearson(&[], &[]).is_nan());
        assert!(pearson(&[1.0], &[2.0]).is_nan());
    }

    #[test]
    fn rejects_nonfinite_values_in_either_sample() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for bad in [[invalid, 1.0], [1.0, invalid]] {
                let good = [2.0, 3.0];
                assert!(pearson(&bad, &good).is_nan());
                assert!(pearson(&good, &bad).is_nan());
            }
        }
    }

    #[test]
    fn returns_nan_when_either_sample_is_constant() {
        // Subnormal values extend 52 powers of two below MIN_POSITIVE.
        let smallest_positive = f64::from_bits(1);
        let varying = [-1.0, 0.0, 1.0];
        let values = [0.0, smallest_positive, -f64::MAX, f64::MAX];

        for value in values {
            let constant = [value; 3];

            assert!(pearson(&constant, &varying).is_nan());
            assert!(pearson(&varying, &constant).is_nan());
            assert!(pearson(&constant, &constant).is_nan());
        }

        assert!(pearson(&[-0.0, 0.0, -0.0], &varying).is_nan());
    }

    #[test]
    fn two_distinct_pairs_have_perfect_positive_or_negative_correlation() {
        let xs = [1.0, 3.0];
        let ys_1 = [2.0, 6.0];
        let ys_2 = [6.0, 2.0];

        assert_approx_eq(pearson(&xs, &ys_1), 1.0, Tolerance::VERY_STRICT);
        assert_approx_eq(pearson(&xs, &ys_2), -1.0, Tolerance::VERY_STRICT);
    }

    #[test]
    fn calculates_an_ordinary_correlation() {
        // Centered values are [-1, 0, 1] and [-1/3, 2/3, -1/3]. Their products
        // cancel, so correlation is zero despite variation.
        let xs = [1.0, 2.0, 3.0];
        let ys = [1.0, 2.0, 1.0];
        assert_eq!(pearson(&xs, &ys), 0.0);

        // For [1, 2, 3] and [1, 3, 2], the centered product sum is 1 and both
        // squared-deviation sums are 2: correlation is 1/2.
        let xs = [1.0, 2.0, 3.0];
        let ys = [1.0, 3.0, 2.0];
        assert_approx_eq(pearson(&xs, &ys), 0.5, Tolerance::STRICT);
    }

    #[test]
    fn preserves_small_changes_around_a_large_offset() {
        // Both samples are constant for two pairs, then increase together.
        // The large starting value must not hide that perfect relationship.
        let xs = [1e16, 1e16, 1e16 + 2.0];
        let ys = [1.0, 1.0, 4.0];
        assert_approx_eq(pearson(&xs, &ys), 1.0, Tolerance::VERY_STRICT);
    }

    #[test]
    fn preserves_correlation_when_covariance_would_overflow() {
        // The samples move exactly together or exactly oppositely. Their
        // covariance is too large for f64, but correlation is ±1.
        let xs = [-f64::MAX, 0.0, f64::MAX];
        let ys = [f64::MAX, 0.0, -f64::MAX];

        assert_approx_eq(pearson(&xs, &xs), 1.0, Tolerance::VERY_STRICT);
        assert_approx_eq(pearson(&xs, &ys), -1.0, Tolerance::VERY_STRICT);
    }

    #[test]
    fn preserves_correlation_when_covariance_would_round_to_zero() {
        // Even the smallest representable changes can move exactly together.
        // Squaring these inputs directly would lose all variation to zero.
        let smallest_positive = f64::from_bits(1);
        let xs = [0.0, 0.0, smallest_positive];
        let ys = [0.0, 0.0, -smallest_positive];

        assert_approx_eq(pearson(&xs, &xs), 1.0, Tolerance::VERY_STRICT);
        assert_approx_eq(pearson(&xs, &ys), -1.0, Tolerance::VERY_STRICT);
    }

    #[test]
    fn handles_one_huge_sample_and_one_tiny_sample() {
        // Units do not affect correlation: both samples have pattern:
        // [-1, 0, 1].
        let large = 2.0_f64.powi(1000);
        let small = 2.0_f64.powi(-1000);
        let xs = [-large, 0.0, large];
        let ys = [-small, 0.0, small];

        assert_approx_eq(pearson(&xs, &ys), 1.0, Tolerance::VERY_STRICT);
        assert_approx_eq(pearson(&ys, &xs), 1.0, Tolerance::VERY_STRICT);
    }

    proptest! {
        #[test]
        fn is_symmetric((xs, ys) in finite_paired_samples(2)) {
            let a = pearson(&xs, &ys).to_bits();
            let b = pearson(&ys, &xs).to_bits();

            prop_assert_eq!(a, b);
        }

        #[test]
        fn agrees_with_covariance_and_stays_in_range(
            (xs, ys) in finite_paired_samples(2)
        ) {
            let denominator = std_dev(&xs) * std_dev(&ys);
            let result = pearson(&xs, &ys);

            if denominator == 0.0 {
                prop_assert!(result.is_nan());
            } else {
                let expected = covariance(&xs, &ys) / denominator;
                prop_assert!((-1.0..=1.0).contains(&result));
                prop_assert!(approx_eq(result, expected, Tolerance::STRICT));
            }
        }

        #[test]
        fn nonconstant_samples_correlate_perfectly_with_themselves(
            xs in finite_nonconstant_sample(2)
        ) {
            let result = pearson(&xs, &xs);
            prop_assert!(approx_eq(result, 1.0, Tolerance::VERY_STRICT));
        }

        #[test]
        fn translation_preserves_correlation(
            (xs, offset) in exact_translation_case(3)
        ) {
            let ys: Vec<_> = xs.iter().rev().copied().collect();

            let shifted_xs: Vec<_> = xs.iter().map(|x| x + offset).collect();
            let shifted_ys: Vec<_> = ys.iter().map(|y| y - offset).collect();

            let expected = pearson(&xs, &ys);
            let result = pearson(&shifted_xs, &shifted_ys);

            prop_assert!(approx_eq(result, expected, Tolerance::STRICT));
        }

        #[test]
        fn positive_scaling_preserves_correlation_and_negation_reverses_it(
            (xs, ys) in finite_paired_samples(2),
            x_scale in positive_power_of_two_scale(),
            y_scale in positive_power_of_two_scale(),
        ) {
            let scaled_x: Vec<_> = xs.iter().map(|x| x * x_scale).collect();
            let scaled_y: Vec<_> = ys.iter().map(|y| -y * y_scale).collect();
            let expected = -pearson(&xs, &ys);
            let result = pearson(&scaled_x, &scaled_y);

            if expected.is_nan() {
                prop_assert!(result.is_nan());
            } else {
                prop_assert!(approx_eq(result, expected, Tolerance::STRICT));
            }
        }
    }
}
