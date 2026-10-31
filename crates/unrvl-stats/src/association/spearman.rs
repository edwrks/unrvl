use unrvl_numerics::convert::usize_to_f64;

use super::pearson;
use crate::internal::ranks::average_ranks;

/// Returns the Spearman rank correlation of paired observations.
///
/// Measures monotonic association by calculating [`pearson`] on the ranks of
/// each sample. Ranks start at one. Tied values receive their average rank:
/// for example, `[30, 10, 10]` becomes `[3, 1.5, 1.5]`. Positive and negative
/// zero are tied. The result lies in `[-1, 1]`.
///
/// Observations are paired by position. Returns `NaN` if fewer than two pairs
/// are provided, any observation is non-finite, or either sample is constant.
/// No pairs are discarded.
///
/// Rank correlation is calculated with compensated arithmetic; the
/// floating-point result is subject to rounding.
///
/// Uses sorting to assign ranks: `O(n log n)` time and `O(n)` additional
/// memory, where `n` is the number of paired observations.
///
/// # Panics
///
/// Panics if the slices have different lengths, or if a finite sample contains
/// more than `2^53` observations.
///
/// # Examples
///
/// ```
/// # use unrvl_stats::association::spearman;
/// let xs = [1.0, 2.0, 3.0, 4.0];
/// let ys = [1.0, 4.0, 9.0, 16.0];
/// // The relationship is increasing, even though it is not linear.
/// assert!((spearman(&xs, &ys) - 1.0).abs() < 1e-14);
/// ```
#[must_use]
pub fn spearman(xs: &[f64], ys: &[f64]) -> f64 {
    assert_eq!(xs.len(), ys.len(), "inputs must have equal lengths");

    if xs.len() < 2 || xs.iter().chain(ys).any(|v| !v.is_finite()) {
        return f64::NAN;
    }

    // Check the rank range before sorting either sample.
    let _ = usize_to_f64(xs.len());

    let x_ranks = average_ranks(xs);
    let y_ranks = average_ranks(ys);

    pearson(&x_ranks, &y_ranks)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_utils::approx::{Tolerance, approx_eq, assert_approx_eq};
    use test_utils::strategies::{finite_nonconstant_sample, finite_paired_samples};

    use super::*;

    #[test]
    #[should_panic(expected = "inputs must have equal lengths")]
    fn panics_on_length_mismatch() {
        let _ = spearman(&[1.0, 2.0], &[3.0, 4.0, 5.0]);
    }

    #[test]
    #[should_panic(expected = "inputs must have equal lengths")]
    fn checks_lengths_before_minimum_sample_size() {
        let _ = spearman(&[], &[1.0]);
    }

    #[test]
    fn returns_nan_for_fewer_than_two_pairs() {
        assert!(spearman(&[], &[]).is_nan());
        assert!(spearman(&[1.0], &[2.0]).is_nan());
    }

    #[test]
    fn rejects_nonfinite_values_before_ranking() {
        // Sorting NaN or infinity could otherwise turn invalid values into
        // ordinary finite ranks and incorrectly produce a valid correlation.
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for bad in [[invalid, 1.0], [1.0, invalid]] {
                let good = [2.0, 3.0];
                assert!(spearman(&bad, &good).is_nan());
                assert!(spearman(&good, &bad).is_nan());
            }
        }
    }

    #[test]
    fn returns_nan_when_either_sample_is_constant() {
        let varying = [1.0, 2.0, 3.0];
        let values = [-f64::MAX, 0.0, f64::MAX];

        for value in values {
            let constant = [value; 3];

            assert!(spearman(&constant, &varying).is_nan());
            assert!(spearman(&varying, &constant).is_nan());
            assert!(spearman(&constant, &constant).is_nan());
        }

        assert!(spearman(&[-0.0, 0.0, -0.0], &varying).is_nan());
    }

    #[test]
    fn treats_positive_and_negative_zero_as_tied() {
        let xs = [-0.0, 1.0, 0.0, -1.0];
        let ys = [1.0, 3.0, 1.0, 0.0];

        assert_approx_eq(spearman(&xs, &ys), 1.0, Tolerance::VERY_STRICT);
    }

    #[test]
    fn calculates_correlation_for_mixed_order_without_ties() {
        // The centered ranks have product sum 4 and squared sums 5 each.
        let xs = [10.0, 20.0, 30.0, 40.0];
        let ys = [10.0, 30.0, 20.0, 40.0];

        assert_approx_eq(spearman(&xs, &ys), 0.8, Tolerance::VERY_STRICT);
    }

    #[test]
    fn uses_average_ranks_when_one_sample_has_ties() {
        // Ranks are [1.5, 1.5, 3] and [1, 2, 3]. After centering, their product
        // sum is 1.5 and squared sums are 1.5 and 2. r = sqrt(3) / 2.
        let xs = [1.0, 1.0, 2.0];
        let ys = [1.0, 2.0, 3.0];
        let expected = 3.0_f64.sqrt() / 2.0;

        assert_approx_eq(spearman(&xs, &ys), expected, Tolerance::VERY_STRICT);
    }

    #[test]
    fn uses_average_ranks_when_both_samples_have_ties() {
        // Ranks are [1.5, 1.5, 3, 4] and [1, 2.5, 2.5, 4]. Their centered
        // product sum is 3.75 and squared sums are 4.5 each: r = 5/6.
        let xs = [1.0, 1.0, 2.0, 3.0];
        let ys = [1.0, 2.0, 2.0, 3.0];
        let expected = 5.0 / 6.0;

        assert_approx_eq(spearman(&xs, &ys), expected, Tolerance::VERY_STRICT);
    }

    #[test]
    fn detects_perfect_monotonic_relationships_even_when_nonlinear() {
        let xs = [1.0, 2.0, 3.0, 4.0];
        let increasing = [1.0, 4.0, 9.0, 16.0];
        let decreasing = [16.0, 9.0, 4.0, 1.0];

        assert_approx_eq(spearman(&xs, &increasing), 1.0, Tolerance::VERY_STRICT);
        assert_approx_eq(spearman(&xs, &decreasing), -1.0, Tolerance::VERY_STRICT);
    }

    #[test]
    fn two_distinct_pairs_have_perfect_positive_or_negative_correlation() {
        let xs = [1.0, 3.0];
        let ys_1 = [2.0, 6.0];
        let ys_2 = [6.0, 2.0];

        assert_approx_eq(spearman(&xs, &ys_1), 1.0, Tolerance::VERY_STRICT);
        assert_approx_eq(spearman(&xs, &ys_2), -1.0, Tolerance::VERY_STRICT);
    }

    #[test]
    fn ranks_extreme_values_without_losing_small_differences() {
        // Ranking compares values directly. Tiny differences near zero must
        // remain distinct even when the same sample contains enormous values.
        let smallest_positive = f64::from_bits(1);
        let max = f64::MAX;

        let xs = [max, smallest_positive, 0.0, -smallest_positive, -max];
        let ys = [5.0, 4.0, 3.0, 2.0, 1.0];

        assert_approx_eq(spearman(&xs, &ys), 1.0, Tolerance::VERY_STRICT);
    }

    proptest! {
        #[test]
        fn is_symmetric((xs, ys) in finite_paired_samples(2)) {
            let a = spearman(&xs, &ys).to_bits();
            let b = spearman(&ys, &xs).to_bits();

            prop_assert_eq!(a, b);
        }

        #[test]
        fn nonconstant_samples_correlate_perfectly_with_themselves(
            xs in finite_nonconstant_sample(2)
        ) {
            let result = spearman(&xs, &xs);
            prop_assert!(approx_eq(result, 1.0, Tolerance::VERY_STRICT));
        }

        #[test]
        fn increasing_transformations_preserve_correlation_and_negation_reverses_it(
            (xs, ys) in finite_paired_samples(2),
        ) {
            // Round to bounded integers to introduce ties. Their cubes are
            // exact in f64, so cubing preserves every ordering and tie.
            let xs: Vec<_> = xs.iter().map(|x| x.round()).collect();
            let ys: Vec<_> = ys.iter().map(|y| y.round()).collect();

            let transformed_x: Vec<_> = xs.iter().map(|x| x * x * x).collect();
            let transformed_y: Vec<_> = ys.iter().map(|y| -(y * y * y)).collect();

            let expected = -spearman(&xs, &ys);
            let result = spearman(&transformed_x, &transformed_y);

            if expected.is_nan() {
                prop_assert!(result.is_nan());
            } else {
                prop_assert!((-1.0..=1.0).contains(&result));
                prop_assert!(approx_eq(result, expected, Tolerance::STRICT));
            }
        }
    }
}
