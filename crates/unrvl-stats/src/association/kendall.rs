use unrvl_numerics::convert::usize_to_f64;

/// Returns Kendall's tau-b correlation of paired observations.
///
/// Measures agreement between the orderings of two samples, accounting for
/// ties. For each pair of observation positions, matching directions are
/// concordant and opposite directions are discordant. A tie in either sample
/// contributes to neither count. Positive and negative zero are tied.
///
/// The statistic is:
///
/// ```text
///               C - D
/// tau_b = -------------------
///         sqrt(Nx) · sqrt(Ny)
/// ```
///
/// Here `C` and `D` count concordant and discordant pairs. `Nx` counts pairs
/// whose x values differ, and `Ny` counts pairs whose y values differ.
/// The result lies in `[-1, 1]`.
///
/// Observations are paired by position. Returns `NaN` if fewer than two
/// observations per sample are provided, any observation is non-finite, or
/// either sample is constant. No observations are discarded.
///
/// Counts and their signed difference are accumulated exactly as integers; the
/// final normalization is subject to floating-point rounding. Rounding beyond
/// `[-1, 1]` is clamped to the nearest endpoint.
///
/// Uses sorting and merge-sort inversion counting: `O(n log n)` time and `O(n)`
/// additional memory.
///
/// # Panics
///
/// Panics if the slices have different lengths, or if the samples are finite
/// and contain more than `2^53` observations each.
///
/// # Examples
///
/// ```
/// # use unrvl_stats::association::kendall;
/// let xs = [1.0, 2.0, 3.0];
/// let ys = [1.0, 3.0, 2.0];
/// // Two pairs agree in direction and one disagrees: tau-b = (2 - 1) / 3.
/// assert!((kendall(&xs, &ys) - 1.0 / 3.0).abs() < 1e-14);
/// ```
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "integer pair counts stay exact until the final floating-point normalization"
)]
#[expect(
    clippy::float_cmp,
    reason = "ties use exact equality, including signed zeros"
)]
#[expect(
    clippy::cast_possible_wrap,
    reason = "the 2^53 sample bound keeps pair counts within i128"
)]
pub fn kendall(xs: &[f64], ys: &[f64]) -> f64 {
    assert_eq!(xs.len(), ys.len(), "inputs must have equal lengths");

    let n = xs.len();
    if n < 2 || xs.iter().chain(ys).any(|value| !value.is_finite()) {
        return f64::NAN;
    }

    // Keep the same sample-length bound as the other association functions.
    // At most n*(n-1)/2 pairs fit comfortably in the 128-bit counters below.
    let _ = usize_to_f64(n);
    let mut observations: Vec<_> = xs.iter().copied().zip(ys.iter().copied()).collect();

    // Order by x, then y. Sorting y within an equal-x group prevents those tied
    // pairs from being counted as discordant. Numeric comparison treats signed
    // zeros as equal and avoids subtracting potentially extreme values.
    observations.sort_unstable_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .expect("inputs should be finite")
            .then_with(|| a.1.partial_cmp(&b.1).expect("inputs should be finite"))
    });

    let x_ties: u128 = observations
        .chunk_by(|a, b| a.0 == b.0)
        .map(|group| pair_count(group.len()))
        .sum();

    let joint_ties: u128 = observations
        .chunk_by(|a, b| a == b)
        .map(|group| pair_count(group.len()))
        .sum();

    let mut ordered_y: Vec<_> = observations.iter().map(|&(_, y)| y).collect();
    let mut scratch = vec![0.0; n];

    // With x ordered, strictly descending y pairs are exactly the discordant
    // pairs. The counter also sorts y, so equal values can then be grouped.
    let discordant = count_inversions(&mut ordered_y, &mut scratch);
    let y_ties: u128 = ordered_y
        .chunk_by(|a, b| a == b)
        .map(|group| pair_count(group.len()))
        .sum();

    let total = pair_count(n);
    let x_pairs = total - x_ties;
    let y_pairs = total - y_ties;
    if x_pairs == 0 || y_pairs == 0 {
        return f64::NAN;
    }

    // Remove pairs tied in x, then pairs tied only in y. Subtracting only
    // y-only ties avoids double-counting joint ties or unsigned underflow.
    let comparable = total - x_ties - (y_ties - joint_ties);

    // Comparable pairs are C + D, so C - D = comparable - 2*D.
    let score = comparable as i128 - 2 * discordant as i128;
    let denominator = (x_pairs as f64).sqrt() * (y_pairs as f64).sqrt();

    (score as f64 / denominator).clamp(-1.0, 1.0)
}

/// Number of observation pairs in a nonempty group.
const fn pair_count(n: usize) -> u128 {
    let n = n as u128;
    n * (n - 1) / 2
}

/// Counts strictly descending pairs while sorting finite values in ascending
/// order. The scratch buffer must have the same length as the input.
fn count_inversions(values: &mut [f64], scratch: &mut [f64]) -> u128 {
    if values.len() < 2 {
        return 0;
    }

    let middle = values.len() / 2;
    let (left, right) = values.split_at_mut(middle);
    let (left_scratch, right_scratch) = scratch.split_at_mut(middle);
    let mut count = count_inversions(left, left_scratch) + count_inversions(right, right_scratch);

    let mut i = 0;
    let mut j = 0;
    for output in scratch.iter_mut() {
        if i < left.len() && (j == right.len() || left[i] <= right[j]) {
            // Take equal values from the left: ties are not inversions.
            *output = left[i];
            i += 1;
        } else {
            *output = right[j];
            j += 1;
            // Both halves are sorted. This right value is smaller than every
            // remaining left value, so count all those pairs at once.
            count += (left.len() - i) as u128;
        }
    }

    values.copy_from_slice(scratch);
    count
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
        let _ = kendall(&[1.0, 2.0], &[3.0, 4.0, 5.0]);
    }

    #[test]
    #[should_panic(expected = "inputs must have equal lengths")]
    fn checks_lengths_before_minimum_sample_size() {
        let _ = kendall(&[], &[1.0]);
    }

    #[test]
    fn returns_nan_for_fewer_than_two_pairs() {
        assert!(kendall(&[], &[]).is_nan());
        assert!(kendall(&[1.0], &[2.0]).is_nan());
    }

    #[test]
    fn rejects_nonfinite_values_in_either_sample() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for bad in [[invalid, 1.0], [1.0, invalid]] {
                let good = [2.0, 3.0];
                assert!(kendall(&bad, &good).is_nan());
                assert!(kendall(&good, &bad).is_nan());
            }
        }
    }

    #[test]
    fn returns_nan_when_either_sample_is_constant() {
        let varying = [1.0, 2.0, 3.0];
        let values = [-f64::MAX, 0.0, f64::MAX];

        for value in values {
            let constant = [value; 3];

            assert!(kendall(&constant, &varying).is_nan());
            assert!(kendall(&varying, &constant).is_nan());
            assert!(kendall(&constant, &constant).is_nan());
        }

        assert!(kendall(&[-0.0, 0.0, -0.0], &varying).is_nan());
    }

    #[test]
    fn two_distinct_pairs_have_perfect_positive_or_negative_correlation() {
        let xs = [1.0, 3.0];
        assert_eq!(kendall(&xs, &[2.0, 6.0]), 1.0);
        assert_eq!(kendall(&xs, &[6.0, 2.0]), -1.0);
    }

    #[test]
    fn counts_agreeing_and_disagreeing_pairs() {
        // Observation pairs (0, 1) and (0, 2) agree; (1, 2) disagrees.
        // With no ties, tau-b = (2 - 1) / 3.
        let xs = [1.0, 2.0, 3.0];
        let ys = [1.0, 3.0, 2.0];

        let result = kendall(&xs, &ys);

        assert_approx_eq(result, 1.0 / 3.0, Tolerance::VERY_STRICT);
    }

    #[test]
    fn returns_zero_when_agreement_and_disagreement_balance() {
        // Of six observation pairs, three agree and three disagree.
        let xs = [1.0, 2.0, 3.0, 4.0];
        let ys = [2.0, 4.0, 1.0, 3.0];

        let result = kendall(&xs, &ys);

        assert_eq!(result, 0.0);
    }

    #[test]
    fn adjusts_for_ties_in_one_sample() {
        // Two pairs agree. The remaining pair is tied only in x.
        // There are two differing x pairs and three differing y pairs.
        let xs = [1.0, 1.0, 2.0];
        let ys = [1.0, 2.0, 3.0];

        let expected = 2.0 / 6.0_f64.sqrt();

        assert_approx_eq(kendall(&xs, &ys), expected, Tolerance::VERY_STRICT);
        assert_approx_eq(kendall(&ys, &xs), expected, Tolerance::VERY_STRICT);
    }

    #[test]
    fn adjusts_for_separate_ties_in_both_samples() {
        // Of six pairs, four agree, one is tied only in x, and one only in y.
        // Each sample has five differing pairs, so tau-b = 4 / 5.
        let xs = [1.0, 1.0, 2.0, 3.0];
        let ys = [1.0, 2.0, 2.0, 3.0];

        assert_approx_eq(kendall(&xs, &ys), 0.8, Tolerance::VERY_STRICT);
    }

    #[test]
    fn excludes_pairs_tied_in_both_samples() {
        // The first two observations are identical in both samples.
        // Their pair adds nothing to the score or either denominator count.
        // The other two pairs agree, giving 2 / sqrt(2*2) = 1.
        let xs = [1.0, 1.0, 2.0];
        let ys = [3.0, 3.0, 4.0];

        assert_approx_eq(kendall(&xs, &ys), 1.0, Tolerance::VERY_STRICT);
    }

    #[test]
    fn treats_positive_and_negative_zero_as_tied() {
        // Signed zeros must give the same tie correction as equal ones.
        let xs = [-0.0, 0.0, 1.0];
        let ys = [1.0, 2.0, 3.0];
        let expected = 2.0 / 6.0_f64.sqrt();

        assert_approx_eq(kendall(&xs, &ys), expected, Tolerance::VERY_STRICT);
    }

    #[test]
    fn detects_perfect_monotonic_relationships_even_when_nonlinear() {
        let xs = [1.0, 2.0, 3.0, 4.0];
        let increasing = [1.0, 4.0, 9.0, 16.0];
        let decreasing = [16.0, 9.0, 4.0, 1.0];

        assert_approx_eq(kendall(&xs, &increasing), 1.0, Tolerance::VERY_STRICT);
        assert_approx_eq(kendall(&xs, &decreasing), -1.0, Tolerance::VERY_STRICT);
    }

    #[test]
    fn compares_extreme_values_without_losing_tiny_differences() {
        // Direct comparisons preserve both tiny and huge differences.
        // Subtracting extremes could overflow; multiplying tiny differences
        // could round them to zero and misclassify them as ties.
        let smallest_positive = f64::MIN_POSITIVE / 2.0_f64.powi(52);
        let max = f64::MAX;

        let xs = [-max, -smallest_positive, 0.0, smallest_positive, max];
        let ys = [5.0, 4.0, 3.0, 2.0, 1.0];

        assert_approx_eq(kendall(&xs, &ys), -1.0, Tolerance::VERY_STRICT);
        assert_approx_eq(kendall(&xs, &xs), 1.0, Tolerance::VERY_STRICT);
    }

    #[test]
    fn counts_more_discordant_pairs_than_fit_in_u32() {
        // 100,000 observations give 4,999,950,000 discordant pairs.
        let xs: Vec<_> = (0..100_000).map(usize_to_f64).collect();
        let ys: Vec<_> = xs.iter().rev().copied().collect();
        assert_approx_eq(kendall(&xs, &ys), -1.0, Tolerance::VERY_STRICT);
    }

    proptest! {
        #[test]
        fn is_symmetric((xs, ys) in finite_paired_samples(2)) {
            let a = kendall(&xs, &ys).to_bits();
            let b = kendall(&ys, &xs).to_bits();

            prop_assert_eq!(a, b);
        }

        #[test]
        fn nonconstant_samples_correlate_perfectly_with_themselves(xs in finite_nonconstant_sample(2)) {
            let result =  kendall(&xs, &xs);

            prop_assert!(approx_eq(result, 1.0, Tolerance::VERY_STRICT));
        }

        #[test]
        fn joint_reversal_preserves_correlation((mut xs, mut ys) in finite_paired_samples(2)) {
            let expected = kendall(&xs, &ys).to_bits();

            xs.reverse();
            ys.reverse();

            let reversed = kendall(&xs, &ys).to_bits();

            prop_assert_eq!(reversed, expected);
        }

        #[test]
        fn increasing_transformations_preserve_correlation_and_negation_reverses_it(
            (xs, ys) in finite_paired_samples(2),
        ) {
            // Bounded integer cubes are exact in f64. They preserve ordering
            // and ties; negating one sample reverses agreement/disagreement.
            let xs: Vec<_> = xs.iter().map(|x| x.round()).collect();
            let ys: Vec<_> = ys.iter().map(|y| y.round()).collect();
            let transformed_x: Vec<_> = xs.iter().map(|x| x * x * x).collect();
            let transformed_y: Vec<_> = ys.iter().map(|y| -(y * y * y)).collect();
            let expected = -kendall(&xs, &ys);
            let result = kendall(&transformed_x, &transformed_y);

            if expected.is_nan() {
                prop_assert!(result.is_nan());
            } else {
                prop_assert!((-1.0..=1.0).contains(&result));
                prop_assert!(approx_eq(result, expected, Tolerance::STRICT));
            }
        }
    }
}
