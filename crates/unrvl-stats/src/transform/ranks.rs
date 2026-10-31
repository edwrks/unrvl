use unrvl_numerics::convert::usize_to_f64;

use crate::internal::ranks::average_ranks;

/// Returns one-based average ranks in the original observation order.
///
/// Ranks start at one. Equal values receive the average of their occupied rank
/// positions; positive and negative zero are tied. For example, `[30, 10, 10]`
/// receives ranks `[3, 1.5, 1.5]`. Values are compared directly, preserving
/// distinctions between adjacent or widely differing finite observations.
///
/// An empty slice returns an empty vector. A singleton receives rank one, and
/// a constant sample receives rank `(n + 1) / 2` at every position. If any
/// observation is non-finite, returns one `NaN` per input observation.
///
/// Uses `O(n log n)` time and `O(n)` additional memory, including the returned
/// vector and sorted observation indices. Does not modify the input. Average
/// ranks are subject to `f64` rounding for very large samples.
///
/// # Panics
///
/// Panics if a finite sample contains more than `2^53` observations.
///
/// # Examples
///
/// ```
/// # use unrvl_stats::transform::ranks;
/// let xs = [30.0, 10.0, 10.0];
/// assert_eq!(ranks(&xs), vec![3.0, 1.5, 1.5]);
/// ```
#[must_use]
pub fn ranks(xs: &[f64]) -> Vec<f64> {
    if xs.iter().any(|value| !value.is_finite()) {
        return vec![f64::NAN; xs.len()];
    }

    let _ = usize_to_f64(xs.len());

    average_ranks(xs)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_utils::strategies::{exact_translation_case, finite_sample};

    use super::*;

    #[test]
    fn ranks_distinct_observations_in_original_order() {
        let xs = [30.0, 10.0, 20.0];

        let actual = ranks(&xs);
        let expected = [3.0, 1.0, 2.0];

        assert_eq!(actual, expected);
    }

    #[test]
    fn averages_each_tied_group() {
        let xs = [4.0, 1.0, 2.0, 2.0, 4.0];

        let actual = ranks(&xs);
        let expected = [4.5, 1.0, 2.5, 2.5, 4.5];

        assert_eq!(actual, expected);
    }

    #[test]
    fn empty_input_returns_an_empty_vector() {
        let actual = ranks(&[]);

        assert_eq!(actual, [] as [f64; 0]);
    }

    #[test]
    fn singleton_receives_rank_one() {
        let actual = ranks(&[42.0]);

        assert_eq!(actual, vec![1.0]);
    }

    #[test]
    fn constant_samples_receive_the_average_rank() {
        let values = [0.0, -0.0, 42.0, f64::from_bits(1), f64::MAX, -f64::MAX];

        for value in values {
            let xs = [value; 4];

            let actual = ranks(&xs);

            assert_eq!(actual, vec![2.5; 4]);
        }
    }

    #[test]
    fn signed_zeros_are_tied() {
        let xs = [-0.0, 1.0, 0.0, -1.0];

        let actual = ranks(&xs);
        let expected = [2.5, 4.0, 2.5, 1.0];

        assert_eq!(actual, expected);
    }

    #[test]
    fn nonfinite_observations_make_all_results_nan() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let xs = [1.0, value, 3.0];

            let actual = ranks(&xs);

            assert_eq!(actual.len(), xs.len());
            assert!(actual.into_iter().all(f64::is_nan));
        }
    }

    #[test]
    fn compares_extreme_values_without_scaling_away_small_differences() {
        let tiny = f64::from_bits(1);
        let xs = [f64::MAX, tiny, 0.0, -tiny, -f64::MAX];

        let actual = ranks(&xs);
        let expected = [5.0, 4.0, 3.0, 2.0, 1.0];

        assert_eq!(actual, expected);
    }

    #[test]
    fn adjacent_values_are_distinct() {
        let value = 1.0_f64;
        let xs = [value, value.next_up(), value.next_down()];

        let actual = ranks(&xs);
        let expected = [2.0, 3.0, 1.0];

        assert_eq!(actual, expected);
    }

    proptest! {
        #[test]
        #[expect(clippy::float_cmp, reason = "ties use exact equality")]
        fn ranks_match_counts_of_lower_and_equal_observations(xs in finite_sample(1)) {
            // Bounded integer observations deliberately introduce ties.
            let xs: Vec<_> = xs.iter().map(|value| value.round()).collect();

            let actual = ranks(&xs);

            prop_assert_eq!(actual.len(), xs.len());

            for (&value, &actual) in xs.iter().zip(&actual) {
                let lower_count = xs.iter().filter(|&&other| other < value).count();
                let equal_count = xs.iter().filter(|&&other| other == value).count();

                let tied_position = usize_to_f64(equal_count).midpoint(1.0);
                let expected = usize_to_f64(lower_count) + tied_position;

                prop_assert_eq!(actual, expected);
            }
        }

        #[test]
        fn reversing_observations_reverses_the_output_order(xs in finite_sample(1)) {
            let reversed: Vec<_> = xs.iter().rev().copied().collect();

            let mut expected = ranks(&xs);
            expected.reverse();

            let actual = ranks(&reversed);

            prop_assert_eq!(actual, expected);
        }

        #[test]
        fn exact_affine_changes_preserve_or_reverse_ranks(
            (xs, offset) in exact_translation_case(2),
        ) {
            let increasing: Vec<_> = xs.iter()
                .map(|&value| 2.0_f64.mul_add(value, offset))
                .collect();

            let decreasing: Vec<_> = xs.iter()
                .map(|&value| (-2.0_f64).mul_add(value, offset))
                .collect();

            let expected = ranks(&xs);
            let rank_sum = usize_to_f64(xs.len()) + 1.0;
            let expected_decreasing: Vec<_> = expected.iter().map(|rank| rank_sum - rank).collect();

            let actual_increasing = ranks(&increasing);
            let actual_decreasing = ranks(&decreasing);

            prop_assert_eq!(actual_increasing, expected);
            prop_assert_eq!(actual_decreasing, expected_decreasing);
        }
    }
}
