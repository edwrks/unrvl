use super::internal::{is_valid_probability, is_valid_sample, type7};

/// Returns multiple quantiles of `xs` using R's default Type 7 interpolation.
///
/// This is equivalent to calling [`quantile`](super::quantile) once for each
/// probability in `ps`, except that the input is copied and sorted only once.
///
/// Each probability must be finite and in the closed interval `[0.0, 1.0]`.
///
/// Results are returned in the same order as the requested probabilities.
///
/// The input slice does not need to be sorted.
///
/// Invalid probabilities produce `NaN` at their corresponding output positions
/// without affecting other requested quantiles.
///
/// Returns one `NaN` per requested probability if `xs` is empty or contains a
/// `NaN` or infinite observation. An empty `ps` returns an empty vector.
#[must_use]
pub fn quantiles(xs: &[f64], ps: &[f64]) -> Vec<f64> {
    if ps.is_empty() {
        return Vec::new();
    }

    if !is_valid_sample(xs) {
        return vec![f64::NAN; ps.len()];
    }

    let mut sorted = xs.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);

    ps.iter()
        .map(|&p| {
            if is_valid_probability(p) {
                type7(&sorted, p)
            } else {
                f64::NAN
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_utils::strategies::{finite_sample, probabilities};

    use super::*;
    use crate::quantile::quantile;

    #[test]
    fn computes_multiple_type7_quantiles() {
        let xs = [40.0, 0.0, 30.0, 10.0, 20.0];
        let ps = [0.0, 0.125, 0.5, 0.875, 1.0];

        let actual = quantiles(&xs, &ps);

        assert_eq!(actual.len(), 5);
        assert_eq!(actual[0], 0.0);
        assert_eq!(actual[1], 5.0);
        assert_eq!(actual[2], 20.0);
        assert_eq!(actual[3], 35.0);
        assert_eq!(actual[4], 40.0);
    }

    #[test]
    fn preserves_probability_order_and_duplicates() {
        let xs = [0.0, 10.0, 20.0, 30.0, 40.0];
        let ps = [1.0, 0.5, 0.0, 0.5];

        let actual = quantiles(&xs, &ps);

        assert_eq!(actual[0], 40.0);
        assert_eq!(actual[1], 20.0);
        assert_eq!(actual[2], 0.0);
        assert_eq!(actual[3], 20.0);
    }

    #[test]
    fn invalid_probability_only_affects_its_position() {
        let xs = [0.0, 10.0, 20.0];
        let ps = [0.0, -0.1, 0.5, f64::NAN, 1.0, 1.1];

        let actual = quantiles(&xs, &ps);

        assert_eq!(actual[0], 0.0);
        assert!(actual[1].is_nan());
        assert_eq!(actual[2], 10.0);
        assert!(actual[3].is_nan());
        assert_eq!(actual[4], 20.0);
        assert!(actual[5].is_nan());
    }

    #[test]
    fn singleton_returns_observation_for_every_valid_probability() {
        let ps = [0.0, 0.25, 0.5, 0.95, 1.0];

        let actual = quantiles(&[42.0], &ps);

        for value in actual {
            assert_eq!(value, 42.0);
        }
    }

    #[test]
    fn empty_probabilities_return_empty_vector() {
        let actual = quantiles(&[1.0, 2.0, 3.0], &[]);

        assert_eq!(actual, [] as [f64; 0]);
    }

    #[test]
    fn interpolates_extreme_opposite_signed_observations() {
        let actual = quantiles(&[-f64::MAX, f64::MAX], &[0.0, 0.5, 1.0]);

        assert_eq!(actual[0], -f64::MAX);
        assert!(actual[1].is_finite());
        assert_eq!(actual[1], 0.0);
        assert_eq!(actual[2], f64::MAX);
    }

    #[test]
    fn empty_sample_returns_nan_for_each_probability() {
        let actual = quantiles(&[], &[0.25, 0.5, 0.75]);

        assert_eq!(actual.len(), 3);
        assert!(actual.iter().all(|x| x.is_nan()));
    }

    #[test]
    fn nonfinite_observation_returns_nan_for_each_probability() {
        for xs in [
            &[1.0, f64::NAN],
            &[1.0, f64::INFINITY],
            &[1.0, f64::NEG_INFINITY],
        ] {
            let actual = quantiles(xs, &[0.25, 0.5, 0.75]);

            assert_eq!(actual.len(), 3);
            assert!(actual.iter().all(|x| x.is_nan()));
        }
    }

    proptest! {
        #[test]
        #[expect(clippy::float_cmp, reason = "comparison must produce exact result")]
        fn quantiles_match_individual_quantile_calls(
            xs in finite_sample(1),
            ps in probabilities(),
        ) {
            let actual = quantiles(&xs, &ps);

            prop_assert_eq!(actual.len(), ps.len());

            for (&p, &batched) in ps.iter().zip(&actual) {
                let individual = quantile(&xs, p);

                prop_assert_eq!(batched, individual);
            }
        }

        #[test]
        fn quantiles_are_monotonic_for_sorted_probabilities(
            xs in finite_sample(1),
            ps in probabilities(),
        ) {
            let mut ps = ps;
            ps.sort_unstable_by(f64::total_cmp);

            let actual = quantiles(&xs, &ps);

            for pair in actual.windows(2) {
                prop_assert!(
                    pair[0] <= pair[1],
                    "lower quantile {} exceeds upper quantile {}",
                    pair[0],
                    pair[1],
                );
            }
        }

        #[test]
        #[expect(clippy::float_cmp, reason = "comparison must produce exact result")]
        fn quantiles_are_invariant_to_input_order(
            xs in finite_sample(1),
            ps in probabilities(),
        ) {
            let expected = quantiles(&xs, &ps);

            let mut reversed = xs;
            reversed.reverse();

            let actual = quantiles(&reversed, &ps);

            prop_assert_eq!(actual.len(), expected.len());

            for (&actual, &expected) in actual.iter().zip(&expected) {
                prop_assert_eq!(actual, expected);
            }
        }
    }
}
