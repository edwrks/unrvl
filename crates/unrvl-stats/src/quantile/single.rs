use super::internal::{is_valid_probability, is_valid_sample, type7};

/// Returns the `p`-quantile of `xs` using R's default Type 7 interpolation.
///
/// For sorted observations, the zero-based position is `h = (n - 1) * p`, where
/// `n` is the sample length. An integer position selects an observation;
/// otherwise, the adjacent observations are linearly interpolated.
///
/// Probabilities zero and one return the minimum and maximum, respectively. A
/// single observation is returned for every valid probability.
///
/// The input slice does not need to be sorted. The observations are copied and
/// sorted before evaluation.
///
/// Returns `NaN` if `xs` is empty, any observation is non-finite, or `p` is
/// non-finite or outside `[0.0, 1.0]`.
///
/// # Panics
///
/// Panics if `xs.len() - 1` exceeds `2^53`.
#[must_use]
pub fn quantile(xs: &[f64], p: f64) -> f64 {
    if !is_valid_sample(xs) || !is_valid_probability(p) {
        return f64::NAN;
    }

    let mut sorted = xs.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);

    type7(&sorted, p)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_utils::approx::{Tolerance, approx_eq, assert_approx_eq};
    use test_utils::strategies::{
        constant_sample, finite_sample, positive_power_of_two_scale, probability,
    };
    use unrvl_numerics::extrema::min_max;

    use super::*;

    #[test]
    fn computes_type7_quantile_of_unsorted_sample() {
        let xs = [40.0, 0.0, 30.0, 10.0, 20.0];

        assert_approx_eq(quantile(&xs, 0.375), 15.0, Tolerance::VERY_STRICT);
    }

    #[test]
    fn zero_and_one_return_sample_bounds() {
        let xs = [4.0, -2.0, 8.0, 1.0];

        assert_eq!(quantile(&xs, 0.0), -2.0);
        assert_eq!(quantile(&xs, 1.0), 8.0);
    }

    #[test]
    fn singleton_returns_observation_for_every_probability() {
        let xs = [42.0];

        for p in [0.0, 0.25, 0.5, 0.95, 1.0] {
            assert_eq!(quantile(&xs, p), 42.0);
        }
    }

    #[test]
    fn interpolates_extreme_opposite_signed_observations() {
        let xs = [-f64::MAX, f64::MAX];
        let actual = quantile(&xs, 0.5);

        assert!(actual.is_finite());
        assert_eq!(actual, 0.0);
    }

    #[test]
    fn empty_sample_returns_nan() {
        assert!(quantile(&[], 0.5).is_nan());
    }

    #[test]
    fn nonfinite_observation_returns_nan() {
        assert!(quantile(&[1.0, f64::NAN], 0.5).is_nan());
        assert!(quantile(&[1.0, f64::INFINITY], 0.5).is_nan());
        assert!(quantile(&[1.0, f64::NEG_INFINITY], 0.5).is_nan());
    }

    #[test]
    fn invalid_probability_returns_nan() {
        let xs = [1.0, 2.0, 3.0];

        assert!(quantile(&xs, -f64::EPSILON).is_nan());
        assert!(quantile(&xs, 1.0 + f64::EPSILON).is_nan());
        assert!(quantile(&xs, f64::NAN).is_nan());
        assert!(quantile(&xs, f64::INFINITY).is_nan());
        assert!(quantile(&xs, f64::NEG_INFINITY).is_nan());
    }

    proptest! {
        #[test]
        fn quantiles_stay_within_the_sample_bounds(
            xs in finite_sample(1),
            p in probability(),
        ) {
            let (minimum, maximum) = min_max(&xs);
            let actual = quantile(&xs, p);

            prop_assert!(
                (minimum..=maximum).contains(&actual),
                "quantile {actual} at p={p} is outside [{minimum}, {maximum}]",
            );
        }

        #[test]
        fn quantiles_are_monotonic_in_probability(
            xs in finite_sample(1),
            p1 in probability(),
            p2 in probability(),
        ) {
            let lower_p = p1.min(p2);
            let upper_p = p1.max(p2);

            let lower = quantile(&xs, lower_p);
            let upper = quantile(&xs, upper_p);

            prop_assert!(
                lower <= upper,
                "q({lower_p})={lower} exceeds q({upper_p})={upper}",
            );
        }

        #[test]
        fn quantiles_are_invariant_to_input_order(
            xs in finite_sample(1),
            p in probability(),
        ) {
            let expected = quantile(&xs, p);

            let mut reversed = xs;
            reversed.reverse();

            let actual = quantile(&reversed, p);

            prop_assert_eq!(actual.to_bits(), expected.to_bits());
        }

        #[test]
        #[expect(clippy::float_cmp, reason = "constant sample must produce exact result")]
        fn constant_sample_quantile_is_the_constant(
            xs in constant_sample(1),
            p in probability(),
        ) {
            let expected = xs[0];
            let actual = quantile(&xs, p);

            prop_assert_eq!(actual, expected);
        }

        #[test]
        fn quantiles_are_scale_equivariant(
            xs in finite_sample(1),
            p in probability(),
            scale in positive_power_of_two_scale(),
        ) {
            let scaled: Vec<_> = xs.iter().map(|&x| x * scale).collect();

            let expected = quantile(&xs, p) * scale;
            let actual = quantile(&scaled, p);

            prop_assert!(approx_eq( actual, expected, Tolerance::VERY_STRICT));
        }
    }
}
