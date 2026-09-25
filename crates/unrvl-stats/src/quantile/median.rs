use crate::quantile::quantile;

/// Returns the median of `xs` using R's default Type 7 interpolation.
///
/// This is equivalent to:
///
/// ```text
/// quantile(xs, 0.50)
/// ```
///
/// For an odd number of observations, the middle observation is returned. For
/// an even number, the result is interpolated halfway between the two middle
/// observations.
///
/// The input slice does not need to be sorted.
///
/// A single-observation sample returns that observation.
///
/// Returns `NaN` for an empty sample or when any observation is `NaN` or
/// infinite.
#[must_use]
pub fn median(xs: &[f64]) -> f64 {
    quantile(xs, 0.50)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_utils::approx::{Tolerance, approx_eq};
    use test_utils::strategies::{
        constant_sample, exact_translation_case, finite_sample, positive_power_of_two_scale,
    };
    use unrvl_numerics::extrema::min_max;

    use super::*;

    #[test]
    fn returns_middle_observation_for_odd_sample() {
        let xs = [5.0, 1.0, 3.0];

        assert_eq!(median(&xs), 3.0);
    }

    #[test]
    fn interpolates_middle_observations_for_even_sample() {
        let xs = [4.0, 1.0, 3.0, 2.0];

        assert_eq!(median(&xs), 2.5);
    }

    #[test]
    fn singleton_returns_observation() {
        assert_eq!(median(&[42.0]), 42.0);
    }

    #[test]
    fn interpolates_extreme_opposite_signed_observations() {
        let xs = [-f64::MAX, f64::MAX];

        let actual = median(&xs);

        assert!(actual.is_finite());
        assert_eq!(actual, 0.0);
    }

    #[test]
    fn empty_sample_returns_nan() {
        assert!(median(&[]).is_nan());
    }

    #[test]
    fn nonfinite_observation_returns_nan() {
        assert!(median(&[1.0, f64::NAN]).is_nan());
        assert!(median(&[1.0, f64::INFINITY]).is_nan());
        assert!(median(&[1.0, f64::NEG_INFINITY]).is_nan());
    }

    proptest! {
        #[test]
        fn median_stays_within_the_sample_bounds(
            xs in finite_sample(1),
        ) {
            let (minimum, maximum) = min_max(&xs);
            let actual = median(&xs);

            prop_assert!(
                (minimum..=maximum).contains(&actual),
                "median {actual} is outside [{minimum}, {maximum}]",
            );
        }

        #[test]
        fn median_is_invariant_to_input_order(
            xs in finite_sample(1),
        ) {
            let expected = median(&xs);

            let mut reversed = xs;
            reversed.reverse();

            let actual = median(&reversed);

            prop_assert_eq!(actual.to_bits(), expected.to_bits());
        }

        #[test]
        #[expect(clippy::float_cmp, reason = "constant sample must produce exact result")]
        fn constant_sample_median_is_the_constant(
            xs in constant_sample(1),
        ) {
            let actual = median(&xs);

            prop_assert_eq!(actual, xs[0]);
        }

        #[test]
        fn median_is_translation_equivariant(
            (sample, offset) in exact_translation_case(2)
        ) {
            let translated: Vec<_> = sample.iter().map(|&x| x + offset).collect();

            let expected = median(&sample) + offset;
            let actual = median(&translated);

            prop_assert!(approx_eq(actual, expected, Tolerance::STRICT));
        }

        #[test]
        fn median_is_scale_equivariant(
            xs in finite_sample(1),
            scale in positive_power_of_two_scale(),
        ) {
            let scaled: Vec<_> =
            xs.iter().map(|&x| x * scale).collect();

            let expected = median(&xs) * scale;
            let actual = median(&scaled);

            prop_assert!(approx_eq(actual, expected, Tolerance::STRICT));
        }
    }
}
