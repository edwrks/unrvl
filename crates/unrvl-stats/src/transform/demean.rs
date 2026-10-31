use crate::internal::deviations::CenteredSample;

/// Returns each observation's deviation from the sample mean, in input order.
///
/// Results have the same units as the observations. An empty slice returns an
/// empty vector. Finite singleton and constant samples return zeros. If any
/// observation is non-finite, returns one `NaN` per input observation.
///
/// Uses scaled, corrected centering to reduce rounding error. Results may
/// differ from subtracting the rounded mean returned by
/// [`crate::moments::mean`]. Rounding and underflow remain possible; very small
/// contributions relative to the input scale can be lost. Deviations outside
/// the finite `f64` range overflow to signed infinity.
///
/// Uses `O(n)` time and `O(n)` memory for the returned vector. Does not modify
/// the input observations.
///
/// # Panics
///
/// Panics if a finite sample contains more than `2^53` observations.
///
/// # Examples
///
/// ```
/// # use unrvl_stats::transform::demean;
/// let xs = [1002.0, 1000.0, 1001.0];
/// assert_eq!(demean(&xs), vec![1.0, -1.0, 0.0]);
/// ```
#[must_use]
pub fn demean(xs: &[f64]) -> Vec<f64> {
    let Some(sample) = CenteredSample::new(xs) else {
        return vec![f64::NAN; xs.len()];
    };

    let unit = sample.centering().unit();

    sample
        .deviations()
        .map(|deviation| deviation * unit)
        .collect()
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_utils::approx::{Tolerance, approx_eq, assert_all_approx_eq};
    use test_utils::strategies::{
        exact_translation_case, finite_sample, positive_power_of_two_scale,
    };
    use unrvl_numerics::convert::usize_to_f64;
    use unrvl_numerics::summation::NeumaierSumExt;

    use super::demean;

    #[test]
    fn preserves_observation_order_and_original_units() {
        let xs = [1002.0, 1000.0, 1001.0];

        let actual = demean(&xs);
        let expected = [1.0, -1.0, 0.0];

        assert_eq!(actual, expected);
    }

    #[test]
    fn empty_input_returns_an_empty_vector() {
        let actual = demean(&[]);

        assert_eq!(actual, [] as [f64; 0]);
    }

    #[test]
    fn singleton_returns_zero() {
        let actual = demean(&[42.0]);

        assert_eq!(actual, vec![0.0]);
    }

    #[test]
    fn finite_constant_samples_return_zeros() {
        let values = [0.0, -0.0, 42.0, f64::from_bits(1), f64::MAX, -f64::MAX];

        for value in values {
            let xs = [value; 3];

            let actual = demean(&xs);

            assert_eq!(actual, vec![0.0; 3]);
        }
    }

    #[test]
    fn nonfinite_observations_make_all_results_nan() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let xs = [1.0, value, 3.0];

            let actual = demean(&xs);

            assert_eq!(actual.len(), xs.len());
            assert!(actual.into_iter().all(f64::is_nan));
        }
    }

    #[test]
    fn preserves_mean_corrections_around_a_large_offset() {
        let xs = [1e16, 1e16, 1e16 + 2.0];

        let actual = demean(&xs);
        let expected = [-2.0 / 3.0, -2.0 / 3.0, 4.0 / 3.0];

        // The true mean lies between representable f64 values. Keeping its
        // correction avoids the [0, 0, 2] from subtracting a rounded mean.
        assert_all_approx_eq(&actual, &expected, Tolerance::VERY_STRICT);
    }

    #[test]
    fn restores_finite_deviations_from_extreme_observations() {
        let max = f64::MAX;
        let cases = [
            ([-max, 0.0, max], [-max, 0.0, max]),
            ([max, max / 2.0, 0.0], [max / 2.0, 0.0, -max / 2.0]),
        ];

        for (xs, expected) in cases {
            let actual = demean(&xs);

            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn preserves_subnormal_deviations() {
        let tiny = f64::from_bits(1);
        let xs = [0.0, tiny, 2.0 * tiny];

        let actual = demean(&xs);
        let expected = [-tiny, 0.0, tiny];

        assert_eq!(actual, expected);
    }

    #[test]
    fn unrepresentable_deviations_overflow_to_signed_infinity() {
        let xs = [-f64::MAX, f64::MAX, f64::MAX];
        let reflected = xs.map(|x| -x);

        let negative = demean(&xs);
        let positive = demean(&reflected);

        assert_eq!(negative[0], f64::NEG_INFINITY);
        assert_eq!(positive[0], f64::INFINITY);
        assert!(negative[1..].iter().all(|value| value.is_finite()));
        assert!(positive[1..].iter().all(|value| value.is_finite()));
    }

    proptest! {
        #[test]
        fn deviations_sum_to_zero_up_to_rounding(xs in finite_sample(1)) {
            let actual = demean(&xs);

            let total = actual.iter().neumaier_sum();
            let n = usize_to_f64(xs.len());
            let scale = xs.iter().map(|value| value.abs()).fold(1.0, f64::max);

            // Allow per-observation rounding at the sample's magnitude.
            let bound = Tolerance::STRICT.rel() * n * scale;

            prop_assert_eq!(actual.len(), xs.len());
            prop_assert!(total.abs() <= bound);
        }

        #[test]
        fn exact_translation_preserves_deviations(
            (xs, offset) in exact_translation_case(2),
        ) {
            let translated: Vec<_> = xs.iter().map(|&value| value + offset).collect();

            let expected = demean(&xs);
            let actual = demean(&translated);

            // Translation can change centering roundoff. Small deviations
            // need an absolute tolerance based on the input magnitude.
            let input_scale = xs.iter()
                .chain(&translated)
                .map(|value| value.abs())
                .fold(1.0, f64::max);

            let tolerance = Tolerance::new(
                Tolerance::STRICT.rel() * input_scale,
                Tolerance::STRICT.rel(),
            );

            prop_assert_eq!(actual.len(), expected.len());

            for (&actual, &expected) in actual.iter().zip(&expected) {
                prop_assert!(approx_eq(actual, expected, tolerance));
            }
        }

        #[test]
        fn power_of_two_scaling_scales_deviations(
            xs in finite_sample(1),
            scale in positive_power_of_two_scale(),
        ) {
            let scaled: Vec<_> = xs.iter().map(|&value| value * scale).collect();

            let expected: Vec<_> = demean(&xs).iter().map(|value| value * scale).collect();
            let actual = demean(&scaled);

            prop_assert_eq!(actual.len(), expected.len());

            for (&actual, &expected) in actual.iter().zip(&expected) {
                prop_assert!(approx_eq(actual, expected, Tolerance::STRICT));
            }
        }
    }
}
