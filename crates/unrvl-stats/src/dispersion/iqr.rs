use crate::quantile::quantiles;

/// Returns the interquartile range, `Q3 - Q1`.
///
/// `Q1` and `Q3` are the 0.25 and 0.75 quantiles, computed using R's default
/// Type 7 interpolation. The input slice does not need to be sorted.
///
/// Returns `NaN` if `xs` is empty or contains a non-finite observation. A
/// single observation or a constant sample returns zero.
///
/// The difference between the computed quartiles can overflow to positive
/// infinity even when all observations are finite.
///
/// # Panics
///
/// Panics if `xs.len() - 1` exceeds `2^53`.
#[must_use]
pub fn iqr(xs: &[f64]) -> f64 {
    let qs = quantiles(xs, &[0.25, 0.75]);
    debug_assert_eq!(qs.len(), 2);

    qs[1] - qs[0]
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_utils::approx::{Tolerance, approx_eq, assert_approx_eq};
    use test_utils::strategies::{
        constant_sample, exact_translation_case, finite_sample, positive_power_of_two_scale,
    };

    use super::*;

    #[test]
    fn returns_nan_for_empty_input() {
        let xs = [];
        let result = iqr(&xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_nan_input() {
        let xs = [1.0, f64::NAN, 3.0];
        let result = iqr(&xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_positive_infinity() {
        let xs = [1.0, f64::INFINITY, 3.0];
        let result = iqr(&xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_negative_infinity() {
        let xs = [1.0, f64::NEG_INFINITY, 3.0];
        let result = iqr(&xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_zero_for_singleton() {
        let xs = [4.0];
        let result = iqr(&xs);

        assert_eq!(result, 0.0);
    }

    #[test]
    fn returns_zero_for_constant_sample() {
        let xs = [2.0, 2.0, 2.0];
        let result = iqr(&xs);

        assert_eq!(result, 0.0);
    }

    #[test]
    fn handles_unsorted_input() {
        let xs = [5.0, 1.0, 4.0, 2.0, 3.0];
        let result = iqr(&xs);

        assert_eq!(result, 2.0);
    }

    #[test]
    fn uses_type7_quartiles() {
        let xs = [1.0, 2.0, 3.0, 4.0];
        let result = iqr(&xs);

        assert_eq!(result, 1.5);
    }

    #[test]
    fn interpolates_quartiles_for_two_observations() {
        let xs = [0.0, 10.0];
        let result = iqr(&xs);

        assert_eq!(result, 5.0);
    }

    #[test]
    fn nonconstant_sample_can_have_zero_iqr() {
        let xs = [0.0, 1.0, 1.0, 1.0, 2.0];
        let result = iqr(&xs);

        assert_eq!(result, 0.0);
    }

    #[test]
    fn returns_finite_iqr_when_full_range_overflows() {
        let xs = [-f64::MAX, f64::MAX];
        let result = iqr(&xs);

        assert!(result.is_finite());
        assert_approx_eq(result, f64::MAX, Tolerance::STRICT);
    }

    #[test]
    fn returns_positive_infinity_when_quartile_difference_overflows() {
        let xs = [-f64::MAX, -f64::MAX, f64::MAX, f64::MAX];
        let result = iqr(&xs);

        assert_eq!(result, f64::INFINITY);
    }

    #[test]
    fn preserves_representable_subnormal_iqr() {
        let tiny = f64::from_bits(1);
        let xs = [-2.0 * tiny, 2.0 * tiny];
        let result = iqr(&xs);

        assert_eq!(result, 2.0 * tiny);
    }

    proptest! {
        #[test]
        #[expect(clippy::float_cmp, reason = "exactly constant observations")]
        fn constant_sample_has_zero_iqr(sample in constant_sample(1)) {
            let result = iqr(&sample);
            let expected = 0.0;

            prop_assert_eq!(result, expected);
        }

        #[test]
        fn bounded_sample_has_finite_nonnegative_iqr(sample in finite_sample(1)) {
            let result = iqr(&sample);

            prop_assert!(result.is_finite());
            prop_assert!(result >= 0.0);
        }

        #[test]
        fn reversing_observations_preserves_iqr(sample in finite_sample(1)) {
            let expected = iqr(&sample);
            let mut reversed = sample;
            reversed.reverse();

            prop_assert_eq!(iqr(&reversed).to_bits(), expected.to_bits());
        }

        #[test]
        #[expect(clippy::float_cmp, reason = "exactly representable quartiles")]
        fn exact_translation_preserves_iqr(
            (sample, offset) in exact_translation_case(2),
        ) {
            let translated: Vec<f64> = sample
                .iter()
                .map(|&x| x + offset)
                .collect();

            prop_assert_eq!(iqr(&translated), iqr(&sample));
        }

        #[test]
        fn positive_scaling_scales_iqr(
            sample in finite_sample(1),
            scale in positive_power_of_two_scale(),
        ) {
            let scaled: Vec<f64> = sample
                .iter()
                .map(|&x| x * scale)
                .collect();

            prop_assert!(approx_eq(
                iqr(&scaled),
                iqr(&sample) * scale,
                Tolerance::STRICT,
            ));
        }
    }
}
