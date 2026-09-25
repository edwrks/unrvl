use crate::quantile::median;

/// Returns the raw median absolute deviation from the sample median.
///
/// Computes `median(|x - median(xs)|)` using R's default Type 7 interpolation
/// for both medians. No normal-consistency scaling factor is applied; this
/// matches R's `mad(xs, constant = 1)`.
///
/// The input slice does not need to be sorted. Returns `NaN` if `xs` is empty
/// or contains a non-finite observation. A single observation or a constant
/// sample returns zero. A nonconstant sample can also have zero MAD.
///
/// Overflow in an outlying absolute deviation does not invalidate the result
/// when that deviation lies above the middle deviations.
///
/// # Panics
///
/// Panics if `xs.len() - 1` exceeds `2^53`.
#[must_use]
pub fn mad(xs: &[f64]) -> f64 {
    let center = median(xs);
    if !center.is_finite() {
        return f64::NAN;
    }

    let mut deviations: Vec<f64> = xs.iter().map(|&x| (x - center).abs()).collect();
    deviations.sort_unstable_by(f64::total_cmp);

    // An outlying deviation may overflow even though the middle deviations
    // are finite. Pass only the middle value or pair to the finite-input
    // median.
    let lower = (deviations.len() - 1) / 2;
    let upper = deviations.len() / 2;

    median(&deviations[lower..=upper])
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_utils::approx::{Tolerance, approx_eq};
    use test_utils::strategies::{
        constant_sample, exact_translation_case, finite_sample, positive_power_of_two_scale,
    };

    use super::*;

    #[test]
    fn returns_nan_for_empty_input() {
        let xs = [];
        let result = mad(&xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_nan_input() {
        let xs = [1.0, f64::NAN, 3.0];
        let result = mad(&xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_positive_infinity() {
        let xs = [1.0, f64::INFINITY, 3.0];
        let result = mad(&xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_negative_infinity() {
        let xs = [1.0, f64::NEG_INFINITY, 3.0];
        let result = mad(&xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_nonfinite_singletons() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let xs = [value];
            let result = mad(&xs);

            assert!(result.is_nan());
        }
    }

    #[test]
    fn returns_zero_for_singleton() {
        let xs = [4.0];
        let result = mad(&xs);

        assert_eq!(result, 0.0);
    }

    #[test]
    fn returns_zero_for_constant_extreme_samples() {
        for value in [-f64::MAX, f64::MAX] {
            let xs = [value, value, value];
            let result = mad(&xs);

            assert_eq!(result, 0.0);
        }
    }

    #[test]
    fn returns_zero_for_signed_zeros() {
        let xs = [-0.0, 0.0, -0.0, 0.0];
        let result = mad(&xs);

        assert_eq!(result, 0.0);
    }

    #[test]
    fn computes_raw_mad_for_unsorted_odd_sample() {
        let xs = [9.0, 1.0, 2.0, 2.0, 4.0];
        let expected = 1.0;
        let result = mad(&xs);

        assert_eq!(result, expected);
    }

    #[test]
    fn interpolates_both_medians_for_even_sample() {
        let xs = [8.0, 1.0, 4.0, 2.0];
        // Center = 3; sorted deviations = [1, 1, 2, 5].
        let expected = 1.5;
        let result = mad(&xs);

        assert_eq!(result, expected);
    }

    #[test]
    fn returns_half_the_distance_for_two_observations() {
        let xs = [-2.0, 8.0];
        let expected = 5.0;
        let result = mad(&xs);

        assert_eq!(result, expected);
    }

    #[test]
    fn handles_negative_observations() {
        let xs = [-9.0, -1.0, -2.0, -2.0, -4.0];
        let expected = 1.0;
        let result = mad(&xs);

        assert_eq!(result, expected);
    }

    #[test]
    fn nonconstant_sample_can_have_zero_mad() {
        let xs = [0.0, 1.0, 1.0, 1.0, 100.0];
        let result = mad(&xs);

        assert_eq!(result, 0.0);
    }

    #[test]
    fn preserves_small_mad_at_large_offset() {
        let offset = 2.0_f64.powi(52);
        let xs = [offset, offset + 1.0, offset + 3.0];
        let expected = 1.0;
        let result = mad(&xs);

        assert_eq!(result, expected);
    }

    #[test]
    fn returns_finite_mad_when_full_range_overflows() {
        let xs = [-f64::MAX, f64::MAX];
        let expected = f64::MAX;
        let result = mad(&xs);

        assert_eq!(result, expected);
    }

    #[test]
    fn returns_zero_when_outlying_deviation_overflows() {
        let xs = [-f64::MAX, f64::MAX, f64::MAX];
        let result = mad(&xs);

        assert_eq!(result, 0.0);
    }

    #[test]
    fn preserves_small_mad_when_outlying_deviation_overflows() {
        let upper = f64::MAX;
        let lower = f64::from_bits(upper.to_bits() - 1);
        let xs = [-f64::MAX, lower, upper];
        let expected = 2.0_f64.powi(971);
        let result = mad(&xs);

        assert_eq!(result, expected);
    }

    #[test]
    fn interpolates_middle_deviations_when_outlying_deviation_overflows() {
        let half_max = f64::MAX / 2.0;
        let samples = [
            [-f64::MAX, half_max, half_max, f64::MAX],
            [-f64::MAX, -half_max, -half_max, f64::MAX],
        ];
        let expected = f64::MAX / 4.0;

        for xs in samples {
            let result = mad(&xs);

            assert_eq!(result, expected);
        }
    }

    #[test]
    fn preserves_smallest_subnormal_mad() {
        let tiny = f64::from_bits(1);
        let xs = [-tiny, 0.0, tiny];
        let expected = tiny;
        let result = mad(&xs);

        assert_eq!(result, expected);
    }

    #[test]
    fn interpolates_subnormal_deviations() {
        let tiny = f64::from_bits(1);
        let xs = [-3.0 * tiny, -tiny, tiny, 3.0 * tiny];
        let expected = 2.0 * tiny;
        let result = mad(&xs);

        assert_eq!(result, expected);
    }

    proptest! {
        #[test]
        #[expect(clippy::float_cmp, reason = "exactly constant observations")]
        fn constant_sample_has_zero_mad(sample in constant_sample(1)) {
            let expected = 0.0;
            let result = mad(&sample);

            prop_assert_eq!(result, expected);
        }

        #[test]
        fn bounded_sample_has_finite_nonnegative_mad(sample in finite_sample(1)) {
            let result = mad(&sample);

            prop_assert!(result.is_finite());
            prop_assert!(result >= 0.0);
        }

        #[test]
        fn reversing_observations_preserves_mad(sample in finite_sample(1)) {
            let expected = mad(&sample);
            let mut reversed = sample;
            reversed.reverse();
            let result = mad(&reversed);

            prop_assert_eq!(result.to_bits(), expected.to_bits());
        }

        #[test]
        #[expect(clippy::float_cmp, reason = "exactly representable medians and deviations")]
        fn exact_translation_preserves_mad(
            (sample, offset) in exact_translation_case(2),
        ) {
            let translated: Vec<f64> = sample.iter().map(|&x| x + offset).collect();
            let expected = mad(&sample);
            let result = mad(&translated);

            prop_assert_eq!(result, expected);
        }

        #[test]
        fn positive_scaling_scales_mad(
            sample in finite_sample(1),
            scale in positive_power_of_two_scale(),
        ) {
            let scaled: Vec<f64> = sample.iter().map(|&x| x * scale).collect();
            let expected = mad(&sample) * scale;
            let result = mad(&scaled);

            prop_assert!(approx_eq(result, expected, Tolerance::STRICT));
        }

        #[test]
        fn negating_observations_preserves_mad(sample in finite_sample(1)) {
            let negated: Vec<f64> = sample.iter().map(|&x| -x).collect();
            let expected = mad(&sample);
            let result = mad(&negated);

            prop_assert!(approx_eq(result, expected, Tolerance::STRICT));
        }

        #[test]
        fn replication_preserves_mad(sample in finite_sample(1)) {
            let expected = mad(&sample);
            let mut replicated = sample.clone();
            replicated.extend_from_slice(&sample);
            let result = mad(&replicated);

            prop_assert_eq!(result.to_bits(), expected.to_bits());
        }

        #[test]
        fn nonfinite_observation_invalidates_sample(
            mut sample in finite_sample(1),
            value in prop::sample::select(vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY]),
            position in any::<usize>(),
        ) {
            let index = position % sample.len();
            sample[index] = value;
            let result = mad(&sample);

            prop_assert!(result.is_nan());
        }
    }
}
