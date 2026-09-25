use unrvl_numerics::extrema::min_max;

/// Returns the range, the maximum observation minus the minimum.
///
/// The input slice does not need to be sorted. Returns `NaN` if `xs` is empty
/// or contains a non-finite observation. A single observation or a constant
/// sample returns zero.
///
/// The difference can overflow to positive infinity even when all observations
/// are finite.
#[must_use]
pub fn range(xs: &[f64]) -> f64 {
    if xs.is_empty() || xs.iter().any(|x| !x.is_finite()) {
        return f64::NAN;
    }

    let (min, max) = min_max(xs);

    max - min
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_utils::approx::{Tolerance, approx_eq};
    use test_utils::strategies::{
        constant_sample, exact_translation_case, finite_sample, finite_value,
        positive_power_of_two_scale,
    };

    use super::*;

    #[test]
    fn returns_nan_for_empty_input() {
        let xs = [];
        let result = range(&xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_nan_input() {
        let xs = [1.0, f64::NAN, 3.0];
        let result = range(&xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_positive_infinity() {
        let xs = [1.0, f64::INFINITY, 3.0];
        let result = range(&xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_negative_infinity() {
        let xs = [1.0, f64::NEG_INFINITY, 3.0];
        let result = range(&xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_nonfinite_singletons() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let xs = [value];
            let result = range(&xs);

            assert!(result.is_nan());
        }
    }

    #[test]
    fn returns_zero_for_singleton() {
        let xs = [4.0];
        let result = range(&xs);

        assert_eq!(result, 0.0);
    }

    #[test]
    fn returns_zero_for_constant_extreme_samples() {
        for value in [-f64::MAX, f64::MAX] {
            let xs = [value, value, value];
            let result = range(&xs);

            assert_eq!(result, 0.0);
        }
    }

    #[test]
    fn returns_positive_zero_for_signed_zeros() {
        let samples: &[&[f64]] = &[&[-0.0], &[0.0], &[-0.0, 0.0], &[0.0, -0.0]];
        let expected = 0.0_f64;

        for xs in samples {
            let result = range(xs);

            assert_eq!(result.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn handles_unsorted_input_with_repeated_extrema() {
        let xs = [3.0, 7.0, 1.0, 4.0, 1.0, 7.0];
        let expected = 6.0;
        let result = range(&xs);

        assert_eq!(result, expected);
    }

    #[test]
    fn handles_negative_observations() {
        let xs = [-5.0, -1.0, -3.0];
        let expected = 4.0;
        let result = range(&xs);

        assert_eq!(result, expected);
    }

    #[test]
    fn handles_observations_across_zero() {
        let xs = [-3.0, 0.0, 5.0];
        let expected = 8.0;
        let result = range(&xs);

        assert_eq!(result, expected);
    }

    #[test]
    fn preserves_small_range_at_large_offset() {
        let offset = 2.0_f64.powi(52);
        let xs = [offset + 3.0, offset, offset + 1.0];
        let expected = 3.0;
        let result = range(&xs);

        assert_eq!(result, expected);
    }

    #[test]
    fn preserves_spacing_near_maximum_finite_value() {
        let upper = f64::MAX;
        let lower = f64::from_bits(upper.to_bits() - 1);
        let xs = [upper, lower];
        let expected = 2.0_f64.powi(971);
        let result = range(&xs);

        assert_eq!(result, expected);
    }

    #[test]
    fn returns_finite_range_for_large_opposite_endpoints() {
        let half_max = f64::MAX / 2.0;
        let xs = [-half_max, half_max];
        let result = range(&xs);

        assert_eq!(result, f64::MAX);
    }

    #[test]
    fn returns_positive_infinity_when_range_overflows() {
        let xs = [-f64::MAX, f64::MAX];
        let result = range(&xs);

        assert_eq!(result, f64::INFINITY);
    }

    #[test]
    fn preserves_smallest_subnormal_range() {
        let tiny = f64::from_bits(1);
        let xs = [tiny, 2.0 * tiny];
        let expected = tiny;
        let result = range(&xs);

        assert_eq!(result, expected);
    }

    #[test]
    fn preserves_subnormal_range_across_zero() {
        let tiny = f64::from_bits(1);
        let xs = [-tiny, tiny];
        let expected = 2.0 * tiny;
        let result = range(&xs);

        assert_eq!(result, expected);
    }

    proptest! {
        #[test]
        #[expect(clippy::float_cmp, reason = "exactly constant observations")]
        fn constant_sample_has_zero_range(sample in constant_sample(1)) {
            let result = range(&sample);
            let expected = 0.0;

            prop_assert_eq!(result, expected);
        }

        #[test]
        fn bounded_sample_has_finite_nonnegative_range(sample in finite_sample(1)) {
            let result = range(&sample);

            prop_assert!(result.is_finite());
            prop_assert!(result >= 0.0);
        }

        #[test]
        fn reversing_observations_preserves_range(sample in finite_sample(1)) {
            let expected = range(&sample);
            let mut reversed = sample;
            reversed.reverse();
            let result = range(&reversed);

            prop_assert_eq!(result.to_bits(), expected.to_bits());
        }

        #[test]
        #[expect(clippy::float_cmp, reason = "exactly representable differences")]
        fn exact_translation_preserves_range(
            (sample, offset) in exact_translation_case(2),
        ) {
            let translated: Vec<f64> = sample.iter().map(|&x| x + offset).collect();
            let expected = range(&sample);
            let result = range(&translated);

            prop_assert_eq!(result, expected);
        }

        #[test]
        fn positive_scaling_scales_range(
            sample in finite_sample(1),
            scale in positive_power_of_two_scale(),
        ) {
            let scaled: Vec<f64> = sample.iter().map(|&x| x * scale).collect();
            let expected = range(&sample) * scale;
            let result = range(&scaled);

            prop_assert!(approx_eq(result, expected, Tolerance::STRICT));
        }

        #[test]
        fn negating_observations_preserves_range(sample in finite_sample(1)) {
            let negated: Vec<f64> = sample.iter().map(|&x| -x).collect();
            let expected = range(&sample);
            let result = range(&negated);

            prop_assert_eq!(result.to_bits(), expected.to_bits());
        }

        #[test]
        fn adding_an_observation_cannot_decrease_range(
            mut sample in finite_sample(1),
            value in finite_value(),
        ) {
            let original = range(&sample);
            sample.push(value);
            let result = range(&sample);

            prop_assert!(result >= original);
        }

        #[test]
        fn nonfinite_observation_invalidates_sample(
            mut sample in finite_sample(1),
            value in prop::sample::select(vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY]),
            position in any::<usize>(),
        ) {
            let index = position % sample.len();
            sample[index] = value;
            let result = range(&sample);

            prop_assert!(result.is_nan());
        }
    }
}
