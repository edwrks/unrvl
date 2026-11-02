use crate::quantile::median;

const NORMAL_CONSISTENCY_FACTOR: f64 = 1.482_602_218_505_602;

/// Scaling convention for [`mad`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MadScale {
    /// Raw median absolute deviation, without a scaling factor.
    Raw,

    /// Normal-consistent scaling by `1 / Phi^-1(0.75)`.
    ///
    /// This is an asymptotic consistency factor for normally distributed
    /// observations, without a finite-sample correction.
    Normal,
}

/// Returns the median absolute deviation with the selected scaling.
///
/// Computes `median(|x - median(xs)|)` using R's default Type 7 interpolation
/// for both medians. [`MadScale::Raw`] returns this value unchanged, matching
/// R's `mad(xs, constant = 1)`. [`MadScale::Normal`] multiplies the completed
/// raw MAD by `1.482602218505602`, representing `1 / Phi^-1(0.75)`, where `Phi`
/// is the standard normal CDF. This gives asymptotic normal consistency,
/// without a finite-sample correction.
///
/// The input slice does not need to be sorted. Returns `NaN` if `xs` is empty
/// or contains a non-finite observation. A single observation or a constant
/// sample returns zero. A nonconstant sample can also have zero MAD.
///
/// Overflow in an outlying absolute deviation does not invalidate the result
/// when that deviation lies above the middle deviations. Normal scaling can
/// return positive infinity when the raw MAD is finite but the product
/// overflows.
///
/// For `n` observations, uses `O(n log n)` time and `O(n)` additional memory.
///
/// # Panics
///
/// Panics if `xs.len() - 1` exceeds `2^53`.
///
/// # Examples
///
/// ```
/// # use unrvl_stats::dispersion::{MadScale, mad};
/// let xs = [1.0, 2.0, 3.0, 6.0];
/// assert_eq!(mad(&xs, MadScale::Raw), 1.0);
/// assert!((mad(&xs, MadScale::Normal) - 1.482602218505602).abs() < 1e-14);
/// ```
#[must_use]
pub fn mad(xs: &[f64], scale: MadScale) -> f64 {
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

    let raw = median(&deviations[lower..=upper]);

    match scale {
        MadScale::Raw => raw,
        MadScale::Normal => raw * NORMAL_CONSISTENCY_FACTOR,
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_utils::approx::{Tolerance, approx_eq};
    use test_utils::strategies::{
        constant_sample, exact_translation_case, finite_sample, nonfinite_value,
        positive_power_of_two_scale,
    };

    use super::*;

    const SCALES: [MadScale; 2] = [MadScale::Raw, MadScale::Normal];

    #[test]
    fn returns_nan_for_empty_input() {
        let xs = [];

        for scale in SCALES {
            let result = mad(&xs, scale);

            assert!(result.is_nan());
        }
    }

    #[test]
    fn returns_nan_for_nan_input() {
        let xs = [1.0, f64::NAN, 3.0];

        for scale in SCALES {
            let result = mad(&xs, scale);

            assert!(result.is_nan());
        }
    }

    #[test]
    fn returns_nan_for_positive_infinity() {
        let xs = [1.0, f64::INFINITY, 3.0];

        for scale in SCALES {
            let result = mad(&xs, scale);

            assert!(result.is_nan());
        }
    }

    #[test]
    fn returns_nan_for_negative_infinity() {
        let xs = [1.0, f64::NEG_INFINITY, 3.0];

        for scale in SCALES {
            let result = mad(&xs, scale);

            assert!(result.is_nan());
        }
    }

    #[test]
    fn returns_nan_for_nonfinite_singletons() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let xs = [value];

            for scale in SCALES {
                let result = mad(&xs, scale);

                assert!(result.is_nan());
            }
        }
    }

    #[test]
    fn returns_zero_for_singleton() {
        let xs = [4.0];

        for scale in SCALES {
            let result = mad(&xs, scale);

            assert_eq!(result, 0.0);
        }
    }

    #[test]
    fn returns_zero_for_constant_extreme_samples() {
        for value in [-f64::MAX, f64::MAX] {
            let xs = [value, value, value];

            for scale in SCALES {
                let result = mad(&xs, scale);

                assert_eq!(result, 0.0);
            }
        }
    }

    #[test]
    fn returns_zero_for_signed_zeros() {
        let xs = [-0.0, 0.0, -0.0, 0.0];

        for scale in SCALES {
            let result = mad(&xs, scale);

            assert_eq!(result, 0.0);
        }
    }

    #[test]
    fn computes_mad_for_unsorted_odd_sample() {
        let xs = [9.0, 1.0, 2.0, 2.0, 4.0];
        let cases = [
            (MadScale::Raw, 1.0),
            (MadScale::Normal, 1.482_602_218_505_602),
        ];

        for (scale, expected) in cases {
            let result = mad(&xs, scale);

            assert_eq!(result, expected);
        }
    }

    #[test]
    fn interpolates_both_medians_for_even_sample() {
        let xs = [8.0, 1.0, 4.0, 2.0];
        // Center = 3; sorted deviations = [1, 1, 2, 5].
        let cases = [
            (MadScale::Raw, 1.5),
            (MadScale::Normal, 2.223_903_327_758_403),
        ];

        for (scale, expected) in cases {
            let result = mad(&xs, scale);

            assert_eq!(result, expected);
        }
    }

    #[test]
    fn returns_half_the_distance_for_two_observations() {
        let xs = [-2.0, 8.0];
        let expected = 5.0;
        let result = mad(&xs, MadScale::Raw);

        assert_eq!(result, expected);
    }

    #[test]
    fn handles_negative_observations() {
        let xs = [-9.0, -1.0, -2.0, -2.0, -4.0];
        let expected = 1.0;
        let result = mad(&xs, MadScale::Raw);

        assert_eq!(result, expected);
    }

    #[test]
    fn nonconstant_sample_can_have_zero_mad() {
        let xs = [0.0, 1.0, 1.0, 1.0, 100.0];

        for scale in SCALES {
            let result = mad(&xs, scale);

            assert_eq!(result, 0.0);
        }
    }

    #[test]
    fn preserves_small_mad_at_large_offset() {
        let offset = 2.0_f64.powi(52);
        let xs = [offset, offset + 1.0, offset + 3.0];
        let expected = 1.0;
        let result = mad(&xs, MadScale::Raw);

        assert_eq!(result, expected);
    }

    #[test]
    fn normal_scaling_can_overflow_a_finite_raw_mad() {
        let xs = [-f64::MAX, f64::MAX];

        assert_eq!(mad(&xs, MadScale::Raw), f64::MAX);
        assert_eq!(mad(&xs, MadScale::Normal), f64::INFINITY);
    }

    #[test]
    fn returns_zero_when_outlying_deviation_overflows() {
        let xs = [-f64::MAX, f64::MAX, f64::MAX];

        for scale in SCALES {
            let result = mad(&xs, scale);

            assert_eq!(result, 0.0);
        }
    }

    #[test]
    fn preserves_small_mad_when_outlying_deviation_overflows() {
        let upper = f64::MAX;
        let lower = f64::from_bits(upper.to_bits() - 1);
        let xs = [-f64::MAX, lower, upper];
        let raw = 2.0_f64.powi(971);
        let cases = [
            (MadScale::Raw, raw),
            (MadScale::Normal, raw * 1.482_602_218_505_602),
        ];

        for (scale, expected) in cases {
            let result = mad(&xs, scale);

            assert_eq!(result, expected);
        }
    }

    #[test]
    fn interpolates_middle_deviations_when_outlying_deviation_overflows() {
        let half_max = f64::MAX / 2.0;
        let samples = [
            [-f64::MAX, half_max, half_max, f64::MAX],
            [-f64::MAX, -half_max, -half_max, f64::MAX],
        ];
        let raw = f64::MAX / 4.0;
        let cases = [
            (MadScale::Raw, raw),
            (MadScale::Normal, raw * 1.482_602_218_505_602),
        ];

        for xs in samples {
            for (scale, expected) in cases {
                let result = mad(&xs, scale);

                assert_eq!(result, expected);
            }
        }
    }

    #[test]
    fn preserves_smallest_subnormal_mad() {
        let tiny = f64::from_bits(1);
        let xs = [-tiny, 0.0, tiny];
        let expected = tiny;

        for scale in SCALES {
            let result = mad(&xs, scale);

            assert_eq!(result, expected);
        }
    }

    #[test]
    fn interpolates_subnormal_deviations() {
        let tiny = f64::from_bits(1);
        let xs = [-3.0 * tiny, -tiny, tiny, 3.0 * tiny];
        let cases = [(MadScale::Raw, 2.0 * tiny), (MadScale::Normal, 3.0 * tiny)];

        for (scale, expected) in cases {
            let result = mad(&xs, scale);

            assert_eq!(result, expected);
        }
    }

    proptest! {
        #[test]
        #[expect(clippy::float_cmp, reason = "exactly constant observations")]
        fn constant_sample_has_zero_mad(sample in constant_sample(1)) {
            let expected = 0.0;

            for scale in SCALES {
                let result = mad(&sample, scale);

                prop_assert_eq!(result, expected);
            }
        }

        #[test]
        fn bounded_sample_has_finite_nonnegative_mad(sample in finite_sample(1)) {
            for scale in SCALES {
                let result = mad(&sample, scale);

                prop_assert!(result.is_finite());
                prop_assert!(result >= 0.0);
            }
        }

        #[test]
        fn reversing_observations_preserves_mad(sample in finite_sample(1)) {
            let mut reversed = sample.clone();
            reversed.reverse();

            for scale in SCALES {
                let expected = mad(&sample, scale);
                let result = mad(&reversed, scale);

                prop_assert_eq!(result.to_bits(), expected.to_bits());
            }
        }

        #[test]
        #[expect(clippy::float_cmp, reason = "exactly representable medians and deviations")]
        fn exact_translation_preserves_mad(
            (sample, offset) in exact_translation_case(2),
        ) {
            let translated: Vec<f64> = sample.iter().map(|&x| x + offset).collect();

            for scale in SCALES {
                let expected = mad(&sample, scale);
                let result = mad(&translated, scale);

                prop_assert_eq!(result, expected);
            }
        }

        #[test]
        fn positive_scaling_scales_mad(
            sample in finite_sample(1),
            scale in positive_power_of_two_scale(),
        ) {
            let scaled: Vec<f64> = sample.iter().map(|&x| x * scale).collect();

            for mad_scale in SCALES {
                let expected = mad(&sample, mad_scale) * scale;
                let result = mad(&scaled, mad_scale);

                prop_assert!(approx_eq(result, expected, Tolerance::STRICT));
            }
        }

        #[test]
        fn negating_observations_preserves_mad(sample in finite_sample(1)) {
            let negated: Vec<f64> = sample.iter().map(|&x| -x).collect();

            for scale in SCALES {
                let expected = mad(&sample, scale);
                let result = mad(&negated, scale);

                prop_assert!(approx_eq(result, expected, Tolerance::STRICT));
            }
        }

        #[test]
        fn replication_preserves_mad(sample in finite_sample(1)) {
            let mut replicated = sample.clone();
            replicated.extend_from_slice(&sample);

            for scale in SCALES {
                let expected = mad(&sample, scale);
                let result = mad(&replicated, scale);

                prop_assert_eq!(result.to_bits(), expected.to_bits());
            }
        }

        #[test]
        fn nonfinite_observation_invalidates_sample(
            mut sample in finite_sample(1),
            value in nonfinite_value(),
            position in any::<usize>(),
        ) {
            let index = position % sample.len();
            sample[index] = value;

            for scale in SCALES {
                let result = mad(&sample, scale);

                prop_assert!(result.is_nan());
            }
        }
    }
}
