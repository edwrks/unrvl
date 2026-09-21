use unrvl_numerics::convert::usize_to_f64;
use unrvl_numerics::summation::NeumaierSumExt;

use crate::moments::mean;

/// Centres observations while retaining the mean's rounding correction.
///
/// Adding the correction back to `center` can round it away again. Subtract it
/// from each deviation instead, before calculating moments or standard scores.
pub fn centered(xs: &[f64]) -> impl Iterator<Item = f64> + Clone + '_ {
    let n = usize_to_f64(xs.len());
    let center = mean(xs);
    let correction = xs.iter().map(|&x| x - center).neumaier_sum() / n;

    xs.iter().map(move |&x| (x - center) - correction)
}

/// Returns the maximum deviation magnitude and deviations divided by it.
///
/// Constant samples have scale zero and yield zero deviations. `None`
/// indicates non-finite centering; scaling does not extend the mean's range.
pub fn scaled_deviations(xs: &[f64]) -> Option<(f64, impl Iterator<Item = f64> + '_)> {
    let deviations = centered(xs);
    let mut scale: f64 = 0.0;

    for deviation in deviations.clone() {
        if !deviation.is_finite() {
            return None;
        }

        scale = scale.max(deviation.abs());
    }

    let divisor = if scale == 0.0 { 1.0 } else { scale };
    Some((scale, deviations.map(move |deviation| deviation / divisor)))
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_utils::approx::{Tolerance, approx_eq};
    use test_utils::strategies::{constant_samples, finite_samples};

    use super::*;

    mod centered {
        use super::*;

        #[test]
        fn returns_deviations_from_mean() {
            let xs = [1.0, 2.0, 3.0];
            let deviations: Vec<_> = centered(&xs).collect();

            assert_eq!(deviations, [-1.0, 0.0, 1.0]);
        }

        proptest! {
            #[test]
            fn rounding_correction_does_not_increase_residual(xs in finite_samples(1)) {
                let center = mean(&xs);

                let uncorrected = xs
                    .iter()
                    .map(|&x| x - center)
                    .neumaier_sum();

                let corrected = centered(&xs).neumaier_sum();

                prop_assert!(corrected.abs() <= uncorrected.abs());
            }

            #[test]
            fn constant_samples_are_exactly_zero(xs in constant_samples(1)) {
                let deviations: Vec<_> = centered(&xs).collect();

                prop_assert!(deviations.iter().all(|&x| x == 0.0));
            }
        }
    }

    mod scaled_deviations {
        use unrvl_numerics::extrema::max_abs;

        use super::*;

        #[test]
        fn uses_maximum_deviation_magnitude_as_scale() {
            let xs = [1000.0, 1001.0, 1002.0];
            let (scale, deviations) = scaled_deviations(&xs).unwrap();

            assert_eq!(scale, 1.0);
            assert_eq!(deviations.collect::<Vec<_>>(), [-1.0, 0.0, 1.0]);
        }

        #[test]
        fn rejects_non_finite_centring() {
            for non_finite in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let xs = [1.0, non_finite];

                assert!(scaled_deviations(&xs).is_none());
            }
        }

        proptest! {
            #[test]
            fn constant_sample_has_zero_scale_and_zero_deviations(xs in constant_samples(1)) {
                let (scale, deviations) = scaled_deviations(&xs).unwrap();

                assert_eq!(scale, 0.0);
                assert!(deviations.into_iter().all(|x| x == 0.0));
            }


            #[test]
            #[expect(clippy::float_cmp, reason = "normalisation must produce an exact unit maximum")]
            fn scaled_deviations_are_bounded_by_one(xs in finite_samples(1)) {
                let (scale, deviations) = scaled_deviations(&xs).unwrap();
                let deviations: Vec<_> = deviations.collect();
                let bounded = deviations.iter().all(|&x| x.is_finite() && x.abs() <= 1.0);

                prop_assert!(scale.is_finite());
                prop_assert!(scale >= 0.0);
                prop_assert!(bounded);

                if scale == 0.0 {
                    prop_assert!(deviations.iter().all(|&x| x == 0.0));
                } else {
                    let maximum = max_abs(&deviations);
                    // One deviation supplied the scale, so its normalized
                    // magnitude should be exactly one.
                    prop_assert_eq!(maximum, 1.0);
                }
            }

            #[test]
            fn scaling_can_reconstruct_centered_deviations(xs in finite_samples(1)) {
                let expected = centered(&xs);
                let (scale, actual) = scaled_deviations(&xs).unwrap();

                for (expected, scaled) in expected.into_iter().zip(actual) {
                    let reconstructed = scaled * scale;
                    let result = approx_eq(reconstructed, expected, Tolerance::VERY_STRICT);

                    prop_assert!(result, "expected {expected:e}, reconstructed {reconstructed:e}");
                }
            }
        }
    }
}
