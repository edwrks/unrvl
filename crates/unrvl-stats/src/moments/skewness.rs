use unrvl_numerics::convert::usize_to_f64;
use unrvl_numerics::summation::NeumaierSum;

use crate::internal::deviations::scaled_deviations;

/// Returns the bias-corrected sample skewness G1 (Joanes & Gill 1998 type 2).
///
/// A perfectly symmetric sample has `G1 = 0`. Positive values indicate right
/// skew, while negative values indicate left skew.
///
/// The statistic is defined as:
///
/// ```text
///              n
/// G1 = ------------------- · ∑ zᵢ³
///        (n - 1)(n - 2)
/// ```
///
/// where `zᵢ` is the standardized deviation using the Bessel-corrected sample
/// standard deviation.
///
/// Before accumulating central powers, deviations are divided by the maximum
/// absolute deviation. This scaling cancels from the standardized result and
/// reduces avoidable overflow and underflow.
///
/// Returns `NaN` for samples shorter than three observations, constant samples,
/// non-finite input, or when centering produces non-finite deviations.
///
/// Scaling the deviations does not prevent overflow while computing the mean
/// or forming the centered deviations for values near the limits of `f64`.
///
/// # Panics
///
/// Panics if the slice length exceeds `2^53`.
#[must_use]
pub fn skewness(xs: &[f64]) -> f64 {
    if xs.len() < 3 {
        return f64::NAN;
    }

    let Some((scale, deviations)) = scaled_deviations(xs) else {
        return f64::NAN;
    };

    if scale == 0.0 {
        return f64::NAN;
    }

    let mut sum_u2 = NeumaierSum::new();
    let mut sum_u3 = NeumaierSum::new();

    for u in deviations {
        let u2 = u * u;

        sum_u2 += u2;
        sum_u3 += u2 * u;
    }

    let sum_u2 = sum_u2.total();
    let sum_u3 = sum_u3.total();

    let n = usize_to_f64(xs.len());
    let correction = n / ((n - 1.0) * (n - 2.0));
    let scaled_std_dev = (sum_u2 / (n - 1.0)).sqrt();

    correction * sum_u3 / scaled_std_dev.powi(3)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_utils::approx::{Tolerance, approx_eq, assert_approx_eq};
    use test_utils::strategies::{finite_nonconstant_samples, finite_value, power_of_two_scale};

    use super::*;

    #[test]
    fn returns_nan_for_empty_input() {
        let xs = &[];
        let result = skewness(xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_singleton() {
        let xs = &[1.0];
        let result = skewness(xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_two_observations() {
        let xs = &[1.0, 2.0];
        let result = skewness(xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_nan_input() {
        let xs = &[1.0, 2.0, f64::NAN];
        let result = skewness(xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_positive_infinity() {
        let xs = &[1.0, 2.0, f64::INFINITY];
        let result = skewness(xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_negative_infinity() {
        let xs = &[1.0, 2.0, f64::NEG_INFINITY];
        let result = skewness(xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_zero_variance() {
        let xs = &[1.0, 1.0, 1.0];
        let result = skewness(xs);

        assert!(result.is_nan());
    }

    #[test]
    fn is_zero_for_symmetric_sample() {
        let xs = &[1.0, 2.0, 3.0, 4.0, 5.0];
        let result = skewness(xs);

        assert_approx_eq(result, 0.0, Tolerance::VERY_STRICT);
    }

    #[test]
    fn matches_known_right_skewed_sample() {
        let xs = &[1.0, 1.0, 2.0];
        let result = skewness(xs);
        let expected = 3.0_f64.sqrt();

        assert_approx_eq(result, expected, Tolerance::VERY_STRICT);
    }

    #[test]
    fn matches_known_left_skewed_sample() {
        let xs = &[1.0, 2.0, 2.0];
        let result = skewness(xs);
        let expected = -3.0_f64.sqrt();

        assert_approx_eq(result, expected, Tolerance::VERY_STRICT);
    }

    #[test]
    fn remains_finite_at_large_scale() {
        let xs = [1.0, 1.0, 2.0, 4.0, 10.0];
        let scaled = xs.map(|x| x * 1e110);

        let result = skewness(&xs);
        let result_scaled = skewness(&scaled);

        assert!(result_scaled.is_finite());
        assert_approx_eq(result_scaled, result, Tolerance::DEFAULT);
    }

    #[test]
    fn remains_finite_at_small_scale() {
        let xs = [1.0, 1.0, 2.0, 4.0, 10.0];
        let scaled = xs.map(|x| x * 1e-110);

        let result = skewness(&xs);
        let result_scaled = skewness(&scaled);

        assert!(result.is_finite());
        assert_approx_eq(result_scaled, result, Tolerance::DEFAULT);
    }

    proptest! {
        #[test]
        fn skewness_is_translation_invariant(
            sample in finite_nonconstant_samples(3),
            offset in finite_value(),
        ) {
            let translated: Vec<_> = sample.iter().map(|&x| x + offset).collect();

            let x = skewness(&sample);
            let y = skewness(&translated);

            prop_assert!(approx_eq(x, y, Tolerance::DEFAULT));
        }

        #[test]
        fn skewness_is_positive_scale_invariant(
            sample in finite_nonconstant_samples(3),
            scale in power_of_two_scale(),
        ) {
            let scaled: Vec<_> = sample.iter().map(|&x| x * scale).collect();

            let x = skewness(&sample);
            let y = skewness(&scaled);

            prop_assert!(approx_eq(x, y, Tolerance::DEFAULT));
        }

        #[test]
        fn skewness_flips_sign_under_negative_scaling(
            sample in finite_nonconstant_samples(3),
            scale in power_of_two_scale(),
        ) {
            let reflected: Vec<_> = sample.iter().map(|&x| x * -scale).collect();

            let x = skewness(&sample);
            let y = skewness(&reflected);

            prop_assert!(approx_eq(y, -x, Tolerance::DEFAULT));
        }

        #[test]
        fn skewness_is_reversal_invariant(sample in finite_nonconstant_samples(3)) {
            let mut reversed = sample.clone();
            reversed.reverse();

            let x = skewness(&sample);
            let y = skewness(&reversed);

            prop_assert!(approx_eq(x, y, Tolerance::DEFAULT));
        }

        #[test]
        fn skewness_is_finite_for_finite_nonconstant_samples(
            sample in finite_nonconstant_samples(3),
        ) {
            prop_assert!(skewness(&sample).is_finite());
        }
    }
}
