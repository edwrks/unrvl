use unrvl_numerics::convert::usize_to_f64;
use unrvl_numerics::summation::NeumaierSum;

use crate::internal::deviations::centered_scaled;

/// Returns the bias-corrected sample skewness G1 (Joanes & Gill 1998 type 2).
///
/// A nonconstant, symmetric sample has zero mathematical skewness. Positive
/// values indicate right skew, while negative values indicate left skew.
/// Floating-point rounding can produce a small nonzero result for a
/// symmetric sample.
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
/// Observations are divided by a power-of-two unit before centering. Centering
/// retains a correction for mean rounding, and central powers are accumulated
/// with Neumaier compensated summation. The scaling cancels from the
/// standardized result and reduces avoidable overflow and underflow.
///
/// Scaling does not eliminate floating-point rounding. When observations span
/// an extreme range of magnitudes, very small scaled values can underflow to
/// zero.
///
/// Returns `NaN` for samples shorter than three observations, constant samples,
/// or input containing non-finite values.
///
/// # Panics
///
/// Panics if a finite sample contains more than `2^53` observations.
#[must_use]
pub fn skewness(xs: &[f64]) -> f64 {
    if xs.len() < 3 {
        return f64::NAN;
    }

    let Some((_, deviations)) = centered_scaled(xs) else {
        return f64::NAN;
    };

    let mut sum_u2 = NeumaierSum::new();
    let mut sum_u3 = NeumaierSum::new();

    for u in deviations {
        let u2 = u * u;

        sum_u2 += u2;
        sum_u3 += u2 * u;
    }

    let sum_u2 = sum_u2.total();
    let sum_u3 = sum_u3.total();

    if sum_u2 == 0.0 {
        return f64::NAN;
    }

    let n = usize_to_f64(xs.len());
    let correction = n / ((n - 1.0) * (n - 2.0));
    let scaled_std_dev = (sum_u2 / (n - 1.0)).sqrt();

    correction * sum_u3 / scaled_std_dev.powi(3)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_utils::approx::{Tolerance, approx_eq, assert_approx_eq};
    use test_utils::strategies::{
        exact_translation_case, finite_nonconstant_sample, positive_power_of_two_scale,
    };

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
    fn returns_nan_for_constant_extreme_samples() {
        let values = [f64::from_bits(1), -f64::from_bits(1), f64::MAX, -f64::MAX];
        for value in values {
            assert!(skewness(&[value; 3]).is_nan());
        }
    }

    #[test]
    fn returns_nan_for_translation_that_rounds_to_a_constant() {
        let e = f64::EPSILON;
        let xs = [1.0, 1.0 + e, 2.0f64.mul_add(e, 1.0)];
        let translated = xs.map(|x| x + 100.0);

        assert_approx_eq(skewness(&xs), 0.0, Tolerance::STRICT);

        // The translation erased the distinctions between observations.
        assert_eq!(translated, [101.0; 3]);
        assert!(skewness(&translated).is_nan());
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

        assert!(result_scaled.is_finite());
        assert_approx_eq(result_scaled, result, Tolerance::DEFAULT);
    }

    #[test]
    fn preserves_shape_at_the_smallest_positive_subnormal_scale() {
        let d = f64::from_bits(1);
        let xs = [0.0, 0.0, d];
        let result = skewness(&xs);
        let expected = 3.0_f64.sqrt();

        assert_approx_eq(result, expected, Tolerance::STRICT);
    }

    #[test]
    fn preserves_shape_at_the_smallest_negative_subnormal_scale() {
        let d = f64::from_bits(1);
        let xs = [0.0, 0.0, -d];
        let result = skewness(&xs);
        let expected = -3.0_f64.sqrt();

        assert_approx_eq(result, expected, Tolerance::STRICT);
    }

    #[test]
    fn remains_finite_when_the_unscaled_sum_would_overflow() {
        let half_max = f64::MAX / 2.0;
        let xs = [half_max, half_max, f64::MAX];
        let result = skewness(&xs);
        let expected = 3.0_f64.sqrt();

        assert_approx_eq(result, expected, Tolerance::STRICT);
    }

    #[test]
    fn preserves_shape_around_a_large_offset() {
        let xs = [1e16, 1e16, 1e16 + 2.0];
        let result = skewness(&xs);
        let expected = 3.0_f64.sqrt();

        assert_approx_eq(result, expected, Tolerance::STRICT);
    }

    proptest! {
        #[test]
        fn skewness_is_translation_invariant(
            (sample, offset) in exact_translation_case(3)
        ) {
            let translated: Vec<_> = sample.iter().map(|&x| x + offset).collect();

            let x = skewness(&sample);
            let y = skewness(&translated);

            prop_assert!(approx_eq(x, y, Tolerance::DEFAULT));
        }

        #[test]
        fn skewness_is_positive_scale_invariant(
            sample in finite_nonconstant_sample(3),
            scale in positive_power_of_two_scale(),
        ) {
            let scaled: Vec<_> = sample.iter().map(|&x| x * scale).collect();

            let x = skewness(&sample);
            let y = skewness(&scaled);

            prop_assert!(approx_eq(x, y, Tolerance::DEFAULT));
        }

        #[test]
        fn skewness_flips_sign_under_negative_scaling(
            sample in finite_nonconstant_sample(3),
            scale in positive_power_of_two_scale(),
        ) {
            let reflected: Vec<_> = sample.iter().map(|&x| x * -scale).collect();

            let x = skewness(&sample);
            let y = skewness(&reflected);

            prop_assert!(approx_eq(y, -x, Tolerance::DEFAULT));
        }

        #[test]
        fn skewness_is_reversal_invariant(sample in finite_nonconstant_sample(3)) {
            let mut reversed = sample.clone();
            reversed.reverse();

            let x = skewness(&sample);
            let y = skewness(&reversed);

            prop_assert!(approx_eq(x, y, Tolerance::DEFAULT));
        }

        #[test]
        fn skewness_is_finite_for_finite_nonconstant_samples(
            sample in finite_nonconstant_sample(3),
        ) {
            prop_assert!(skewness(&sample).is_finite());
        }
    }
}
