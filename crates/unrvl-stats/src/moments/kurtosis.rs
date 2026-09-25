use unrvl_numerics::convert::usize_to_f64;
use unrvl_numerics::summation::NeumaierSum;

use crate::internal::deviations::centered_scaled;

/// Returns the bias-corrected sample excess kurtosis G2 (Joanes & Gill 1998
/// type 2).
///
/// Excess kurtosis uses the normal distribution's kurtosis as its zero
/// reference. Positive values indicate greater tail extremity relative to
/// normal, while negative values indicate less.
///
/// The statistic is defined as:
///
/// ```text
///              n(n + 1)                        3(n - 1)²
/// G2 = ------------------------- · ∑ zᵢ⁴ - ------------------
///        (n - 1)(n - 2)(n - 3)               (n - 2)(n - 3)
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
/// Scaling does not eliminate floating-point rounding. When observations
/// span an extreme range of magnitudes, very small scaled values can underflow
/// to zero.
///
/// Returns `NaN` for samples shorter than four observations, constant
/// samples, or input containing non-finite values.
///
/// # Panics
///
/// Panics if a finite sample contains more than `2^53` observations.
#[must_use]
pub fn excess_kurtosis(xs: &[f64]) -> f64 {
    if xs.len() < 4 {
        return f64::NAN;
    }

    let Some((_, deviations)) = centered_scaled(xs) else {
        return f64::NAN;
    };

    let mut sum_u2 = NeumaierSum::new();
    let mut sum_u4 = NeumaierSum::new();

    for u in deviations {
        let u2 = u * u;

        sum_u2 += u2;
        sum_u4 += u2 * u2;
    }

    let sum_u2 = sum_u2.total();
    let sum_u4 = sum_u4.total();

    if sum_u2 == 0.0 {
        return f64::NAN;
    }

    let n = usize_to_f64(xs.len());
    let n_minus_1 = n - 1.0;
    let n_minus_2 = n - 2.0;
    let n_minus_3 = n - 3.0;

    let scaled_var = sum_u2 / n_minus_1;
    let sum_z4 = sum_u4 / scaled_var.powi(2);

    let bias_factor = n * (n + 1.0) / (n_minus_1 * n_minus_2 * n_minus_3);
    let excess_correction = 3.0 * n_minus_1.powi(2) / (n_minus_2 * n_minus_3);

    bias_factor.mul_add(sum_z4, -excess_correction)
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
        let result = excess_kurtosis(xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_singleton() {
        let xs = &[1.0];
        let result = excess_kurtosis(xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_two_observations() {
        let xs = &[1.0, 2.0];
        let result = excess_kurtosis(xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_three_observations() {
        let xs = &[1.0, 2.0, 3.0];
        let result = excess_kurtosis(xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_nan_input() {
        let xs = &[1.0, 2.0, 3.0, f64::NAN];
        let result = excess_kurtosis(xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_positive_infinity() {
        let xs = &[1.0, 2.0, 3.0, f64::INFINITY];
        let result = excess_kurtosis(xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_negative_infinity() {
        let xs = &[1.0, 2.0, 3.0, f64::NEG_INFINITY];
        let result = excess_kurtosis(xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_zero_variance() {
        let xs = &[1.0, 1.0, 1.0, 1.0];
        let result = excess_kurtosis(xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_constant_extreme_samples() {
        let values = [f64::from_bits(1), -f64::from_bits(1), f64::MAX, -f64::MAX];
        for value in values {
            assert!(excess_kurtosis(&[value; 4]).is_nan());
        }
    }

    #[test]
    fn returns_nan_for_translation_that_rounds_to_a_constant() {
        let e = f64::EPSILON;
        let xs = [1.0, 1.0 + e, 2.0f64.mul_add(e, 1.0), 3.0f64.mul_add(e, 1.0)];
        let translated = xs.map(|x| x + 100.0);

        assert_approx_eq(excess_kurtosis(&xs), -1.2, Tolerance::STRICT);

        // The translation erased the distinctions between observations.
        assert_eq!(translated, [101.0; 4]);
        assert!(excess_kurtosis(&translated).is_nan());
    }

    #[test]
    fn matches_known_symmetric_sample() {
        let xs = &[1.0, 2.0, 3.0, 4.0, 5.0];
        let result = excess_kurtosis(xs);
        let expected = -1.2;

        assert_approx_eq(result, expected, Tolerance::VERY_STRICT);
    }

    #[test]
    fn matches_known_high_kurtosis_sample() {
        let xs = &[1.0, 1.0, 1.0, 2.0];
        let result = excess_kurtosis(xs);
        let expected = 4.0;

        assert_approx_eq(result, expected, Tolerance::VERY_STRICT);
    }

    #[test]
    fn matches_known_low_kurtosis_sample() {
        let xs = &[1.0, 1.0, 2.0, 2.0];
        let result = excess_kurtosis(xs);
        let expected = -6.0;

        assert_approx_eq(result, expected, Tolerance::VERY_STRICT);
    }

    #[test]
    fn remains_finite_at_large_scale() {
        let xs = [1.0, 1.0, 2.0, 4.0, 10.0];
        let scaled = xs.map(|x| x * 1e110);

        let result = excess_kurtosis(&xs);
        let result_scaled = excess_kurtosis(&scaled);

        assert!(result_scaled.is_finite());
        assert_approx_eq(result_scaled, result, Tolerance::DEFAULT);
    }

    #[test]
    fn remains_finite_at_small_scale() {
        let xs = [1.0, 1.0, 2.0, 4.0, 10.0];
        let scaled = xs.map(|x| x * 1e-110);

        let result = excess_kurtosis(&xs);
        let result_scaled = excess_kurtosis(&scaled);

        assert!(result_scaled.is_finite());
        assert_approx_eq(result_scaled, result, Tolerance::DEFAULT);
    }

    #[test]
    fn preserves_shape_at_the_smallest_positive_subnormal_scale() {
        let d = f64::from_bits(1);
        let xs = [0.0, 0.0, 0.0, d];
        let result = excess_kurtosis(&xs);
        let expected = 4.0;

        assert_approx_eq(result, expected, Tolerance::STRICT);
    }

    #[test]
    fn preserves_shape_at_the_smallest_negative_subnormal_scale() {
        let d = f64::from_bits(1);
        let xs = [0.0, 0.0, 0.0, -d];
        let result = excess_kurtosis(&xs);
        let expected = 4.0;

        assert_approx_eq(result, expected, Tolerance::STRICT);
    }

    #[test]
    fn remains_finite_when_the_unscaled_sum_would_overflow() {
        let half_max = f64::MAX / 2.0;
        let xs = [half_max, half_max, half_max, f64::MAX];
        let result = excess_kurtosis(&xs);
        let expected = 4.0;

        assert_approx_eq(result, expected, Tolerance::STRICT);
    }

    #[test]
    fn preserves_shape_around_a_large_offset() {
        let xs = [1e16, 1e16, 1e16, 1e16 + 2.0];
        let result = excess_kurtosis(&xs);
        let expected = 4.0;

        assert_approx_eq(result, expected, Tolerance::STRICT);
    }

    proptest! {
        #[test]
        fn excess_kurtosis_is_translation_invariant(
            (sample, offset) in exact_translation_case(4)
        ) {
            let translated: Vec<_> = sample.iter().map(|&x| x + offset).collect();

            let x = excess_kurtosis(&sample);
            let y = excess_kurtosis(&translated);

            prop_assert!(approx_eq(x, y, Tolerance::DEFAULT));
        }

        #[test]
        fn excess_kurtosis_is_positive_scale_invariant(
            sample in finite_nonconstant_sample(4),
            scale in positive_power_of_two_scale(),
        ) {
            let scaled: Vec<_> = sample.iter().map(|&x| x * scale).collect();

            let x = excess_kurtosis(&sample);
            let y = excess_kurtosis(&scaled);

            prop_assert!(approx_eq(x, y, Tolerance::DEFAULT));
        }

        #[test]
        fn excess_kurtosis_is_negative_scale_invariant(
            sample in finite_nonconstant_sample(4),
            scale in positive_power_of_two_scale(),
        ) {
            let reflected: Vec<_> = sample.iter().map(|&x| x * -scale).collect();

            let x = excess_kurtosis(&sample);
            let y = excess_kurtosis(&reflected);

            prop_assert!(approx_eq(x, y, Tolerance::DEFAULT));
        }

        #[test]
        fn excess_kurtosis_is_reversal_invariant(
            sample in finite_nonconstant_sample(4),
        ) {
            let mut reversed = sample.clone();
            reversed.reverse();

            let x = excess_kurtosis(&sample);
            let y = excess_kurtosis(&reversed);

            prop_assert!(approx_eq(x, y, Tolerance::DEFAULT));
        }

        #[test]
        fn excess_kurtosis_is_finite_for_finite_nonconstant_samples(
            sample in finite_nonconstant_sample(4),
        ) {
            prop_assert!(excess_kurtosis(&sample).is_finite());
        }
    }
}
