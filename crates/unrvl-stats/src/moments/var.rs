use unrvl_numerics::convert::usize_to_f64;
use unrvl_numerics::summation::NeumaierSumExt;

use crate::internal::deviations::centered;

/// Returns the sample variance of a slice using Bessel's correction.
///
/// The statistic is defined as:
///
/// ```text
///         ∑ (xᵢ - x̄)²
/// s² = -----------------
///           n - 1
/// ```
///
/// where `x̄` is the sample mean.
///
/// The variance is computed with `n - 1` degrees of freedom. Squared
/// deviations are accumulated with Neumaier compensated summation. Centring
/// retains a correction for mean rounding to better preserve small variations
/// around a large or nearly constant level.
///
/// Returns `NaN` if fewer than two observations are provided or when the input
/// contains non-finite values.
///
/// Squared deviations or the final variance may overflow the representable
/// `f64` range. Very small nonzero variances may underflow to zero.
///
/// # Panics
///
/// Panics if the slice length exceeds `2^53`.
#[must_use]
pub fn var(xs: &[f64]) -> f64 {
    let n = xs.len();

    if n < 2 {
        return f64::NAN;
    }

    let sum_d2 = centered(xs).map(|d| d * d).neumaier_sum();

    sum_d2 / usize_to_f64(n - 1)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_utils::approx::{Tolerance, approx_eq, assert_approx_eq};
    use test_utils::strategies::{
        constant_samples, finite_samples, finite_scale, power_of_two_scale,
    };

    use super::*;

    #[test]
    fn returns_nan_for_empty_input() {
        let xs = &[];
        let result = var(xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_too_few_inputs() {
        let xs = &[1.0];
        let result = var(xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_nan_input() {
        let xs = &[1.0, 2.0, f64::NAN];
        let result = var(xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_positive_infinity() {
        let xs = &[1.0, 2.0, f64::INFINITY];
        let result = var(xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_negative_infinity() {
        let xs = &[1.0, 2.0, f64::NEG_INFINITY];
        let result = var(xs);

        assert!(result.is_nan());
    }

    #[test]
    fn calculates_variance() {
        let xs = &[2.0, 5.0, 9.0, 12.0];
        let result = var(xs);

        assert_approx_eq(result, 58.0 / 3.0, Tolerance::DEFAULT);
    }

    #[test]
    fn preserves_variation_around_large_offset() {
        let xs = &[1e16, 1e16, 1e16 + 2.0];
        let result = var(xs);

        assert_approx_eq(result, 4.0 / 3.0, Tolerance::DEFAULT);
    }

    proptest! {
        #[test]
        #[expect(clippy::float_cmp, reason = "exactly constant observations")]
        fn constant_sample_has_zero_variance(sample in constant_samples(2)) {
            let expected = 0.0;
            let result = var(&sample);
            prop_assert_eq!(result, expected);
        }

        #[test]
        fn variance_is_non_negative(sample in finite_samples(2)) {
            let result = var(&sample);

            if result.is_finite() {
                prop_assert!(result >= 0.0);
            }
        }

        #[test]
        fn variance_is_translation_invariant(
            sample in finite_samples(2),
            offset in finite_scale(),
        ) {
            let translated: Vec<_> = sample.iter().map(|&x| x + offset).collect();

            let x = var(&sample);
            let y = var(&translated);

            prop_assert!(approx_eq(x, y, Tolerance::DEFAULT));
        }

        #[test]
        fn variance_is_scale_equivariant(
            sample in finite_samples(2),
            scale in power_of_two_scale(),
        ) {
            let scaled: Vec<_> = sample.iter().map(|&x| x * scale).collect();

            let x = var(&sample) * scale * scale;
            let y = var(&scaled);

            prop_assert!(approx_eq(x, y, Tolerance::DEFAULT));
        }

        #[test]
        fn variance_is_reversal_invariant(sample in finite_samples(2)) {
            let mut reversed = sample.clone();
            reversed.reverse();

            let x = var(&sample);
            let y = var(&reversed);

            prop_assert!(approx_eq(x, y, Tolerance::DEFAULT));
        }

    }
}
