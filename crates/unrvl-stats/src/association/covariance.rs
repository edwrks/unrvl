use unrvl_numerics::convert::usize_to_f64;
use unrvl_numerics::summation::NeumaierSum;

use crate::internal::deviations::centered_scaled;

/// Returns the sample covariance of paired observations.
///
/// The statistic is defined as:
///
/// ```text
///              ∑ (xᵢ - x̄)(yᵢ - ȳ)
/// cov(x, y) = ---------------------
///                     n - 1
/// ```
///
/// Observations are paired by position. Returns `NaN` if fewer than two pairs
/// are provided or any observation is non-finite. If both samples are finite
/// and either is constant, returns zero. No pairs are discarded.
///
/// Uses separate power-of-two scales, corrected centering, and compensation
/// for product and summation rounding. These measures do not eliminate rounding
/// or underflow; small contributions relative to the input scales can be lost.
/// The result can overflow to signed infinity or underflow to zero.
///
/// Uses `O(n)` time and `O(1)` additional memory, where `n` is the number of
/// paired observations.
///
/// # Panics
///
/// Panics if the slices have different lengths, or if a finite sample contains
/// more than `2^53` observations.
///
/// # Examples
///
/// ```
/// # use unrvl_stats::association::covariance;
/// let xs = [1.0, 2.0, 3.0];
/// let ys = [2.0, 4.0, 6.0];
/// assert_eq!(covariance(&xs, &ys), 2.0);
/// ```
#[must_use]
pub fn covariance(xs: &[f64], ys: &[f64]) -> f64 {
    assert_eq!(xs.len(), ys.len(), "inputs must have equal lengths");

    let n = xs.len();
    if n < 2 {
        return f64::NAN;
    }

    let Some((x_unit, x_deviations)) = centered_scaled(xs) else {
        return f64::NAN;
    };

    let Some((y_unit, y_deviations)) = centered_scaled(ys) else {
        return f64::NAN;
    };

    let mut sum = NeumaierSum::new();
    for (x, y) in x_deviations.zip(y_deviations) {
        let product = x * y;
        sum.add(product);
        sum.add(x.mul_add(y, -product));
    }

    let scaled_covariance = sum.total() / usize_to_f64(n - 1);
    restore_units(scaled_covariance, x_unit, y_unit)
}

/// Restores covariance units from two positive, finite power-of-two scales.
/// Shared by scalar and matrix covariance to preserve extreme-scale handling.
pub(super) fn restore_units(scaled_covariance: f64, x_unit: f64, y_unit: f64) -> f64 {
    let scale = x_unit * y_unit;

    // A representable product of power-of-two units is exact, even when
    // subnormal. Combining the units first also handles opposing scales.
    if scale.is_finite() && scale != 0.0 {
        scaled_covariance * scale
    } else if scale.is_infinite() {
        // The units overflow together, but a small covariance may still fit.
        (scaled_covariance * x_unit.min(y_unit)) * x_unit.max(y_unit)
    } else {
        // The product of the units underflowed to zero, but the covariance may
        // still round to a nonzero subnormal. Apply the larger unit first.
        (scaled_covariance * x_unit.max(y_unit)) * x_unit.min(y_unit)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_utils::approx::{Tolerance, approx_eq, assert_approx_eq};
    use test_utils::strategies::{
        MAX_SAMPLE_LEN, finite_paired_samples, finite_sample, positive_power_of_two_scale,
    };

    use super::*;
    use crate::moments::var;

    #[test]
    #[should_panic(expected = "inputs must have equal lengths")]
    fn panics_on_length_mismatch() {
        let _ = covariance(&[1.0, 2.0], &[3.0, 4.0, 5.0]);
    }

    #[test]
    #[should_panic(expected = "inputs must have equal lengths")]
    fn checks_lengths_before_minimum_sample_size() {
        let _ = covariance(&[], &[1.0]);
    }

    #[test]
    fn returns_nan_for_fewer_than_two_pairs() {
        assert!(covariance(&[], &[]).is_nan());
        assert!(covariance(&[1.0], &[2.0]).is_nan());
    }

    #[test]
    fn rejects_nonfinite_values_in_either_sample_even_if_other_is_constant() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for bad in [[invalid, 1.0], [1.0, invalid]] {
                for good in [[2.0, 3.0], [2.0, 2.0]] {
                    assert!(covariance(&bad, &good).is_nan());
                    assert!(covariance(&good, &bad).is_nan());
                }
            }
        }
    }

    #[test]
    fn calculates_sample_covariance_with_either_sign() {
        let xs = &[1.0, 3.0];
        assert_eq!(covariance(xs, &[2.0, 6.0]), 4.0);
        assert_eq!(covariance(xs, &[6.0, 2.0]), -4.0);
    }

    #[test]
    fn calculates_zero_sample_covariance() {
        let xs = [-1.0, -1.0, 1.0, 1.0];
        let ys = [-1.0, 1.0, -1.0, 1.0];

        assert_eq!(covariance(&xs, &ys), 0.0);
    }

    #[test]
    fn finite_constant_samples_have_zero_covariance() {
        let varying = [-f64::MAX, 0.0, f64::MAX];
        let smallest_positive = f64::from_bits(1);

        for value in [0.0, -0.0, smallest_positive, -f64::MAX, f64::MAX] {
            let constant = [value; 3];

            assert_eq!(covariance(&constant, &varying), 0.0);
            assert_eq!(covariance(&varying, &constant), 0.0);
            assert_eq!(covariance(&constant, &constant), 0.0);
        }
    }

    #[test]
    fn preserves_variation_around_large_offsets() {
        let xs = [1e16, 1e16, 1e16 + 2.0];
        let ys = [1.0, 1.0, 4.0];
        let expected = 2.0;

        assert_approx_eq(covariance(&xs, &ys), expected, Tolerance::VERY_STRICT);
    }

    #[test]
    fn compensates_product_rounding_before_cancellation() {
        let delta = 2.0_f64.powi(-27);
        let xs = [1.0 + delta, -1.0, -(1.0 + delta), 1.0];
        let ys = [1.0 - delta, 1.0, -(1.0 - delta), -1.0];

        let result = covariance(&xs, &ys);

        // Both means are zero. The cross-product sum is -2 * delta², whereas
        // rounding each product before summing would give zero.
        let expected = -2.0_f64.powi(-53) / 3.0;
        assert_approx_eq(result, expected, Tolerance::new(0.0, 1e-15));
    }

    #[test]
    fn avoids_overflow_when_unscaled_cross_product_sum_is_too_large() {
        // Both means are zero, so the paired deviation products are:
        // [-1e308, 0, -1e308].
        //
        // Their sum, approximately -2e308, is too large in magnitude for f64.
        // Dividing by n - 1 = 2 gives -1e308, which fits.
        // We should get this finite covariance, not negative infinity.
        let xs = [-1e154, 0.0, 1e154];
        let ys = [1e154, 0.0, -1e154];

        let result = covariance(&xs, &ys);
        let expected = -1e308;

        assert_approx_eq(result, expected, Tolerance::STRICT);
    }

    #[test]
    fn preserves_small_variation_around_huge_offsets() {
        // Start with a simple pattern:
        // covariance([0, 0, h], [0, 0, k]) = h * k / 3.
        // Adding the same base to every value does not change covariance.
        //
        // These powers of two make both the base and the changes exact in f64.
        // Each change is tiny relative to the base, but still a huge number.
        let base = 2.0_f64.powi(520);
        let x_step = 2.0_f64.powi(468);
        let y_step = 2.0_f64.powi(469);
        let xs = [base, base, base + x_step];
        let ys = [base, base, base + y_step];

        let result = covariance(&xs, &ys);

        // The answer is 2^468 * 2^469 / 3 = 2^937 / 3, which fits.
        // Internally, each input uses a scaling unit of 2^520. Multiplying
        // those units gives 2^1040, which overflows. Restoring the units in
        // stages must still recover the finite answer.
        let expected = 2.0_f64.powi(937) / 3.0;
        assert_approx_eq(result, expected, Tolerance::STRICT);
    }

    #[test]
    fn preserves_the_answer_when_one_sample_is_huge_and_the_other_is_tiny() {
        // Covariance([a, a, a + h], [b, b, b + k]) = h * k / 3.
        // One sample is huge and the other is tiny. Their starting values
        // do not affect the answer, only their changes matter.
        let x_base = 2.0_f64.powi(1000);
        let x_step = 2.0_f64.powi(948);
        let y_base = 2.0_f64.powi(-1000);

        // MIN_POSITIVE is 2^-1022, so dividing by 2^30 gives 2^-1052.
        let y_step = f64::MIN_POSITIVE / 2.0_f64.powi(30);
        let xs = [x_base, x_base, x_base + x_step];
        let ys = [y_base, y_base, y_base + y_step];

        // The answer is 2^948 * 2^-1052 / 3 = 2^-104 / 3.
        // The scaling units cancel: 2^1000 * 2^-1000 = 1. Applying the tiny
        // unit first instead could round an intermediate result to zero.
        let expected = 2.0_f64.powi(-104) / 3.0;
        let tolerance = Tolerance::new(0.0, 1e-14);

        // Swapping the samples must preserve the answer too.
        assert_approx_eq(covariance(&xs, &ys), expected, tolerance);
        assert_approx_eq(covariance(&ys, &xs), expected, tolerance);
    }

    #[test]
    fn preserves_subnormal_result_when_scale_product_underflows() {
        // With two observations and zero means, covariance is 2*x*y:
        // the product sum is 2*x*y, and n - 1 = 1.
        let x = 2.0_f64.powi(-537);
        let y = 2.0_f64.powi(-538);

        // Multiplying x*y directly gives 2^-1075, which rounds to zero.
        // But the final answer is twice that: 2^-1074, which fits exactly.
        let smallest_positive = f64::from_bits(1);
        assert_eq!(covariance(&[-x, x], &[-y, y]), smallest_positive);
    }

    #[test]
    fn returns_signed_infinity_when_the_final_answer_is_too_large() {
        // For [-m, 0, m], sample variance is m². With m = f64::MAX,
        // the final answer itself is too large for f64; infinity is correct.
        // Reversing the second sample makes the answer negative.
        let xs = [-f64::MAX, 0.0, f64::MAX];
        let ys = [f64::MAX, 0.0, -f64::MAX];
        assert_eq!(covariance(&xs, &xs), f64::INFINITY);
        assert_eq!(covariance(&xs, &ys), f64::NEG_INFINITY);
    }

    #[test]
    fn returns_zero_when_the_final_answer_is_too_small() {
        // Call the smallest positive f64 t. The exact covariance is 2*t².
        // That is much smaller than t, so rounding to zero is correct.
        let smallest_positive = f64::from_bits(1);
        let xs = [-smallest_positive, smallest_positive];
        assert_eq!(covariance(&xs, &xs), 0.0);
    }

    proptest! {
        #[test]
        fn is_symmetric((xs, ys) in finite_paired_samples(2)) {
            let a = covariance(&xs, &ys).to_bits();
            let b = covariance(&ys, &xs).to_bits();

            prop_assert_eq!(a, b);
        }

        #[test]
        fn agrees_with_variance_for_identical_samples(xs in finite_sample(2)) {
            let c = covariance(&xs, &xs);
            let v = var(&xs);

            prop_assert!(approx_eq(c, v, Tolerance::STRICT));
        }

        #[test]
        fn preserves_covariance_under_joint_reversal((mut xs, mut ys) in finite_paired_samples(2)) {
            let expected = covariance(&xs, &ys);

            xs.reverse();
            ys.reverse();

            let reversed = covariance(&xs, &ys);

            prop_assert!(approx_eq(reversed, expected, Tolerance::STRICT));
        }

        #[test]
        fn scales_with_both_variables(
            (xs, ys) in finite_paired_samples(2),
            x_scale in positive_power_of_two_scale(),
            y_scale in positive_power_of_two_scale(),
        ) {
            let scaled_x: Vec<_> = xs.iter().map(|x| x * x_scale).collect();
            let scaled_y: Vec<_> = ys.iter().map(|y| -y * y_scale).collect();

            let scaled_result = covariance(&scaled_x, &scaled_y);
            let expected = -(covariance(&xs, &ys) * x_scale) * y_scale;

            prop_assert!(approx_eq(scaled_result, expected, Tolerance::STRICT));
        }

        #[test]
        fn is_invariant_to_exact_translations(
            pairs in prop::collection::vec((-10_000_i32..10_000, -10_000_i32..10_000), 2..=MAX_SAMPLE_LEN),
            x_offset in -100_i32..100,
            y_offset in -100_i32..100,
        ) {
            let (xs, ys): (Vec<_>, Vec<_>) = pairs.iter().map(|&(x, y)| (f64::from(x), f64::from(y))).unzip();
            let translated_x: Vec<_> = xs.iter().map(|x| x + f64::from(x_offset)).collect();
            let translated_y: Vec<_> = ys.iter().map(|y| y + f64::from(y_offset)).collect();

            let a = covariance(&xs, &ys);
            let b = covariance(&translated_x, &translated_y);

            prop_assert!(approx_eq(a, b, Tolerance::DEFAULT));
        }
    }
}
