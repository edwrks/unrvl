use unrvl_numerics::convert::usize_to_f64;
use unrvl_numerics::prelude::*;

/// Returns the arithmetic mean of a slice using Neumaier compensated summation.
///
/// The statistic is defined as:
///
/// ```text
///      ∑ xᵢ
/// x̄ = -----
///       n
/// ```
/// Returns `NaN` for an empty slice or when the input contains non-finite
/// values.
///
/// A finite constant sample returns its value exactly.
///
/// Compensated summation reduces rounding error but does not prevent an
/// intermediate sum from overflowing, in which case the result may be
/// non-finite.
///
/// # Panics
///
/// Panics if the slice length exceeds `2^53`.
#[must_use]
pub fn mean(xs: &[f64]) -> f64 {
    let Some((&first, rest)) = xs.split_first() else {
        return f64::NAN;
    };

    #[expect(clippy::float_cmp, reason = "detect exactly constant observations")]
    if first.is_finite() && rest.iter().all(|&x| x == first) {
        return first;
    }

    let n = usize_to_f64(xs.len());
    xs.iter().neumaier_sum() / n
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_utils::approx::{Tolerance, approx_eq, assert_approx_eq};
    use test_utils::strategies::{
        constant_sample, exact_translation_case, finite_sample, positive_power_of_two_scale,
    };
    use unrvl_numerics::extrema::min_max;

    use super::*;

    #[test]
    fn returns_nan_for_empty_input() {
        let xs = &[];
        let result = mean(xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_nan_input() {
        let xs = &[1.0, 2.0, f64::NAN];
        let result = mean(xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_positive_infinity() {
        let xs = &[1.0, 2.0, f64::INFINITY];
        let result = mean(xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_nan_for_negative_infinity() {
        let xs = &[1.0, 2.0, f64::NEG_INFINITY];
        let result = mean(xs);

        assert!(result.is_nan());
    }

    #[test]
    fn returns_value_for_singleton() {
        let xs = &[1.0];
        let result = mean(xs);

        assert_eq!(result, 1.0);
    }

    #[test]
    fn returns_mean_for_positive_values() {
        let xs = &[1.0, 2.0, 3.0, 4.0, 5.0];
        let result = mean(xs);

        assert_approx_eq(result, 3.0, Tolerance::VERY_STRICT);
    }

    #[test]
    fn returns_mean_for_negative_values() {
        let xs = &[-1.0, -2.0, -3.0];
        let result = mean(xs);

        assert_approx_eq(result, -2.0, Tolerance::VERY_STRICT);
    }

    #[test]
    fn returns_mean_for_mixed_sign_values() {
        let xs = &[-1.0, 2.0, -3.0];
        let result = mean(xs);

        assert_approx_eq(result, -2.0 / 3.0, Tolerance::VERY_STRICT);
    }

    #[test]
    fn returns_exact_mean_for_constant_max_values() {
        let xs = &[f64::MAX, f64::MAX];
        let result = mean(xs);

        assert_eq!(result, f64::MAX);
    }

    #[test]
    fn returns_non_finite_when_intermediate_sum_overflows() {
        let xs = &[f64::MAX, f64::MAX / 2.0];
        let result = mean(xs);

        assert!(!result.is_finite());
    }

    #[test]
    fn uses_compensated_summation() {
        let xs = &[1.0, 1e16, 1.0, -1e16];
        let result = mean(xs);

        assert_approx_eq(result, 0.5, Tolerance::VERY_STRICT);
    }

    proptest! {
        #[test]
        #[expect(clippy::float_cmp, reason = "exactly constant observations")]
        fn constant_sample_has_constant_mean(sample in constant_sample(1)) {
            let expected = sample[0];
            let result = mean(&sample);
            prop_assert_eq!(result, expected);
        }

        #[test]
        fn mean_is_within_sample_bounds(sample in finite_sample(1)) {
            let (min, max) = min_max(&sample);
            let result = mean(&sample);
            prop_assert!(min <= result && result <= max);
        }

        #[test]
        fn mean_is_translation_equivariant(
            (sample, offset) in exact_translation_case(2)
        ) {
            let translated: Vec<_> = sample.iter().map(|x| x + offset).collect();
            let x = mean(&sample) + offset;
            let y = mean(&translated);
            prop_assert!(approx_eq(x, y, Tolerance::DEFAULT));

        }

        #[test]
        fn mean_is_scale_equivariant(
            sample in finite_sample(1),
            scale in positive_power_of_two_scale()
        ) {
            let scaled: Vec<_> = sample.iter().map(|x| x * scale).collect();
            let x = mean(&sample) * scale;
            let y = mean(&scaled);
            prop_assert!(approx_eq(x, y, Tolerance::DEFAULT));

        }

        #[test]
        fn mean_is_reversal_invariant(sample in finite_sample(1)) {
            let mut reordered = sample.clone();
            reordered.reverse();

            let x = mean(&sample);
            let y = mean(&reordered);

            prop_assert!(approx_eq(x, y, Tolerance::DEFAULT));
        }

        #[test]
        fn mean_is_replication_invariant(sample in finite_sample(1)) {
            let mut replicated = sample.clone();
            replicated.extend_from_slice(&sample);

            let x = mean(&sample);
            let y = mean(&replicated);

            prop_assert!(approx_eq(x, y, Tolerance::DEFAULT));
        }
    }
}
