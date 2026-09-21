//! Input checks and interpolation helpers for quantile calculations.
//!
//! The validation predicates check sample finiteness and the probability
//! domain. Type 7 evaluation expects a nonempty, finite sample sorted in
//! ascending order and a finite probability in `[0, 1]`.
//!
//! Validation and sorting are the caller's responsibility. Interpolation
//! avoids forming a potentially overflowing difference when its endpoints
//! straddle zero.

use unrvl_numerics::convert::usize_to_f64;

/// Returns whether a sample is non-empty and contains only finite values.
#[inline]
pub(super) fn is_valid_sample(xs: &[f64]) -> bool {
    !xs.is_empty() && xs.iter().all(|x| x.is_finite())
}

/// Returns whether `p` is a finite probability in the closed interval `[0, 1]`.
#[inline]
pub(super) fn is_valid_probability(p: f64) -> bool {
    p.is_finite() && (0.0..=1.0).contains(&p)
}

/// Returns the Type 7 quantile of an already-sorted sample.
///
/// Type 7 is the default quantile definition used by R. For a sample of length
/// `n`, the zero-based interpolation position is:
///
/// ```text
/// h = (n - 1) · p
/// ```
///
/// If `h` is an integer, the observation at that index is returned. Otherwise,
/// the result is linearly interpolated between the observations immediately
/// below and above `h`.
///
/// A single-observation sample returns that observation for every probability.
///
/// # Preconditions
///
/// `sorted` must be non-empty, finite, and sorted in ascending order, and `p`
/// must be a finite probability in `[0, 1]`.
#[must_use]
pub(super) fn type7(sorted: &[f64], p: f64) -> f64 {
    debug_assert_ne!(sorted, []);
    debug_assert!(is_valid_probability(p));

    let n = sorted.len();

    // With one value, every quantile is that value.
    if n == 1 {
        return sorted[0];
    }

    // The 0-quantile is the minimum.
    if p == 0.0 {
        return sorted[0];
    }

    // The 1-quantile is the maximum.
    #[expect(clippy::float_cmp, reason = "exact boundary of the probability domain")]
    if p == 1.0 {
        return sorted[n - 1];
    }

    let h = p * usize_to_f64(n - 1);

    #[expect(clippy::cast_sign_loss, reason = "floor is a nonnegative")]
    #[expect(clippy::cast_possible_truncation, reason = "floor is in bounds")]
    let lower = h.floor() as usize;
    let higher = lower + 1;
    let fraction = h - usize_to_f64(lower);

    if fraction == 0.0 {
        return sorted[lower];
    }

    interpolate(sorted[lower], sorted[higher], fraction)
}

/// Linearly interpolates between two sorted, finite values.
///
/// The result is mathematically equivalent to:
///
/// ```text
/// a + fraction · (b - a)
/// ```
///
/// Near zero, a fused multiply-add avoids rounding weighted endpoints
/// separately before combining them. For larger endpoints that straddle
/// zero, a weighted representation avoids overflowing `b - a`.
///
/// # Preconditions
///
/// `a` and `b` must be finite with `a <= b`, and `fraction` must lie in the
/// closed interval `[0, 1]`.
#[inline]
fn interpolate(a: f64, b: f64, fraction: f64) -> f64 {
    debug_assert!(a.is_finite());
    debug_assert!(b.is_finite());
    debug_assert!(a <= b);
    debug_assert!((0.0..=1.0).contains(&fraction));

    // With sorted endpoints, negative `a` and positive `b` is the only case
    // where `b - a` becomes an addition of magnitudes and can therefore
    // overflow.
    let straddles_zero = a <= 0.0 && b >= 0.0;

    // Within ±MIN_POSITIVE, the endpoints lie on the uniform subnormal grid
    // (including the normal boundary), so `b - a` is safe and exactly
    // representable.
    let outside_tiny_range = a < -f64::MIN_POSITIVE || b > f64::MIN_POSITIVE;

    // Use the weighted form only when forming `b - a` may overflow.
    // Otherwise prefer the difference form, which needs just one rounded FMA.
    if straddles_zero && outside_tiny_range {
        a.mul_add(1.0 - fraction, fraction * b)
    } else {
        fraction.mul_add(b - a, a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod is_valid_sample {
        use super::*;

        #[test]
        fn accepts_finite_nonempty_sample() {
            assert!(is_valid_sample(&[-1.0, 0.0, 1.0]));
        }

        #[test]
        fn rejects_empty_sample() {
            assert!(!is_valid_sample(&[]));
        }

        #[test]
        fn rejects_nonfinite_values() {
            assert!(!is_valid_sample(&[1.0, f64::NAN]));
            assert!(!is_valid_sample(&[1.0, f64::INFINITY]));
            assert!(!is_valid_sample(&[1.0, f64::NEG_INFINITY]));
        }
    }

    mod is_valid_probability {
        use super::*;

        #[test]
        fn accepts_closed_unit_interval() {
            assert!(is_valid_probability(0.0));
            assert!(is_valid_probability(0.5));
            assert!(is_valid_probability(1.0));
        }

        #[test]
        fn rejects_values_outside_unit_interval() {
            assert!(!is_valid_probability(-f64::EPSILON));
            assert!(!is_valid_probability(1.0 + f64::EPSILON));
        }

        #[test]
        fn rejects_nonfinite_values() {
            assert!(!is_valid_probability(f64::NAN));
            assert!(!is_valid_probability(f64::INFINITY));
            assert!(!is_valid_probability(f64::NEG_INFINITY));
        }
    }

    mod type7 {
        use super::*;

        #[test]
        fn singleton_returns_its_value() {
            for p in [0.0, 0.25, 0.5, 0.75, 1.0] {
                assert_eq!(type7(&[42.0], p), 42.0);
            }
        }

        #[test]
        fn boundaries_return_minimum_and_maximum() {
            let xs = [-10.0, 0.0, 10.0, 20.0];

            assert_eq!(type7(&xs, 0.0), -10.0);
            assert_eq!(type7(&xs, 1.0), 20.0);
        }

        #[test]
        fn exact_position_returns_observation() {
            let xs = [0.0, 10.0, 20.0, 30.0, 40.0];

            // h = (5 - 1) * 0.25 = 1
            assert_eq!(type7(&xs, 0.25), 10.0);

            // h = (5 - 1) * 0.5 = 2
            assert_eq!(type7(&xs, 0.5), 20.0);
        }

        #[test]
        fn interpolates_between_observations() {
            let xs = [0.0, 10.0, 20.0, 30.0, 40.0];

            // h = 4 * 0.125 = 0.5
            assert_eq!(type7(&xs, 0.125), 5.0);

            // h = 4 * 0.375 = 1.5
            assert_eq!(type7(&xs, 0.375), 15.0);
        }

        #[test]
        fn interpolates_across_zero() {
            let xs = [-10.0, 10.0];

            assert_eq!(type7(&xs, 0.25), -5.0);
            assert_eq!(type7(&xs, 0.5), 0.0);
            assert_eq!(type7(&xs, 0.75), 5.0);
        }

        #[test]
        fn handles_extreme_opposite_endpoints() {
            let xs = [-f64::MAX, f64::MAX];

            let result = type7(&xs, 0.5);

            assert!(result.is_finite());
            assert_eq!(result, 0.0);
        }
    }

    mod interpolate {
        use super::*;

        #[test]
        fn returns_endpoints_at_boundaries() {
            assert_eq!(interpolate(10.0, 20.0, 0.0), 10.0);
            assert_eq!(interpolate(10.0, 20.0, 1.0), 20.0);
        }

        #[test]
        fn interpolates_same_sign_values() {
            assert_eq!(interpolate(10.0, 20.0, 0.25), 12.5);
            assert_eq!(interpolate(-20.0, -10.0, 0.25), -17.5);
        }

        #[test]
        fn interpolates_values_straddling_zero() {
            assert_eq!(interpolate(-20.0, 20.0, 0.25), -10.0);
            assert_eq!(interpolate(-20.0, 20.0, 0.5), 0.0);
        }

        #[test]
        fn avoids_overflow_when_endpoints_straddle_zero() {
            let result = interpolate(-f64::MAX, f64::MAX, 0.5);

            assert!(result.is_finite());
            assert_eq!(result, 0.0);
        }

        #[test]
        fn preserves_exact_subnormal_midpoints() {
            let d = f64::from_bits(1);

            assert_eq!(interpolate(-3.0 * d, d, 0.5), -d);
            assert_eq!(interpolate(-d, 3.0 * d, 0.5), d);
        }
    }
}
