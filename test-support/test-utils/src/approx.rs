//! Approximate equality assertions for finite floating-point values.
//!
//! Provides [`assert_approx_eq`] for individual values and
//! [`assert_all_approx_eq`] for element-wise slice comparisons.
//! Slice assertion failures identify the first failing element by its
//! zero-based index.
//!
//! Comparisons accept either an absolute or a relative tolerance, as
//! described by [`Tolerance`]. Use [`Tolerance::DEFAULT`] or
//! [`Tolerance::STRICT`] for predefined limits, or [`Tolerance::new`]
//! to specify your own.
//!
//! All assertions reject NaN and infinite values, even when both inputs
//! are identical.

/// Absolute and relative tolerances for approximate floating-point comparisons.
///
/// Values compare equal if either their absolute difference is at most `abs`
/// or their relative difference is at most `rel`.
///
/// The relative difference is computed as `|x / scale - y / scale|`, where
/// `scale = max(|x|, |y|)`. Exactly equal values always compare equal.
///
/// Both tolerances are finite and non-negative.
#[derive(Debug, Clone, Copy)]
pub struct Tolerance {
    abs: f64,
    rel: f64,
}

impl Tolerance {
    /// A tolerance of `1e-12` for both absolute and relative differences.
    pub const DEFAULT: Self = Self::new(1e-12, 1e-12);

    /// A tighter tolerance of `1e-14` for both absolute and relative
    /// differences.
    pub const STRICT: Self = Self::new(1e-14, 1e-14);

    /// A very strict tolerance of `1e-15` for both absolute and relative
    /// differences.
    pub const VERY_STRICT: Self = Self::new(1e-15, 1e-15);

    /// Creates a tolerance with the given absolute and relative limits.
    ///
    /// A relative tolerance is a fraction, so `0.01` represents 1%.
    ///
    /// # Panics
    ///
    /// Panics if either tolerance is negative, infinite, or NaN.
    #[must_use]
    pub const fn new(abs: f64, rel: f64) -> Self {
        assert!(abs.is_finite(), "absolute tolerance must be finite");
        assert!(abs >= 0.0, "absolute tolerance must be non-negative");

        assert!(rel.is_finite(), "relative tolerance must be finite");
        assert!(rel >= 0.0, "relative tolerance must be non-negative");

        Self { abs, rel }
    }

    /// Returns the absolute tolerance.
    #[must_use]
    pub const fn abs(&self) -> f64 {
        self.abs
    }

    /// Returns the relative tolerance.
    #[must_use]
    pub const fn rel(&self) -> f64 {
        self.rel
    }
}

/// Asserts that actual value `x` approximately equals expected value `y`.
///
/// The comparison succeeds if either the absolute or relative tolerance
/// is satisfied. See [`Tolerance`] for the comparison rule.
///
/// # Panics
///
/// Panics if either value is infinite or NaN, or if the values differ
/// by more than both tolerances.
#[track_caller]
pub fn assert_approx_eq(x: f64, y: f64, tolerance: Tolerance) {
    assert!(x.is_finite(), "actual is non-finite: {x}");
    assert!(y.is_finite(), "expected is non-finite: {y}");

    assert!(
        approx_eq(x, y, tolerance),
        "assertion `left ≈ right` failed: actual={x:.17e}, expected={y:.17e}, tolerance={tolerance:?}"
    );
}

/// Asserts that each actual value in `xs` approximately equals the
/// corresponding expected value in `ys`.
///
/// The comparison succeeds if either the absolute or relative tolerance
/// is satisfied. See [`Tolerance`] for the comparison rule.
///
/// # Panics
///
/// Panics if the slices have different lengths, any value is infinite
/// or NaN, or any pair differs by more than both tolerances.
///
/// Element failures report the zero-based index of the first failing pair.
#[track_caller]
pub fn assert_all_approx_eq(xs: &[f64], ys: &[f64], tolerance: Tolerance) {
    let n = xs.len();
    let m = ys.len();
    assert_eq!(n, m, "length mismatch: got {n}, want {m}");

    for (i, (&x, &y)) in xs.iter().zip(ys).enumerate() {
        assert!(x.is_finite(), "index {i}: actual is non-finite: {x}");
        assert!(y.is_finite(), "index {i}: expected is non-finite: {y}");

        assert!(
            approx_eq(x, y, tolerance),
            "index {i}: assertion `left ≈ right` failed: actual={x:.17e}, expected={y:.17e}, tolerance={tolerance:?}"
        );
    }
}

/// Returns whether `x` approximately equals expected value `y`.
///
/// The comparison succeeds if either the absolute or relative tolerance
/// is satisfied. See [`Tolerance`] for the comparison rule.
///
/// # Panics
///
/// Panics if either value is infinite or NaN, or if the values differ
/// by more than both tolerances.
#[must_use]
pub fn approx_eq(x: f64, y: f64, tolerance: Tolerance) -> bool {
    if !x.is_finite() || !y.is_finite() {
        return false;
    }

    if x == y {
        return true;
    }

    let scale = x.abs().max(y.abs());
    let abs_delta = (x - y).abs();
    let rel_delta = if x.is_sign_negative() == y.is_sign_negative() {
        abs_delta / scale
    } else {
        x.abs() / scale + y.abs() / scale
    };

    abs_delta <= tolerance.abs || rel_delta <= tolerance.rel
}

#[cfg(test)]
mod tests {
    use super::*;

    mod assert_approx_eq {
        use super::*;

        #[test]
        fn passes_for_equal_values_with_zero_tolerances() {
            let tol = Tolerance::new(0.0, 0.0);
            assert_approx_eq(1.0, 1.0, tol);
        }

        #[test]
        fn uses_absolute_tolerance_near_zero() {
            let tol = Tolerance::new(1e-12, 0.0);
            assert_approx_eq(5e-14, 0.0, tol);
        }

        #[test]
        fn uses_relative_tolerance_at_large_magnitudes() {
            let tol = Tolerance::DEFAULT;
            assert_approx_eq(1_000_000_000.000_01, 1_000_000_000.0, tol);
        }

        #[test]
        fn comparison_is_symmetric() {
            let tol = Tolerance::new(0.0, 1e-12);
            let x = 1_000_000_000.000_01;
            let y = 1_000_000_000.0;
            assert_approx_eq(x, y, tol);
            assert_approx_eq(y, x, tol);
        }

        #[test]
        fn avoids_overflow_when_comparing_opposite_finite_extremes() {
            let tol = Tolerance::new(0.0, 2.0);
            assert_approx_eq(f64::MAX, -f64::MAX, tol);
        }

        #[test]
        #[should_panic(expected = "assertion `left ≈ right` failed")]
        fn rejects_opposite_finite_extremes_outside_relative_tolerance() {
            let tol = Tolerance::new(0.0, 1.5);
            assert_approx_eq(f64::MAX, -f64::MAX, tol);
        }

        #[test]
        #[should_panic(expected = "assertion `left ≈ right` failed")]
        fn panics_outside_both_tolerances() {
            let tol = Tolerance::new(0.01, 0.001);
            assert_approx_eq(1.0, 1.02, tol);
        }

        #[test]
        #[should_panic(expected = "is non-finite")]
        fn panics_for_non_finite_value() {
            let tol = Tolerance::DEFAULT;
            assert_approx_eq(1.0, f64::NAN, tol);
        }
    }

    mod assert_all_approx_eq {
        use super::*;

        #[test]
        fn applies_absolute_and_relative_tolerances_per_element() {
            let tol = Tolerance::DEFAULT;
            let xs = [5e-14, 1_000_000_000.000_01];
            let ys = [0.0, 1_000_000_000.0];

            assert_all_approx_eq(&xs, &ys, tol);
        }

        #[test]
        fn passes_for_empty_slices() {
            let tol = Tolerance::DEFAULT;
            assert_all_approx_eq(&[], &[], tol);
        }

        #[test]
        #[should_panic(expected = "length mismatch: got 2, want 1")]
        fn panics_on_length_mismatch() {
            let tol = Tolerance::DEFAULT;
            assert_all_approx_eq(&[1.0, 2.0], &[1.0], tol);
        }

        #[test]
        #[should_panic(expected = "index 1: assertion `left ≈ right` failed")]
        fn panics_on_value_mismatch() {
            let tol = Tolerance::new(0.01, 0.01);
            assert_all_approx_eq(&[1.0, 2.0], &[1.0, 2.5], tol);
        }

        #[test]
        #[should_panic(expected = "index 1: expected is non-finite")]
        fn panics_for_non_finite_value() {
            let tol = Tolerance::new(0.01, 0.01);
            assert_all_approx_eq(&[1.0, 2.0], &[1.0, f64::INFINITY], tol);
        }
    }
}
