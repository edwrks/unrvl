//! Compensated dot-product algorithms for improved floating-point accuracy.

use crate::summation::NeumaierSum;

/// Computes `sum(xs[i] * ys[i])` using the Ogita-Rump-Oishi algorithm.
///
/// This function is intended for finite operands. It does not provide
/// extended-real arithmetic: `NaN`, infinities, or intermediate product
/// overflow can produce a non-finite result.
///
/// # Panics
///
/// Panics if `xs` and `ys` have different lengths.
///
/// # Example
///
/// ```
/// # use unrvl_numerics::dot::compensated_dot;
/// let p53 = (1_u64 << 53) as f64;
//
/// let xs = [p53, 1.0, -p53];
/// let ys = [1.0, 1.0, 1.0];
///
/// assert_eq!(compensated_dot(&xs, &ys), 1.0);
/// ```
#[must_use]
pub fn compensated_dot(xs: &[f64], ys: &[f64]) -> f64 {
    assert_eq!(xs.len(), ys.len(), "inputs must have equal lengths");

    let mut dot = NeumaierSum::new();

    for (&x, &y) in xs.iter().zip(ys) {
        let p = x * y;
        let e = x.mul_add(y, -p);

        dot.add(p);
        dot.add(e);
    }

    dot.total()
}

#[cfg(test)]
mod tests {
    use test_utils::approx::{Tolerance, assert_approx_eq};

    use super::*;

    #[test]
    #[should_panic(expected = "inputs must have equal lengths")]
    fn panics_on_length_mismatch() {
        let xs = [1.0];
        let ys = [1.0, 2.0];
        let _ = compensated_dot(&xs, &ys);
    }

    #[test]
    fn returns_zero_on_empty_inputs() {
        let xs = [];
        let ys = [];

        assert_eq!(compensated_dot(&xs, &ys), 0.0);
    }

    #[test]
    fn calculates_an_ordinary_decimal_dot_product() {
        let xs = [0.1, 0.2, 0.3];
        let ys = [0.4, 0.5, 0.6];

        let tol = Tolerance::new(1e-16, 0.0);
        assert_approx_eq(compensated_dot(&xs, &ys), 0.32, tol);
    }

    #[test]
    fn compensates_for_sum_rounding() {
        let p53 = 2.0_f64.powi(53);

        let xs = [p53, 1.0, -p53];
        let ys = [1.0, 1.0, 1.0];

        let naive = xs.iter().zip(&ys).map(|(&x, &y)| x * y).sum::<f64>();

        let compensated = compensated_dot(&xs, &ys);
        let expected = 1.0;

        // p53·1 + 1·1 + (−p53)·1 = 1 in real arithmetic.
        // Naive accumulation lands on 0 because the middle 1 is lost when added
        // to p53. CompensatedDot recovers it.
        assert_eq!(naive, 0.0);
        assert_eq!(compensated, expected);
    }

    #[test]
    fn compensates_for_product_rounding() {
        let delta = 2.0_f64.powi(-27);

        let xs = [1.0 + delta, -1.0];
        let ys = [1.0 - delta, 1.0];

        let naive = xs.iter().zip(&ys).map(|(&x, &y)| x * y).sum::<f64>();

        let compensated = compensated_dot(&xs, &ys);
        let expected = -2.0_f64.powi(-54);

        // With δ = 2^-27, the exact dot product is:
        //
        // (1 + δ)(1 - δ) + (-1)(1)
        // = (1 - δ^2) - 1
        // = -δ^2
        // = -2^-54.
        //
        // The first product, 1 - 2^-54, lies partway between adjacent `f64`
        // values around 1 and rounds to 1 under round-to-nearest, ties-to-even.
        // Naive evaluation therefore produces 1 - 1 = 0. The fused multiply-add
        // recovers the lost product residual, and compensated summation
        // preserves it.
        assert_eq!(naive, 0.0);
        assert_eq!(compensated, expected);
    }
}
