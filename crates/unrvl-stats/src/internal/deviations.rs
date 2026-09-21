use unrvl_numerics::convert::usize_to_f64;
use unrvl_numerics::summation::NeumaierSumExt;

const EXPONENT_MASK: u64 = 0x7ff0_0000_0000_0000;

/// Returns a power-of-two scaling unit and corrected centered deviations.
///
/// Observations are divided by the unit before calculating their center,
/// keeping the centering arithmetic in a small numerical range. The mean
/// and its rounding correction are calculated with Neumaier compensated
/// summation.
///
/// The returned unit is positive and finite. The deviations are expressed
/// in that unit and retain the centering correction separately, rather than
/// adding it back to the rounded center.
///
/// A finite constant sample produces zero deviations. An all-zero sample
/// uses a unit of one.
///
/// Power-of-two scaling preserves values when the scaled results are
/// representable. Very small observations relative to the largest magnitude
/// can still lose precision or underflow during scaling.
///
/// Returns `None` for an empty slice or if any observation is non-finite.
///
/// # Panics
///
/// Panics if a finite sample contains more than `2^53` observations.
#[must_use]
#[expect(clippy::float_cmp, reason = "detect exactly constant")]
pub fn centered_scaled(xs: &[f64]) -> Option<(f64, impl Iterator<Item = f64> + Clone + '_)> {
    let &first = xs.first()?;

    let mut maximum = 0.0_f64;
    let mut is_constant = true;

    for &x in xs {
        if !x.is_finite() {
            return None;
        }

        maximum = maximum.max(x.abs());
        is_constant &= x == first;
    }

    let unit = power_of_two_unit(maximum);
    let n = usize_to_f64(xs.len());
    let scaled = xs.iter().map(move |&x| x / unit);

    let center = if is_constant {
        // Preserve a constant sample's scaled center exactly.
        first / unit
    } else {
        scaled.clone().neumaier_sum() / n
    };

    let correction = scaled.clone().map(|x| x - center).neumaier_sum() / n;
    let deviations = scaled.map(move |x| (x - center) - correction);

    Some((unit, deviations))
}

/// Returns the largest power of two no greater than a positive `x`.
///
/// Returns one when `x` is zero.
///
/// For positive `x`, dividing by the returned unit produces a value in
/// `[1, 2)`. The unit is represented exactly, including when `x` is subnormal.
///
/// # Preconditions
///
/// `x` must be finite and nonnegative.
#[inline]
fn power_of_two_unit(x: f64) -> f64 {
    debug_assert!(x.is_finite() && x >= 0.0);

    if x == 0.0 {
        return 1.0;
    }

    let bits = x.to_bits();
    let exponent = bits & EXPONENT_MASK;

    if exponent != 0 {
        // Normal value: retain its exponent and clear its fraction.
        f64::from_bits(exponent)
    } else {
        // Subnormal value: retain its highest set fraction bit.
        // These values have no encoded exponent. Each significand bit therefore
        // represents a power of two directly.
        let highest_bit = bits.ilog2();
        f64::from_bits(1_u64 << highest_bit)
    }
}

#[cfg(test)]
mod tests {
    use test_utils::approx::{Tolerance, assert_all_approx_eq, assert_approx_eq};

    use super::*;

    #[test]
    fn selects_power_of_two_units_across_the_finite_range() {
        assert_eq!(power_of_two_unit(0.0), 1.0);
        assert_eq!(power_of_two_unit(1.0), 1.0);
        assert_eq!(power_of_two_unit(3.0), 2.0);

        assert_eq!(power_of_two_unit(f64::from_bits(3)), f64::from_bits(2));
        assert_eq!(power_of_two_unit(f64::MIN_POSITIVE), f64::MIN_POSITIVE);

        let largest_power_of_two = f64::from_bits(0x7fe0_0000_0000_0000);
        assert_eq!(power_of_two_unit(f64::MAX), largest_power_of_two);
    }

    #[test]
    fn rejects_empty_or_nonfinite_samples() {
        assert!(centered_scaled(&[]).is_none());

        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(centered_scaled(&[1.0, value]).is_none());
        }
    }

    #[test]
    fn centers_in_input_units() {
        let xs = [1000.0, 1001.0, 1002.0];
        let (unit, deviations) = centered_scaled(&xs).unwrap();
        let expected = [-1.0 / 512.0, 0.0, 1.0 / 512.0];

        assert_eq!(unit, 512.0);
        assert_eq!(deviations.collect::<Vec<_>>(), expected,);
    }

    #[test]
    fn centers_subnormal_observations_before_information_is_lost() {
        let d = f64::from_bits(1);
        let xs = [0.0, 0.0, d];
        let (unit, deviations) = centered_scaled(&xs).unwrap();
        let expected = [-1.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0];

        assert_eq!(unit, d);
        assert_all_approx_eq(
            &deviations.collect::<Vec<_>>(),
            &expected,
            Tolerance::VERY_STRICT,
        );
    }

    #[test]
    fn retains_rounding_correction_around_a_large_offset() {
        let xs = [1e16, 1e16, 1e16 + 2.0];
        let (unit, deviations) = centered_scaled(&xs).unwrap();

        let reconstructed: Vec<_> = deviations.map(|d| d * unit).collect();
        let expected = [-2.0 / 3.0, -2.0 / 3.0, 4.0 / 3.0];

        assert_all_approx_eq(&reconstructed, &expected, Tolerance::VERY_STRICT);
    }

    #[test]
    fn centers_opposite_finite_extremes_without_overflow() {
        let xs = [-f64::MAX, 0.0, f64::MAX];
        let (unit, deviations) = centered_scaled(&xs).unwrap();

        assert!(unit.is_finite());
        assert!(unit > 0.0);

        for (actual, expected) in deviations.zip([-f64::MAX / unit, 0.0, f64::MAX / unit]) {
            assert_approx_eq(actual, expected, Tolerance::VERY_STRICT);
        }
    }

    #[test]
    fn constants_have_positive_units_and_zero_deviations() {
        let values = [
            0.0,
            42.0,
            f64::from_bits(1),
            -f64::from_bits(1),
            f64::MAX,
            -f64::MAX,
        ];

        for value in values {
            let xs = [value; 4];
            let (unit, deviations) = centered_scaled(&xs).unwrap();

            assert!(unit.is_finite());
            assert!(unit > 0.0);

            for deviation in deviations {
                assert_eq!(deviation, 0.0);
            }
        }
    }
}
