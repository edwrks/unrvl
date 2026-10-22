//! Power-of-two scaling and binary exponents for `f64`.
//!
//! [`power_of_two_unit`] chooses a scale: E.g., for `12`, it returns `8`.
//!
//! [`binary_exponent`] gives the corresponding exponent: `3`, because
//! `12 = 1.5 * 2^3`. Both handle subnormal values directly from their bits.
//!
//! [`scale_power_of_two`] computes `value * 2^exponent` without requiring
//! `2^exponent` itself to fit in `f64`. Exact results in the normal range are
//! preserved; smaller magnitudes are rounded once into the representable range.
//!
//! For example, `2^-100 * 2^1100 = 2^1000` fits in `f64`, even though the
//! factor `2^1100` does not. Combining the exponents first avoids overflow.

/// Binary64 stores 52 fraction bits; its leading normal significand bit is
/// implicit.
const FRACTION_BITS: u32 = f64::MANTISSA_DIGITS - 1;

/// The 11 encoded exponent bits, excluding the sign and fraction fields.
const EXPONENT_MASK: u64 = 0x7ff0_0000_0000_0000;

/// Encoded normal exponents are biased by 1023 in binary64.
const EXPONENT_BIAS: i32 = f64::MAX_EXP - 1;

/// Smallest normal exponent: -1022, with a magnitude of `2^-1022`.
/// Rust's `MIN_EXP` is one greater, so subtract one.
const MIN_NORMAL_EXP: i32 = f64::MIN_EXP - 1;

/// Largest exponent for a normal significand in `[1, 2)`.
const MAX_NORMAL_EXP: i32 = f64::MAX_EXP - 1;

/// Smallest subnormal exponent: -1074, since `2^-1022 * 2^-52 = 2^-1074`.
/// The extra -52 accounts for the positions of the stored fraction bits.
const MIN_SUBNORMAL_EXP: i32 = MIN_NORMAL_EXP - FRACTION_BITS.cast_signed();

/// Exponent -1075 gives half the smallest subnormal when the fraction is one.
/// That tie rounds to zero; a fraction greater than one rounds up.
const HALF_SUBNORMAL_EXP: i32 = MIN_SUBNORMAL_EXP - 1;

/// Returns the largest power of two no greater than `value.abs()`.
///
/// For finite, nonzero values, returns a positive scale such that
/// `value.abs() / scale` is in `[1, 2)`, including for subnormals.
///
/// Returns `1.0` for either signed zero, and `NaN` for non-finite values.
///
/// # Examples
///
/// ```
/// # use unrvl_numerics::scaling::power_of_two_unit;
/// assert_eq!(power_of_two_unit(-12.0), 8.0);
/// assert_eq!(power_of_two_unit(0.0), 1.0);
/// ```
#[must_use]
#[inline]
pub fn power_of_two_unit(value: f64) -> f64 {
    if !value.is_finite() {
        return f64::NAN;
    }

    if value == 0.0 {
        return 1.0;
    }

    let bits = value.abs().to_bits();
    let exponent = bits & EXPONENT_MASK;

    if exponent != 0 {
        // For normal values, clearing the fraction keeps just the power of two:
        // 12 = 1.5 * 2^3 becomes 8 = 1.0 * 2^3.
        f64::from_bits(exponent)
    } else {
        // Subnormals have no implicit leading 1. Their highest set fraction bit
        // determines the largest representable power of two.
        f64::from_bits(1_u64 << bits.ilog2())
    }
}

/// Returns `floor(log2(value.abs()))` for a nonzero finite value.
///
/// The exponent is in `[-1074, 1023]` and satisfies
/// `value = fraction * 2^exponent`, where `1 <= fraction.abs() < 2`.
///
/// Extracting the exponent from the bit representation avoids logarithm
/// rounding errors near powers of two and handles subnormal values.
///
/// Returns `None` for either signed zero, NaN, or infinity.
///
/// # Examples
///
/// ```
/// # use unrvl_numerics::scaling::binary_exponent;
/// assert_eq!(binary_exponent(-12.0), Some(3));
/// assert_eq!(binary_exponent(f64::from_bits(1)), Some(-1074));
/// assert_eq!(binary_exponent(0.0), None);
/// ```
#[must_use]
#[inline]
pub fn binary_exponent(value: f64) -> Option<i32> {
    if value == 0.0 || !value.is_finite() {
        return None;
    }

    let bits = value.abs().to_bits();
    let encoded = (bits & EXPONENT_MASK) >> FRACTION_BITS;

    let exponent = if encoded == 0 {
        // Subnormals are stored as fraction_bits * 2^-1074.
        // The highest set bit at position k gives exponent k - 1074.
        bits.ilog2().cast_signed() + MIN_SUBNORMAL_EXP
    } else {
        // Normal exponents are stored with a bias of 1023.
        i32::try_from(encoded).ok()? - EXPONENT_BIAS
    };

    Some(exponent)
}

/// Returns `value * 2^exponent`, avoiding an overflowing or underflowing scale.
///
/// Combining the binary exponents before scaling avoids intermediate overflow
/// or underflow. Finite values whose exact scaled result lies in the normal
/// range are scaled exactly. Results below that range are rounded to nearest,
/// ties to even, possibly to zero or the smallest normal value.
///
/// Overflow returns signed infinity, while underflow preserves the sign of
/// zero. NaN, infinities, and signed zeros are returned unchanged.
///
/// Accepts any `i32` exponent.
///
/// # Examples
///
/// ```
/// # use unrvl_numerics::scaling::scale_power_of_two;
/// let tiny = f64::from_bits(1);
/// let large = f64::from_bits(0x7fe0_0000_0000_0000); // 2^1023
///
/// assert_eq!(scale_power_of_two(tiny, 2097), large);
/// assert_eq!(scale_power_of_two(large, -2097), tiny);
/// ```
#[must_use]
#[inline]
pub fn scale_power_of_two(value: f64, exponent: i32) -> f64 {
    let Some(value_exponent) = binary_exponent(value) else {
        // Scaling leaves zeros, infinities, and NaNs unchanged. Returning the
        // original value also preserves the sign of zero.
        return value;
    };

    // Normalize to a signed fraction with magnitude in [1, 2).
    let unit = power_of_two_unit(value);
    let fraction = value / unit;

    // Saturation is safe: either integer-overflow endpoint lies far outside
    // the representable binary64 exponent range.
    let exponent = exponent.saturating_add(value_exponent);

    if exponent > MAX_NORMAL_EXP {
        // Case: Overflow
        //
        // Even a fraction of 1 gives at least 2^1024, too large for f64.
        f64::INFINITY.copysign(value)
    } else if exponent < HALF_SUBNORMAL_EXP {
        // Case: Underflow to Zero
        //
        // Even a fraction just below 2 gives less than half the smallest
        // subnormal. The nearest representable result is signed zero.
        0.0_f64.copysign(value)
    } else if exponent < MIN_NORMAL_EXP {
        // Case: Subnormal
        //
        // Factor out 2^-1022 so the first multiplication stays normal
        // and exact. The final multiplication rounds once to the tiny result.
        //
        // Include -1075 because values above the halfway point can still round
        // to the smallest subnormal.
        let factor = normal_power_of_two(exponent - MIN_NORMAL_EXP);
        (fraction * factor) * f64::MIN_POSITIVE
    } else {
        // Case: Normal
        //
        // The result is normal, so shifting its exponent can preserve all
        // fraction bits. The power-of-two factor also fits in f64.
        fraction * normal_power_of_two(exponent)
    }
}

/// Constructs exactly `2^exponent` for a normal-range exponent.
#[inline]
fn normal_power_of_two(exponent: i32) -> f64 {
    debug_assert!((MIN_NORMAL_EXP..=MAX_NORMAL_EXP).contains(&exponent));

    // A biased exponent with a zero fraction represents 1.0 * 2^exponent.
    let encoded = (exponent + EXPONENT_BIAS).cast_unsigned();
    f64::from_bits(u64::from(encoded) << FRACTION_BITS)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_utils::strategies::finite_value;

    use super::{binary_exponent, power_of_two_unit, scale_power_of_two};

    mod power_of_two_unit {
        use proptest::prelude::*;
        use test_utils::strategies::finite_value;

        use crate::scaling::power_of_two_unit;

        #[test]
        fn selects_units_for_normal_values() {
            let large = f64::from_bits(0x7fe0_0000_0000_0000); // 2^1023
            let table = [
                (1.0, 1.0),
                (0.5, 0.5),
                (3.0, 2.0),
                (f64::MAX, large),
                (f64::MIN_POSITIVE, f64::MIN_POSITIVE),
            ];

            for (value, expected) in table {
                assert_eq!(power_of_two_unit(value), expected);
                assert_eq!(power_of_two_unit(-value), expected);
            }
        }

        #[test]
        fn selects_units_for_subnormal_values() {
            let tiny = f64::from_bits(1);

            for (value, expected) in [(tiny, tiny), (3.0 * tiny, 2.0 * tiny)] {
                assert_eq!(power_of_two_unit(value), expected);
                assert_eq!(power_of_two_unit(-value), expected);
            }
        }

        #[test]
        fn returns_one_for_signed_zeros() {
            for value in [0.0_f64, -0.0] {
                assert_eq!(power_of_two_unit(value), 1.0);
            }
        }

        #[test]
        fn returns_nan_for_nonfinite_values() {
            for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                assert!(power_of_two_unit(value).is_nan());
            }
        }

        proptest! {
            #[test]
            fn units_normalize_finite_nonzero_magnitudes(
                value in finite_value().prop_filter("nonzero value", |value| *value != 0.0),
            ) {
                let unit = power_of_two_unit(value);
                prop_assert!((1.0..2.0).contains(&(value.abs() / unit)));
            }
        }
    }

    mod binary_exponent {
        use crate::scaling::{MAX_NORMAL_EXP, MIN_NORMAL_EXP, MIN_SUBNORMAL_EXP, binary_exponent};

        #[test]
        fn extracts_normal_exponents() {
            let table = [
                (1.0, 0),
                (0.5, -1),
                (3.0, 1),
                (f64::MAX, MAX_NORMAL_EXP),
                (f64::MIN_POSITIVE, MIN_NORMAL_EXP),
            ];

            for (value, expected) in table {
                assert_eq!(binary_exponent(value), Some(expected));
                assert_eq!(binary_exponent(-value), Some(expected));
            }
        }

        #[test]
        fn extracts_subnormal_exponents() {
            let tiny = f64::from_bits(1);
            let table = [
                (tiny, MIN_SUBNORMAL_EXP),
                (3.0 * tiny, MIN_SUBNORMAL_EXP + 1),
            ];

            for (value, expected) in table {
                assert_eq!(binary_exponent(value), Some(expected));
                assert_eq!(binary_exponent(-value), Some(expected));
            }
        }

        #[test]
        fn returns_none_for_signed_zeros() {
            for value in [0.0_f64, -0.0] {
                assert_eq!(binary_exponent(value), None);
            }
        }

        #[test]
        fn returns_none_for_nonfinite_values() {
            for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                assert_eq!(binary_exponent(value), None);
            }
        }
    }

    mod scale_power_of_two {
        use proptest::prelude::*;
        use test_utils::strategies::finite_value;

        use crate::scaling::{
            EXPONENT_BIAS, FRACTION_BITS, HALF_SUBNORMAL_EXP, MIN_NORMAL_EXP, MIN_SUBNORMAL_EXP,
            scale_power_of_two,
        };

        #[test]
        fn preserves_signed_zeros() {
            for value in [0.0_f64, -0.0] {
                for exponent in [i32::MIN, 0, i32::MAX] {
                    assert_eq!(
                        scale_power_of_two(value, exponent).to_bits(),
                        value.to_bits()
                    );
                }
            }
        }

        #[test]
        fn preserves_nonfinite_values() {
            for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let scaled = scale_power_of_two(value, i32::MAX);
                if value.is_nan() {
                    assert!(scaled.is_nan());
                } else {
                    assert_eq!(scaled, value);
                }
            }
        }

        #[test]
        fn combines_extreme_exponents_before_scaling() {
            let tiny = f64::from_bits(1);
            let large = f64::from_bits(0x7fe0_0000_0000_0000); // 2^1023

            assert_eq!(scale_power_of_two(tiny, 2097), large);
            assert_eq!(scale_power_of_two(large, -2097), tiny);
            assert_eq!(scale_power_of_two(-tiny, 2097), -large);
        }

        #[test]
        fn returns_signed_infinity_on_overflow() {
            assert_eq!(scale_power_of_two(f64::MAX, i32::MAX), f64::INFINITY);
            assert_eq!(scale_power_of_two(-f64::MAX, i32::MAX), f64::NEG_INFINITY);
        }

        #[test]
        fn preserves_negative_zero_on_underflow() {
            let tiny = f64::from_bits(1);

            assert_eq!(
                scale_power_of_two(-tiny, i32::MIN).to_bits(),
                (-0.0_f64).to_bits()
            );
        }

        #[test]
        fn scales_to_the_smallest_subnormal() {
            let tiny = f64::from_bits(1);

            assert_eq!(scale_power_of_two(1.0, MIN_SUBNORMAL_EXP), tiny);
        }

        #[test]
        fn rounds_subnormal_halfway_cases_to_even() {
            let tiny = f64::from_bits(1);

            assert_eq!(scale_power_of_two(1.0, HALF_SUBNORMAL_EXP), 0.0);
            assert_eq!(scale_power_of_two(3.0, HALF_SUBNORMAL_EXP), 2.0 * tiny);
            assert_eq!(scale_power_of_two(5.0, HALF_SUBNORMAL_EXP), 2.0 * tiny);
        }

        #[test]
        fn rounds_above_halfway_to_the_smallest_subnormal() {
            let tiny = f64::from_bits(1);
            let a = scale_power_of_two(1.0_f64.next_up(), HALF_SUBNORMAL_EXP);
            let b = scale_power_of_two(-1.5, HALF_SUBNORMAL_EXP);

            assert_eq!(a, tiny);
            assert_eq!(b, -tiny);
        }

        #[test]
        fn rounds_at_the_normal_subnormal_boundary() {
            let a = scale_power_of_two(1.0_f64.next_down(), MIN_NORMAL_EXP);
            let b = scale_power_of_two(1.0_f64.next_down().next_down(), MIN_NORMAL_EXP);

            assert_eq!(a, f64::MIN_POSITIVE);
            assert_eq!(b, f64::MIN_POSITIVE.next_down());
        }

        proptest! {
            #[test]
            fn agrees_with_one_multiplication_when_the_factor_fits(
                value in finite_value(),
                encoded_exponent in 1_u16..2047,
            ) {
                // Encodings 1..2047 describe all normal powers of two.
                // Generate the factor directly, independently of the scaling operation.
                let factor = f64::from_bits(u64::from(encoded_exponent) << FRACTION_BITS);
                let exponent = i32::from(encoded_exponent) - EXPONENT_BIAS;
                let expected = value * factor;
                prop_assert_eq!(scale_power_of_two(value, exponent).to_bits(), expected.to_bits());
            }
        }
    }

    proptest! {
        #[test]
        fn unit_and_exponent_describe_the_same_power_of_two(
            value in finite_value().prop_filter("nonzero value", |value| *value != 0.0),
        ) {
            let unit = power_of_two_unit(value);
            let exponent = binary_exponent(value).unwrap();
            prop_assert_eq!(scale_power_of_two(unit, -exponent).to_bits(), 1.0_f64.to_bits());
        }
    }
}
