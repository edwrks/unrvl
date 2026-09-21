//! Numeric conversions with explicit representability guarantees.

/// Converts a `usize` to `f64` when it is within the lossless integer range.
///
/// # Panics
///
/// Panics if `n` exceeds the largest consecutive integer exactly
/// representable by `f64`, which is `2^53`.
#[must_use]
#[inline]
#[expect(clippy::cast_precision_loss, reason = "asserts exact representability")]
pub fn usize_to_f64(n: usize) -> f64 {
    const MAX_CONSECUTIVE_INT: u64 = 1_u64 << f64::MANTISSA_DIGITS;

    let valid = u64::try_from(n).is_ok_and(|value| value <= MAX_CONSECUTIVE_INT);
    assert!(valid, "value must be exactly representable as f64");

    n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_representable_values() {
        assert_eq!(usize_to_f64(0), 0.0);
        assert_eq!(usize_to_f64(1), 1.0);
        assert_eq!(usize_to_f64(42), 42.0);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn accepts_end_of_consecutive_exact_range() {
        let value = 1_usize << f64::MANTISSA_DIGITS;
        assert_eq!(usize_to_f64(value), 9_007_199_254_740_992.0,);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    #[should_panic(expected = "value must be exactly representable as f64")]
    fn rejects_value_above_consecutive_exact_range() {
        let value = (1_usize << f64::MANTISSA_DIGITS) + 1;
        let _ = usize_to_f64(value);
    }
}
