use std::time::Duration;

use thiserror::Error;

/// An epoch timestamp outside the range of signed Unix nanoseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum TimestampError {
    /// Converting Unix seconds to nanoseconds would overflow `i64`.
    #[error("Unix timestamp {value} seconds is outside the signed nanosecond range")]
    SecondsOutOfRange {
        /// Number of seconds supplied to the constructor.
        value: i64,
    },

    /// Converting Unix milliseconds to nanoseconds would overflow `i64`.
    #[error("Unix timestamp {value} milliseconds is outside the signed nanosecond range")]
    MillisecondsOutOfRange {
        /// Number of milliseconds supplied to the constructor.
        value: i64,
    },

    /// Converting Unix microseconds to nanoseconds would overflow `i64`.
    #[error("Unix timestamp {value} microseconds is outside the signed nanosecond range")]
    MicrosecondsOutOfRange {
        /// Number of microseconds supplied to the constructor.
        value: i64,
    },
}

/// An instant represented as signed nanoseconds since the Unix epoch.
///
/// The epoch is `1970-01-01T00:00:00Z`. Negative values precede the epoch.
/// All `i64` values are representable, covering approximately 1677-2262.
/// Timestamps compare by instant and preserve integer nanoseconds exactly.
///
/// Uses the Unix timescale without representing leap seconds. The type does
/// not resolve timezones: converting milliseconds into nanoseconds does not
/// make the observations more precise.
///
/// # Examples
///
/// ```
/// # use unrvl_timeseries::{Timestamp, TimestampError};
/// let timestamp = Timestamp::from_unix_millis(1_500)?;
/// assert_eq!(timestamp.as_unix_nanos(), 1_500_000_000);
///
/// let before_epoch = Timestamp::from_unix_nanos(-1);
/// assert!(before_epoch < Timestamp::UNIX_EPOCH);
/// # Ok::<(), TimestampError>(())
/// ```
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Timestamp(i64);

impl Timestamp {
    /// Earliest representable timestamp: `i64::MIN` Unix nanoseconds.
    pub const MIN: Self = Self(i64::MIN);

    /// Latest representable timestamp: `i64::MAX` Unix nanoseconds.
    pub const MAX: Self = Self(i64::MAX);

    /// The Unix epoch, `1970-01-01T00:00:00Z`.
    pub const UNIX_EPOCH: Self = Self(0);

    /// Creates a timestamp from signed Unix seconds.
    ///
    /// # Errors
    ///
    /// Returns [`TimestampError::SecondsOutOfRange`] if the nanosecond count
    /// cannot be represented by `i64`.
    pub const fn from_unix_secs(secs: i64) -> Result<Self, TimestampError> {
        match secs.checked_mul(1_000_000_000) {
            Some(nanos) => Ok(Self(nanos)),
            None => Err(TimestampError::SecondsOutOfRange { value: secs }),
        }
    }

    /// Creates a timestamp from signed Unix milliseconds.
    ///
    /// # Errors
    ///
    /// Returns [`TimestampError::MillisecondsOutOfRange`] if the nanosecond
    /// count cannot be represented by `i64`.
    pub const fn from_unix_millis(millis: i64) -> Result<Self, TimestampError> {
        match millis.checked_mul(1_000_000) {
            Some(nanos) => Ok(Self(nanos)),
            None => Err(TimestampError::MillisecondsOutOfRange { value: millis }),
        }
    }

    /// Creates a timestamp from signed Unix microseconds.
    ///
    /// # Errors
    ///
    /// Returns [`TimestampError::MicrosecondsOutOfRange`] if the nanosecond
    /// count cannot be represented by `i64`.
    pub const fn from_unix_micros(micros: i64) -> Result<Self, TimestampError> {
        match micros.checked_mul(1_000) {
            Some(nanos) => Ok(Self(nanos)),
            None => Err(TimestampError::MicrosecondsOutOfRange { value: micros }),
        }
    }

    /// Creates a timestamp from signed Unix nanoseconds, without conversion.
    #[must_use]
    pub const fn from_unix_nanos(nanos: i64) -> Self {
        Self(nanos)
    }

    /// Returns the signed number of nanoseconds since the Unix epoch exactly.
    #[must_use]
    pub const fn as_unix_nanos(self) -> i64 {
        self.0
    }

    /// Adds a nonnegative elapsed duration.
    ///
    /// Returns `None` if the resulting timestamp is out of range. Durations
    /// larger than `i64::MAX` nanoseconds are valid when the result fits.
    #[must_use]
    pub fn checked_add(self, duration: Duration) -> Option<Self> {
        let nanos = u64::try_from(duration.as_nanos()).ok()?;

        self.0.checked_add_unsigned(nanos).map(Self)
    }

    /// Subtracts a nonnegative elapsed duration.
    ///
    /// Returns `None` if the resulting timestamp is out of range. Durations
    /// larger than `i64::MAX` nanoseconds are valid when the result fits.
    #[must_use]
    pub fn checked_sub(self, duration: Duration) -> Option<Self> {
        let nanos = u64::try_from(duration.as_nanos()).ok()?;

        self.0.checked_sub_unsigned(nanos).map(Self)
    }

    /// Returns the nonnegative elapsed duration since `earlier`.
    ///
    /// Returns `None` if `earlier` is later than this timestamp. Equal
    /// timestamps yield [`Duration::ZERO`]. Every ordered pair is supported,
    /// including the full span from [`Self::MIN`] to [`Self::MAX`].
    #[must_use]
    pub fn checked_duration_since(self, earlier: Self) -> Option<Duration> {
        if self < earlier {
            return None;
        }

        Some(Duration::from_nanos(self.0.abs_diff(earlier.0)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod from_unix_secs {
        use super::*;

        #[test]
        fn rejects_unrepresentable_seconds() {
            // First whole-second magnitude that overflows when converted to
            // nanoseconds.
            let first_out_of_range = 9_223_372_037;
            let cases = [i64::MIN, -first_out_of_range, first_out_of_range, i64::MAX];

            for value in cases {
                let expected = TimestampError::SecondsOutOfRange { value };

                assert_eq!(Timestamp::from_unix_secs(value), Err(expected));
            }
        }

        #[test]
        fn converts_seconds_exactly_including_range_boundaries() {
            // Largest whole-second magnitude that still fits in signed
            // nanoseconds.
            let boundary_secs = 9_223_372_036;
            let boundary_nanos = 9_223_372_036_000_000_000;
            let cases = [
                (-boundary_secs, -boundary_nanos),
                (-1, -1_000_000_000),
                (0, 0),
                (1, 1_000_000_000),
                (boundary_secs, boundary_nanos),
            ];

            for (value, expected) in cases {
                let timestamp = Timestamp::from_unix_secs(value).unwrap();

                assert_eq!(timestamp.as_unix_nanos(), expected);
            }
        }
    }

    mod from_unix_millis {
        use super::*;

        #[test]
        fn rejects_unrepresentable_milliseconds() {
            // First whole-millisecond magnitude that overflows in nanoseconds.
            let first_out_of_range = 9_223_372_036_855;
            let cases = [i64::MIN, -first_out_of_range, first_out_of_range, i64::MAX];

            for value in cases {
                let expected = TimestampError::MillisecondsOutOfRange { value };

                assert_eq!(Timestamp::from_unix_millis(value), Err(expected));
            }
        }

        #[test]
        fn converts_milliseconds_exactly_including_range_boundaries() {
            // Largest whole-millisecond magnitude that still fits in signed
            // nanoseconds.
            let boundary_millis = 9_223_372_036_854;
            let boundary_nanos = 9_223_372_036_854_000_000;
            let cases = [
                (-boundary_millis, -boundary_nanos),
                (-1, -1_000_000),
                (0, 0),
                (1, 1_000_000),
                (boundary_millis, boundary_nanos),
            ];

            for (value, expected) in cases {
                let timestamp = Timestamp::from_unix_millis(value).unwrap();

                assert_eq!(timestamp.as_unix_nanos(), expected);
            }
        }
    }

    mod from_unix_micros {
        use super::*;

        #[test]
        fn rejects_unrepresentable_microseconds() {
            // First whole-microsecond magnitude that overflows in nanoseconds.
            let first_out_of_range = 9_223_372_036_854_776;
            let cases = [i64::MIN, -first_out_of_range, first_out_of_range, i64::MAX];

            for value in cases {
                let expected = TimestampError::MicrosecondsOutOfRange { value };

                assert_eq!(Timestamp::from_unix_micros(value), Err(expected));
            }
        }

        #[test]
        fn converts_microseconds_exactly_including_range_boundaries() {
            // Largest whole-microsecond magnitude that still fits in signed
            // nanoseconds.
            let boundary_micros = 9_223_372_036_854_775;
            let boundary_nanos = 9_223_372_036_854_775_000;
            let cases = [
                (-boundary_micros, -boundary_nanos),
                (-1, -1_000),
                (0, 0),
                (1, 1_000),
                (boundary_micros, boundary_nanos),
            ];

            for (value, expected) in cases {
                let timestamp = Timestamp::from_unix_micros(value).unwrap();

                assert_eq!(timestamp.as_unix_nanos(), expected);
            }
        }
    }

    mod checked_add {
        use super::*;

        #[test]
        fn returns_none_when_the_result_is_out_of_range() {
            // MIN to the epoch is 2^63 ns, one more than i64::MAX.
            let min_to_epoch = Duration::from_nanos(1_u64 << 63);

            // 2^64 ns is one nanosecond longer than the full timestamp span.
            let beyond_full_span = Duration::new(18_446_744_073, 709_551_616);

            let cases = [
                (Timestamp::MAX, Duration::from_nanos(1)),
                (Timestamp::UNIX_EPOCH, min_to_epoch),
                (Timestamp::MIN, beyond_full_span),
                (Timestamp::MIN, Duration::MAX),
            ];

            for (timestamp, duration) in cases {
                assert_eq!(timestamp.checked_add(duration), None);
            }
        }

        #[test]
        fn adds_exactly_across_epoch_and_range_boundaries() {
            // -1.500000001 s + 2.000000002 s = 0.500000001 s.
            let before_epoch_nanos = -1_500_000_001;
            let after_epoch_nanos = 500_000_001;
            let across_epoch = Duration::new(2, 2);

            // MIN to the epoch is 2^63 ns; MIN to MAX is 2^64 - 1 ns.
            let min_to_epoch = Duration::from_nanos(1_u64 << 63);
            let full_span = Duration::from_nanos(u64::MAX);

            let cases = [
                (before_epoch_nanos, across_epoch, after_epoch_nanos),
                (-1, Duration::from_nanos(1), 0),
                (0, Duration::from_nanos(1), 1),
                (i64::MIN, Duration::from_nanos(1), i64::MIN + 1),
                (i64::MAX - 1, Duration::from_nanos(1), i64::MAX),
                (i64::MIN, min_to_epoch, 0),
                (i64::MIN, full_span, i64::MAX),
            ];

            for (start, duration, expected) in cases {
                let timestamp = Timestamp::from_unix_nanos(start);
                let expected = Timestamp::from_unix_nanos(expected);

                assert_eq!(timestamp.checked_add(duration), Some(expected));
            }
        }

        #[test]
        fn zero_preserves_the_timestamp_including_range_limits() {
            let cases = [i64::MIN, -1, 0, 1, i64::MAX];

            for value in cases {
                let timestamp = Timestamp::from_unix_nanos(value);

                assert_eq!(timestamp.checked_add(Duration::ZERO), Some(timestamp));
            }
        }
    }

    mod checked_sub {
        use super::*;

        #[test]
        fn returns_none_when_the_result_is_out_of_range() {
            // Subtracting 2^63 ns from the epoch reaches MIN; starting at -1
            // overflows.
            let min_to_epoch = Duration::from_nanos(1_u64 << 63);
            // 2^64 ns is one nanosecond longer than the full timestamp span.
            let beyond_full_span = Duration::new(18_446_744_073, 709_551_616);

            let cases = [
                (Timestamp::MIN, Duration::from_nanos(1)),
                (Timestamp::from_unix_nanos(-1), min_to_epoch),
                (Timestamp::MAX, beyond_full_span),
                (Timestamp::MAX, Duration::MAX),
            ];

            for (timestamp, duration) in cases {
                assert_eq!(timestamp.checked_sub(duration), None);
            }
        }

        #[test]
        fn subtracts_exactly_across_epoch_and_range_boundaries() {
            // 0.500000001 s - 2.000000002 s = -1.500000001 s.
            let before_epoch_nanos = -1_500_000_001;
            let after_epoch_nanos = 500_000_001;
            let across_epoch = Duration::new(2, 2);

            // Epoch to MIN subtracts 2^63 ns; MAX to MIN subtracts 2^64 - 1 ns.
            let min_to_epoch = Duration::from_nanos(1_u64 << 63);
            let full_span = Duration::from_nanos(u64::MAX);

            let cases = [
                (after_epoch_nanos, across_epoch, before_epoch_nanos),
                (0, Duration::from_nanos(1), -1),
                (1, Duration::from_nanos(1), 0),
                (i64::MIN + 1, Duration::from_nanos(1), i64::MIN),
                (i64::MAX, Duration::from_nanos(1), i64::MAX - 1),
                (0, min_to_epoch, i64::MIN),
                (i64::MAX, full_span, i64::MIN),
            ];

            for (start, duration, expected) in cases {
                let timestamp = Timestamp::from_unix_nanos(start);
                let expected = Timestamp::from_unix_nanos(expected);

                assert_eq!(timestamp.checked_sub(duration), Some(expected));
            }
        }

        #[test]
        fn zero_preserves_the_timestamp_including_range_limits() {
            let cases = [i64::MIN, -1, 0, 1, i64::MAX];

            for value in cases {
                let timestamp = Timestamp::from_unix_nanos(value);

                assert_eq!(timestamp.checked_sub(Duration::ZERO), Some(timestamp));
            }
        }
    }

    mod checked_duration_since {
        use super::*;

        #[test]
        fn returns_none_for_reversed_timestamps() {
            let cases = [(i64::MIN, i64::MAX), (-1, 0), (0, 1)];

            for (later, earlier) in cases {
                let timestamp = Timestamp::from_unix_nanos(later);
                let earlier = Timestamp::from_unix_nanos(earlier);

                assert_eq!(timestamp.checked_duration_since(earlier), None);
            }
        }

        #[test]
        fn measures_exactly_across_epoch_and_range_boundaries() {
            // The fractional-second observations are 2 seconds and 2
            // nanoseconds apart.
            let before_epoch_nanos = -1_500_000_001;
            let after_epoch_nanos = 500_000_001;
            let across_epoch = Duration::new(2, 2);

            // Independently written as seconds/subseconds: 2^63, 2^63 - 1,
            // and 2^64 - 1 nanoseconds, respectively.
            let min_to_epoch = Duration::new(9_223_372_036, 854_775_808);
            let epoch_to_max = Duration::new(9_223_372_036, 854_775_807);
            let full_span = Duration::new(18_446_744_073, 709_551_615);

            let cases = [
                (after_epoch_nanos, before_epoch_nanos, across_epoch),
                (0, -1, Duration::from_nanos(1)),
                (1, 0, Duration::from_nanos(1)),
                (i64::MIN + 1, i64::MIN, Duration::from_nanos(1)),
                (i64::MAX, i64::MAX - 1, Duration::from_nanos(1)),
                (0, i64::MIN, min_to_epoch),
                (i64::MAX, 0, epoch_to_max),
                (i64::MAX, i64::MIN, full_span),
            ];

            for (later, earlier, expected) in cases {
                let timestamp = Timestamp::from_unix_nanos(later);
                let earlier = Timestamp::from_unix_nanos(earlier);

                assert_eq!(timestamp.checked_duration_since(earlier), Some(expected));
            }
        }

        #[test]
        fn equal_timestamps_have_zero_elapsed_duration() {
            let cases = [i64::MIN, -1, 0, 1, i64::MAX];

            for value in cases {
                let timestamp = Timestamp::from_unix_nanos(value);

                assert_eq!(
                    timestamp.checked_duration_since(timestamp),
                    Some(Duration::ZERO),
                );
            }
        }
    }

    #[test]
    fn nanoseconds_round_trip_without_precision_loss() {
        // 2^53 + 1 is the first positive integer that f64 cannot represent
        // exactly.
        let beyond_f64_precision = 9_007_199_254_740_993;
        let cases = [
            i64::MIN,
            i64::MIN + 1,
            -beyond_f64_precision,
            -1,
            0,
            1,
            beyond_f64_precision,
            i64::MAX - 1,
            i64::MAX,
        ];

        for value in cases {
            let timestamp = Timestamp::from_unix_nanos(value);

            assert_eq!(timestamp.as_unix_nanos(), value);
        }
    }

    #[test]
    fn constants_identify_epoch_and_range_limits() {
        assert_eq!(Timestamp::UNIX_EPOCH.as_unix_nanos(), 0);
        assert_eq!(Timestamp::MIN.as_unix_nanos(), i64::MIN);
        assert_eq!(Timestamp::MAX.as_unix_nanos(), i64::MAX);
    }
}
