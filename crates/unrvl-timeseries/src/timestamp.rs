use std::time::Duration;

use thiserror::Error;

/// An epoch timestamp outside the range of signed Unix nanoseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum TimestampError {
    /// Converting Unix seconds to nanoseconds would overflow `i64`.
    #[error("Unix timestamp {value} seconds is outside the signed ns range")]
    SecondsOutOfRange {
        /// Number of seconds supplied to the constructor.
        value: i64,
    },

    /// Converting Unix milliseconds to nanoseconds would overflow `i64`.
    #[error("Unix timestamp {value} milliseconds is outside the signed ns range")]
    MillisecondsOutOfRange {
        /// Number of milliseconds supplied to the constructor.
        value: i64,
    },

    /// Converting Unix microseconds to nanoseconds would overflow `i64`.
    #[error("Unix timestamp {value} microseconds is outside the signed ns range")]
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
/// Uses the Unix timescale, which excludes leap seconds. Resolve local times
/// to Unix time before construction. Converting milliseconds to nanoseconds
/// preserves the precision of the source value.
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
mod tests;
