use std::ops::{Bound, RangeBounds};
use std::time::Duration;

use thiserror::Error;

use crate::Timestamp;

/// Endpoints that cannot form a nonempty [`TimeInterval`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error(
    "interval end ({end_nanos} ns) must be later than start ({start_nanos} ns)",
    end_nanos = .end.as_unix_nanos(),
    start_nanos = .start.as_unix_nanos(),
)]
pub struct IntervalError {
    /// Supplied start, which must be earlier than `end`.
    pub start: Timestamp,

    /// Supplied end, which must be later than `start`.
    pub end: Timestamp,
}

/// A nonempty time interval with an inclusive start and exclusive end.
///
/// Represents `[start, end)` between supplied instants. It can describe one
/// continuous session open period or a schedule's declared coverage.
///
/// Adjacent intervals may share a boundary timestamp, which belongs only
/// to the later interval.
///
/// # Examples
///
/// ```
/// # use unrvl_timeseries::{IntervalError, TimeInterval, Timestamp};
/// let start = Timestamp::UNIX_EPOCH;
/// let end = Timestamp::from_unix_nanos(3);
/// let interval = TimeInterval::new(start, end)?;
///
/// assert!(interval.contains(start));
/// assert!(!interval.contains(end));
/// # Ok::<(), IntervalError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TimeInterval {
    start: Timestamp,
    end: Timestamp,
}

impl TimeInterval {
    /// Creates a nonempty interval `[start, end)`.
    ///
    /// # Errors
    ///
    /// Returns [`IntervalError`] with both supplied endpoints if `start` is
    /// equal to or later than `end`.
    pub fn new(start: Timestamp, end: Timestamp) -> Result<Self, IntervalError> {
        if start >= end {
            return Err(IntervalError { start, end });
        }

        Ok(Self { start, end })
    }

    /// Returns the inclusive start.
    #[must_use]
    pub const fn start(&self) -> Timestamp {
        self.start
    }

    /// Returns the exclusive end.
    #[must_use]
    pub const fn end(&self) -> Timestamp {
        self.end
    }

    /// Returns whether `timestamp` lies in `[start, end)`.
    #[must_use]
    pub fn contains(&self, timestamp: Timestamp) -> bool {
        self.start <= timestamp && timestamp < self.end
    }

    /// Returns the positive elapsed duration between the endpoints exactly.
    ///
    /// Supports the full span from [`Timestamp::MIN`] to [`Timestamp::MAX`].
    #[must_use]
    pub const fn duration(&self) -> Duration {
        let nanos = self
            .end
            .as_unix_nanos()
            .abs_diff(self.start.as_unix_nanos());

        Duration::from_nanos(nanos)
    }
}

impl RangeBounds<Timestamp> for TimeInterval {
    fn start_bound(&self) -> Bound<&Timestamp> {
        Bound::Included(&self.start)
    }

    fn end_bound(&self) -> Bound<&Timestamp> {
        Bound::Excluded(&self.end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_equal_and_reversed_endpoints_with_context() {
        let epoch = Timestamp::UNIX_EPOCH;
        let next = Timestamp::from_unix_nanos(1);
        let cases = [
            (epoch, epoch),
            (next, epoch),
            (Timestamp::MAX, Timestamp::MIN),
        ];

        for (start, end) in cases {
            let error = TimeInterval::new(start, end).unwrap_err();

            assert_eq!(error, IntervalError { start, end });
        }
    }

    #[test]
    fn accepts_a_single_nanosecond_interval() {
        let start = Timestamp::from_unix_nanos(i64::MAX - 1);
        let end = Timestamp::MAX;
        let interval = TimeInterval::new(start, end).unwrap();

        assert_eq!(interval.start(), start);
        assert_eq!(interval.end(), end);
        assert_eq!(interval.duration(), Duration::from_nanos(1));
    }

    #[test]
    fn membership_includes_start_and_excludes_end() {
        let start = Timestamp::UNIX_EPOCH;
        let end = Timestamp::from_unix_nanos(3);
        let interval = TimeInterval::new(start, end).unwrap();
        let cases = [
            (-1, false),
            (0, true),
            (1, true),
            (2, true),
            (3, false),
            (4, false),
        ];

        for (nanos, expected) in cases {
            let timestamp = Timestamp::from_unix_nanos(nanos);

            assert_eq!(interval.contains(timestamp), expected);
        }
    }

    #[test]
    fn adjacent_intervals_assign_the_shared_endpoint_to_the_later_interval() {
        let start = Timestamp::UNIX_EPOCH;
        let boundary = Timestamp::from_unix_nanos(2);
        let end = Timestamp::from_unix_nanos(4);

        let earlier = TimeInterval::new(start, boundary).unwrap();
        let later = TimeInterval::new(boundary, end).unwrap();

        assert!(!earlier.contains(boundary));
        assert!(later.contains(boundary));
    }

    #[test]
    fn measures_duration_across_epoch_with_nanosecond_precision() {
        let start = Timestamp::from_unix_secs(-1).unwrap();
        let end = Timestamp::from_unix_nanos(2);
        let interval = TimeInterval::new(start, end).unwrap();

        assert_eq!(interval.duration(), Duration::new(1, 2));
    }

    #[test]
    fn supports_the_full_timestamp_span() {
        let interval = TimeInterval::new(Timestamp::MIN, Timestamp::MAX).unwrap();

        // The distance between the signed limits is 2^64 - 1 nanoseconds.
        assert_eq!(interval.duration(), Duration::from_nanos(u64::MAX));
        assert!(interval.contains(Timestamp::MIN));
        assert!(!interval.contains(Timestamp::MAX));
    }

    #[test]
    fn selects_series_observations_with_the_same_half_open_bounds() {
        let timestamps = [-1, 0, 1, 2, 3].map(Timestamp::from_unix_nanos);
        let values = [-10.0, 0.0, 10.0, 20.0, 30.0];

        let series = crate::SeriesView::new(&timestamps, &values).unwrap();
        let interval = TimeInterval::new(timestamps[1], timestamps[4]).unwrap();

        let selected = series.between(interval).unwrap();

        assert_eq!(selected.timestamps(), &timestamps[1..4]);
        assert_eq!(selected.values(), &[0.0, 10.0, 20.0]);
    }

    #[test]
    fn describes_invalid_endpoints_with_units() {
        let error = IntervalError {
            start: Timestamp::from_unix_nanos(1),
            end: Timestamp::from_unix_nanos(-1),
        };

        assert_eq!(
            error.to_string(),
            "interval end (-1 ns) must be later than start (1 ns)",
        );
    }
}
