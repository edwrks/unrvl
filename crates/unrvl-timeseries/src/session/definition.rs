use thiserror::Error;

use crate::{TimeInterval, Timestamp};

/// Invalid session identity or open periods.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SessionError {
    /// No identifier was supplied.
    #[error("session identifier must not be empty")]
    EmptyId,

    /// No open periods were supplied.
    #[error("session {id:?} must contain at least one open period")]
    EmptyIntervals {
        /// Supplied session identifier.
        id: String,
    },

    /// An open period starts before its predecessor ends.
    ///
    /// Includes both overlaps and intervals supplied out of chronological
    /// order.
    #[error(
        "session {id:?} open period {index} starts at {current_start} ns, \
         before period {previous_index} ends at {previous_end} ns",
        current_start = .current.start().as_unix_nanos(),
        previous_end = .previous.end().as_unix_nanos(),
    )]
    InvalidIntervalSequence {
        /// Supplied session identifier.
        id: String,

        /// Zero-based index of the preceding interval.
        previous_index: usize,

        /// Zero-based index of the offending interval.
        index: usize,

        /// Preceding open period.
        previous: TimeInterval,

        /// Open period that must start at or after `previous.end()`.
        current: TimeInterval,
    },
}

/// A named session with one or more scheduled open periods.
///
/// Periods are ordered and nonoverlapping. Gaps between periods represent
/// breaks. Adjacent periods are allowed, and periods may cross UTC midnight.
/// Each period includes its start and excludes its end, following
/// [`TimeInterval`].
///
/// Identifiers are nonempty, caller-supplied strings, preserved exactly.
///
/// A [`SessionSchedule`](crate::SessionSchedule) groups sessions within a
/// declared coverage interval. Each session's envelope spans its first opening
/// to its final close. [`SessionSchedule::new`](crate::SessionSchedule::new)
/// checks that identifiers are unique within the schedule and that these
/// envelopes are ordered, nonoverlapping, and contained in the coverage.
///
/// Access is immutable to preserve the interval guarantees. Cloning a session
/// copies its identifier and interval vector.
///
/// # Examples
///
/// ```
/// # use unrvl_timeseries::{Session, TimeInterval, Timestamp};
/// let morning = TimeInterval::new(
///     Timestamp::from_unix_secs(9 * 3600)?,
///     Timestamp::from_unix_secs(12 * 3600)?,
/// )?;
///
/// let afternoon = TimeInterval::new(
///     Timestamp::from_unix_secs(13 * 3600)?,
///     Timestamp::from_unix_secs(16 * 3600)?,
/// )?;
///
/// let session = Session::new("regular", vec![morning, afternoon])?;
///
/// assert!(session.is_open_at(Timestamp::from_unix_secs(10 * 3600)?));
/// assert!(!session.is_open_at(Timestamp::from_unix_secs(12 * 3600)?));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct Session {
    id: String,
    intervals: Vec<TimeInterval>,
}

impl Session {
    /// Takes ownership of a nonempty identifier and its open periods.
    ///
    /// Requires at least one period, with each subsequent period starting
    /// at or after its predecessor ends. Adjacent periods are valid. Input
    /// is preserved without sorting, merging, or identifier normalization.
    ///
    /// Validation uses `O(n)` time and `O(1)` additional memory for `n`
    /// periods.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError::EmptyId`] for an empty identifier or
    /// [`SessionError::EmptyIntervals`] if there are no open periods.
    ///
    /// Reports the first invalid neighboring pair with
    /// [`SessionError::InvalidIntervalSequence`], including its indices,
    /// intervals, and session identifier.
    pub fn new(id: impl Into<String>, intervals: Vec<TimeInterval>) -> Result<Self, SessionError> {
        let id = id.into();
        if id.is_empty() {
            return Err(SessionError::EmptyId);
        }

        if intervals.is_empty() {
            return Err(SessionError::EmptyIntervals { id });
        }

        for (previous_index, pair) in intervals.windows(2).enumerate() {
            let previous = pair[0];
            let current = pair[1];

            if current.start() < previous.end() {
                return Err(SessionError::InvalidIntervalSequence {
                    id,
                    previous_index,
                    index: previous_index + 1,
                    previous,
                    current,
                });
            }
        }

        Ok(Self { id, intervals })
    }

    /// Returns the original session identifier.
    #[must_use]
    pub const fn id(&self) -> &str {
        self.id.as_str()
    }

    /// Returns the original, nonempty slice of open periods in time order.
    #[must_use]
    pub const fn intervals(&self) -> &[TimeInterval] {
        self.intervals.as_slice()
    }

    /// Returns the start of the first scheduled open period.
    #[must_use]
    #[expect(
        clippy::missing_panics_doc,
        reason = "session construction guarantees a nonempty interval vector"
    )]
    pub const fn first_open(&self) -> Timestamp {
        self.intervals()
            .first()
            .expect("a session has at least one open period")
            .start()
    }

    /// Returns the end of the last scheduled open period.
    ///
    /// This session is closed at that timestamp because open periods exclude
    /// their end.
    #[must_use]
    #[expect(
        clippy::missing_panics_doc,
        reason = "session construction guarantees a nonempty interval vector"
    )]
    pub const fn final_close(&self) -> Timestamp {
        self.intervals()
            .last()
            .expect("a session has at least one open period")
            .end()
    }

    /// Returns whether `timestamp` falls within one of this session's open
    /// periods.
    ///
    /// Returns false during breaks and outside the open periods.
    ///
    /// Uses `O(log n)` time and `O(1)` additional memory for `n` intervals.
    #[must_use]
    pub fn is_open_at(&self, timestamp: Timestamp) -> bool {
        let index = self
            .intervals
            .partition_point(|interval| interval.end() <= timestamp);

        self.intervals
            .get(index)
            .is_some_and(|interval| interval.contains(timestamp))
    }

    /// Returns the span from the first open to the final close, including any
    /// breaks.
    pub(crate) fn envelope(&self) -> TimeInterval {
        TimeInterval::new(self.first_open(), self.final_close())
            .expect("session construction guarantees a nonempty, ordered envelope")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn interval(start_nanos: i64, end_nanos: i64) -> TimeInterval {
        TimeInterval::new(
            Timestamp::from_unix_nanos(start_nanos),
            Timestamp::from_unix_nanos(end_nanos),
        )
        .unwrap()
    }

    #[test]
    fn rejects_an_empty_identifier() {
        let error = Session::new("", vec![interval(0, 1)]).unwrap_err();

        assert_eq!(error, SessionError::EmptyId);
    }

    #[test]
    fn rejects_empty_intervals_with_the_session_identifier() {
        let error = Session::new("regular", vec![]).unwrap_err();

        assert_eq!(error, SessionError::EmptyIntervals { id: "regular".into() });
    }

    #[test]
    fn rejects_overlapping_unordered_and_duplicate_intervals() {
        let cases = [
            ("overlap", [interval(0, 4), interval(3, 6)]),
            ("out of order", [interval(4, 6), interval(0, 3)]),
            ("duplicate", [interval(0, 3), interval(0, 3)]),
        ];

        for (id, intervals) in cases {
            let error = Session::new(id, intervals.to_vec()).unwrap_err();

            assert_eq!(
                error,
                SessionError::InvalidIntervalSequence {
                    id: id.into(),
                    previous_index: 0,
                    index: 1,
                    previous: intervals[0],
                    current: intervals[1],
                },
            );
        }
    }

    #[test]
    fn reports_the_first_invalid_pair_after_valid_intervals() {
        let intervals = [
            interval(0, 2),
            interval(3, 5),
            interval(4, 6),
            interval(4, 7),
        ];
        let error = Session::new("regular", intervals.to_vec()).unwrap_err();

        assert_eq!(
            error,
            SessionError::InvalidIntervalSequence {
                id: "regular".into(),
                previous_index: 1,
                index: 2,
                previous: intervals[1],
                current: intervals[2],
            },
        );
    }

    #[test]
    fn preserves_the_owned_identifier_and_interval_buffers() {
        let id = String::from("  venue:overnight  ");

        let openings = [interval(-2, 0), interval(1, 3)];
        let intervals = openings.to_vec();

        let id_ptr = id.as_ptr();
        let intervals_ptr = intervals.as_ptr();

        let session = Session::new(id, intervals).unwrap();

        assert_eq!(session.id(), "  venue:overnight  ");
        assert_eq!(session.id().as_ptr(), id_ptr);
        assert_eq!(session.intervals(), openings);
        assert_eq!(session.intervals().as_ptr(), intervals_ptr);
    }

    #[test]
    fn bounds_span_the_open_periods_and_include_breaks() {
        let session = Session::new("regular", vec![interval(1, 3), interval(5, 8)]).unwrap();

        assert_eq!(session.first_open(), Timestamp::from_unix_nanos(1));
        assert_eq!(session.final_close(), Timestamp::from_unix_nanos(8));
        assert_eq!(session.envelope(), interval(1, 8));
    }

    #[test]
    fn membership_respects_openings_breaks_and_the_final_close() {
        let session = Session::new("regular", vec![interval(0, 3), interval(5, 8)]).unwrap();
        let cases = [
            (-1, false),
            (0, true),
            (2, true),
            (3, false),
            (4, false),
            (5, true),
            (7, true),
            (8, false),
            (9, false),
        ];

        for (nanos, expected) in cases {
            let timestamp = Timestamp::from_unix_nanos(nanos);

            assert_eq!(
                session.is_open_at(timestamp),
                expected,
                "timestamp={nanos} ns"
            );
        }
    }

    #[test]
    fn accepts_adjacent_opening_intervals() {
        let session = Session::new("continuous", vec![interval(0, 3), interval(3, 6)]).unwrap();

        assert_eq!(session.intervals().len(), 2);
        assert!(session.is_open_at(Timestamp::from_unix_nanos(3)));
    }

    #[test]
    fn supports_a_single_opening_across_utc_midnight() {
        // The Unix epoch is midnight UTC, between these opening and closing
        // times.
        let open = Timestamp::from_unix_secs(-3600).unwrap();
        let close = Timestamp::from_unix_secs(3600).unwrap();
        let opening = TimeInterval::new(open, close).unwrap();
        let session = Session::new("overnight", vec![opening]).unwrap();

        assert_eq!(session.intervals(), [opening]);
        assert!(session.is_open_at(open));
        assert!(session.is_open_at(Timestamp::UNIX_EPOCH));
        assert!(!session.is_open_at(close));
    }

    #[test]
    fn supports_membership_across_the_full_timestamp_span() {
        let opening = TimeInterval::new(Timestamp::MIN, Timestamp::MAX).unwrap();
        let session = Session::new("full span", vec![opening]).unwrap();

        assert!(session.is_open_at(Timestamp::MIN));
        assert!(session.is_open_at(Timestamp::UNIX_EPOCH));
        assert!(!session.is_open_at(Timestamp::MAX));
    }

    #[test]
    fn describes_invalid_sequences_with_the_session_and_units() {
        let error = Session::new("regular", vec![interval(0, 3), interval(2, 5)]).unwrap_err();

        assert_eq!(
            error.to_string(),
            "session \"regular\" open period 1 starts at 2 ns, before period 0 ends at 3 ns",
        );
    }
}
