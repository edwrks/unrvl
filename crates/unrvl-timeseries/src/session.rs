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
}

#[cfg(test)]
mod tests;
