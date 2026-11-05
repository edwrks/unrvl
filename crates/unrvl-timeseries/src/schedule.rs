use std::collections::HashMap;
use std::time::Duration;

use thiserror::Error;

use crate::{Session, TimeInterval, Timestamp};

/// Invalid session identifiers, ordering, or schedule coverage.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ScheduleError {
    /// A session identifier has already appeared in this schedule.
    #[error("session {id:?} at index {index} duplicates the identifier at index {previous_index}")]
    DuplicateSessionId {
        /// Repeated session identifier.
        id: String,

        /// Zero-based index of the first session with this identifier.
        previous_index: usize,

        /// Zero-based index of the repeated identifier.
        index: usize,
    },

    /// A session extends beyond the declared schedule coverage.
    #[error(
        "session {id:?} at index {index} has envelope [{start} ns, {end} ns) \
         outside schedule coverage [{coverage_start} ns, {coverage_end} ns)",
        start = .envelope.start().as_unix_nanos(),
        end = .envelope.end().as_unix_nanos(),
        coverage_start = .coverage.start().as_unix_nanos(),
        coverage_end = .coverage.end().as_unix_nanos(),
    )]
    SessionOutsideCoverage {
        /// Identifier of the session outside coverage.
        id: String,

        /// Zero-based index of the session outside coverage.
        index: usize,

        /// Interval from the session's first opening to its final close.
        envelope: TimeInterval,

        /// Supplied schedule coverage.
        coverage: TimeInterval,
    },

    /// A session starts before the preceding session's final close.
    ///
    /// Includes overlaps and sessions supplied out of chronological order.
    #[error(
        "session {id:?} at index {index} starts at {start} ns, \
         before session {previous_id:?} at index {previous_index} ends at {previous_end} ns",
        start = .current.start().as_unix_nanos(),
        previous_end = .previous.end().as_unix_nanos(),
    )]
    InvalidSessionSequence {
        /// Identifier of the preceding session.
        previous_id: String,

        /// Identifier of the offending session.
        id: String,

        /// Zero-based index of the preceding session.
        previous_index: usize,

        /// Zero-based index of the offending session.
        index: usize,

        /// Envelope of the preceding session.
        previous: TimeInterval,

        /// Envelope that must start at or after `previous.end()`.
        current: TimeInterval,
    },
}

/// The schedule's status at a timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleStatus {
    /// The timestamp falls within a session's open period.
    Open,

    /// The timestamp is covered and every session is closed.
    Closed,

    /// The schedule has no information for this timestamp.
    OutsideCoverage,
}

/// Ordered sessions within an explicit, nonempty coverage interval.
///
/// Each session has a unique identifier within this schedule. Its envelope,
/// from the first opening to the final close, must lie within the coverage.
/// Session envelopes are ordered and nonoverlapping, including their breaks;
/// adjacent envelopes are allowed.
///
/// Coverage is the time range for which you supply the complete session list.
/// The intervals stored in each [`Session`] define its open hours. Within
/// coverage, all other times are closed. An empty session list means closed
/// throughout. Outside coverage, the schedule is unknown.
///
/// Construction cannot detect omitted sessions. The schedule describes planned
/// hours.
///
/// # Examples
///
/// ```
/// # use unrvl_timeseries::{ScheduleStatus, Session, SessionSchedule, TimeInterval, Timestamp};
/// let coverage = TimeInterval::new(Timestamp::UNIX_EPOCH, Timestamp::from_unix_secs(24 * 3600)?)?;
/// let opening = TimeInterval::new(
///     Timestamp::from_unix_secs(9 * 3600)?,
///     Timestamp::from_unix_secs(16 * 3600)?,
/// )?;
/// let session = Session::new("regular", vec![opening])?;
/// let schedule = SessionSchedule::new(coverage, vec![session])?;
///
/// assert_eq!(schedule.status_at(opening.start()), ScheduleStatus::Open);
/// assert_eq!(schedule.status_at(opening.end()), ScheduleStatus::Closed);
/// assert_eq!(schedule.status_at(coverage.end()), ScheduleStatus::OutsideCoverage);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct SessionSchedule {
    coverage: TimeInterval,
    sessions: Vec<Session>,
}

impl SessionSchedule {
    /// Takes ownership of sessions within the supplied coverage.
    ///
    /// Requires unique identifiers and chronological, nonoverlapping session
    /// envelopes contained in `coverage`. An envelope includes any breaks
    /// between a session's first opening and final close. Adjacent envelopes
    /// and envelopes touching the coverage endpoints are valid.
    ///
    /// Empty session lists are valid. Sessions and identifiers are preserved
    /// without sorting or merging.
    ///
    /// Validation uses expected `O(n + b)` time and `O(n)` additional memory
    /// for `n` sessions and `b` total identifier bytes.
    ///
    /// # Errors
    ///
    /// Reports the first invalid session in index order. For each session,
    /// checks for a [`ScheduleError::DuplicateSessionId`], then
    /// [`ScheduleError::SessionOutsideCoverage`], then
    /// [`ScheduleError::InvalidSessionSequence`]. Errors include identifiers,
    /// indices, and interval bounds where applicable.
    pub fn new(coverage: TimeInterval, sessions: Vec<Session>) -> Result<Self, ScheduleError> {
        let mut identifiers = HashMap::new();

        for (index, session) in sessions.iter().enumerate() {
            if let Some(previous_index) = identifiers.insert(session.id(), index) {
                return Err(ScheduleError::DuplicateSessionId {
                    id: session.id().to_owned(),
                    previous_index,
                    index,
                });
            }

            let current = envelope(session);
            if current.start() < coverage.start() || current.end() > coverage.end() {
                return Err(ScheduleError::SessionOutsideCoverage {
                    id: session.id().to_owned(),
                    index,
                    envelope: current,
                    coverage,
                });
            }

            if index > 0 {
                let previous_session = &sessions[index - 1];
                let previous = envelope(previous_session);

                if current.start() < previous.end() {
                    return Err(ScheduleError::InvalidSessionSequence {
                        previous_id: previous_session.id().to_owned(),
                        id: session.id().to_owned(),
                        previous_index: index - 1,
                        index,
                        previous,
                        current,
                    });
                }
            }
        }

        Ok(Self { coverage, sessions })
    }

    /// Returns the supplied coverage, including its start and excluding its
    /// end.
    #[must_use]
    pub const fn coverage(&self) -> TimeInterval {
        self.coverage
    }

    /// Returns the original session slice in chronological order.
    #[must_use]
    pub const fn sessions(&self) -> &[Session] {
        self.sessions.as_slice()
    }

    /// Returns the session with an open period containing `timestamp`.
    ///
    /// Returns `None` during breaks, between sessions, and outside coverage.
    /// Use [`Self::status_at`] to distinguish closed time from unknown time.
    #[must_use]
    pub fn session_at(&self, timestamp: Timestamp) -> Option<&Session> {
        // The first session that closes after the timestamp is the only
        // possible match because session envelopes are ordered and disjoint.
        let index = self.completed_session_count(timestamp);

        self.sessions
            .get(index)
            .filter(|session| session.is_open_at(timestamp))
    }

    /// Returns the scheduled status at `timestamp`.
    ///
    /// Coverage and open periods include their start and exclude their end.
    /// Breaks and gaps between sessions are [`ScheduleStatus::Closed`] within
    /// coverage. An uncovered timestamp is [`ScheduleStatus::OutsideCoverage`].
    ///
    /// Uses binary search over sessions and the candidate session's open
    /// periods.
    #[must_use]
    pub fn status_at(&self, timestamp: Timestamp) -> ScheduleStatus {
        if !self.coverage.contains(timestamp) {
            return ScheduleStatus::OutsideCoverage;
        }

        if self.session_at(timestamp).is_some() {
            ScheduleStatus::Open
        } else {
            ScheduleStatus::Closed
        }
    }

    /// Returns the last supplied session whose final close is at or before
    /// `timestamp`.
    ///
    /// The query may lie outside coverage. Returns `None` if no supplied
    /// session has reached its final close by that time.
    #[must_use]
    pub fn previous_session(&self, timestamp: Timestamp) -> Option<&Session> {
        let index = self.completed_session_count(timestamp).checked_sub(1)?;
        self.sessions.get(index)
    }

    /// Returns the first supplied session whose first open is strictly after
    /// `timestamp`.
    ///
    /// The query may lie outside coverage. Returns `None` if no supplied
    /// session starts later.
    #[must_use]
    pub fn next_session(&self, timestamp: Timestamp) -> Option<&Session> {
        let index = self
            .sessions
            .partition_point(|session| first_open(session) <= timestamp);

        self.sessions.get(index)
    }

    /// Returns the total scheduled open time within `range`.
    ///
    /// Returns `Some(Duration::ZERO)` when a covered range has no open time.
    /// Returns `None` if any part of the range lies outside coverage.
    #[must_use]
    pub fn open_duration(&self, range: TimeInterval) -> Option<Duration> {
        let start = range.start();
        let end = range.end();
        if start < self.coverage.start() || end > self.coverage.end() {
            return None;
        }

        let first = self.completed_session_count(start);
        let mut total = Duration::ZERO;

        // Open periods are ordered and disjoint across the schedule.
        for period in self.sessions[first..].iter().flat_map(Session::intervals) {
            if period.start() >= end {
                break;
            }

            let overlap_start = start.max(period.start());
            let overlap_end = end.min(period.end());
            if let Some(overlap) = overlap_end.checked_duration_since(overlap_start) {
                total += overlap;
            }
        }

        Some(total)
    }

    /// Yields each session and its final close in chronological order.
    ///
    /// Use these timestamps to look up expected closing observations. Each
    /// timestamp labels the session that ends there, even if it also marks the
    /// next session's first open or the end of coverage.
    pub fn close_labels(&self) -> impl Iterator<Item = (&Session, Timestamp)> + '_ {
        self.sessions
            .iter()
            .map(|session| (session, final_close(session)))
    }

    fn completed_session_count(&self, timestamp: Timestamp) -> usize {
        self.sessions
            .partition_point(|session| final_close(session) <= timestamp)
    }
}

fn envelope(session: &Session) -> TimeInterval {
    TimeInterval::new(first_open(session), final_close(session))
        .expect("session construction guarantees a nonempty, ordered envelope")
}

const fn first_open(session: &Session) -> Timestamp {
    session
        .intervals()
        .first()
        .expect("a session has at least one open period")
        .start()
}

const fn final_close(session: &Session) -> Timestamp {
    session
        .intervals()
        .last()
        .expect("a session has at least one open period")
        .end()
}

#[cfg(test)]
mod tests;
