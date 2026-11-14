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
///
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
    /// [`ScheduleError::InvalidSessionSequence`].
    ///
    /// Errors include identifiers, indices, and interval bounds where
    /// applicable.
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

            let current = session.envelope();
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
                let previous = previous_session.envelope();

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
            .partition_point(|session| session.first_open() <= timestamp);

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
            .map(|session| (session, session.final_close()))
    }

    /// Counts sessions whose final close is at or before `timestamp`.
    /// The count is the index of the first session closing later, or the end of
    /// the session slice if none remains.
    fn completed_session_count(&self, timestamp: Timestamp) -> usize {
        self.sessions
            .partition_point(|session| session.final_close() <= timestamp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{interval, session};

    mod construction {
        use super::*;

        #[test]
        fn rejects_duplicate_identifiers_with_the_first_occurrence() {
            let coverage = interval(0, 10);
            let regular = session("regular", 0, 2);
            let overnight = session("overnight", 3, 5);
            let duplicate_regular = session("regular", 6, 8);
            let sessions = vec![regular, overnight, duplicate_regular];
            let error = SessionSchedule::new(coverage, sessions).unwrap_err();

            assert_eq!(
                error,
                ScheduleError::DuplicateSessionId {
                    id: "regular".into(),
                    previous_index: 0,
                    index: 2
                },
            );
        }

        #[test]
        fn rejects_sessions_extending_beyond_either_coverage_endpoint() {
            let coverage = interval(0, 10);
            let cases = [
                ("starts before", interval(-1, 1)),
                ("ends after", interval(9, 11)),
                ("encloses coverage", interval(-1, 11)),
            ];

            for (id, opening) in cases {
                let session = Session::new(id, vec![opening]).unwrap();
                let error = SessionSchedule::new(coverage, vec![session]).unwrap_err();

                assert_eq!(
                    error,
                    ScheduleError::SessionOutsideCoverage {
                        id: id.into(),
                        index: 0,
                        envelope: opening,
                        coverage,
                    },
                );
            }
        }

        #[test]
        fn rejects_overlapping_unordered_and_interleaved_session_envelopes() {
            let coverage = interval(0, 10);
            let overlapping_first = session("first", 0, 4);
            let overlapping_second = session("second", 3, 6);
            let unordered_first = session("first", 4, 6);
            let unordered_second = session("second", 0, 3);

            // The second session fits inside the first session's break, but
            // their envelopes overlap. Breaks cannot be assigned to
            // another session.
            let with_break = Session::new("first", vec![interval(0, 2), interval(6, 8)]).unwrap();
            let during_break = session("second", 3, 5);

            let cases = [
                (overlapping_first, overlapping_second, interval(0, 4)),
                (unordered_first, unordered_second, interval(4, 6)),
                (with_break, during_break, interval(0, 8)),
            ];

            for (first, second, previous) in cases {
                let current = second.intervals()[0];
                let error = SessionSchedule::new(coverage, vec![first, second]).unwrap_err();

                assert_eq!(
                    error,
                    ScheduleError::InvalidSessionSequence {
                        previous_id: "first".into(),
                        id: "second".into(),
                        previous_index: 0,
                        index: 1,
                        previous,
                        current,
                    },
                );
            }
        }

        #[test]
        fn reports_the_first_invalid_session_before_later_failures() {
            let coverage = interval(0, 10);
            let first = session("first", 0, 2);
            let second = session("second", 3, 5);
            let overlap = session("overlap", 4, 6);
            let duplicate_first = session("first", 8, 12);
            let sessions = vec![first, second, overlap, duplicate_first];
            let error = SessionSchedule::new(coverage, sessions).unwrap_err();

            assert!(matches!(
                error,
                ScheduleError::InvalidSessionSequence { previous_index: 1, index: 2, .. }
            ));
        }

        #[test]
        fn accepts_coverage_without_sessions() {
            let coverage = interval(0, 10);
            let schedule = SessionSchedule::new(coverage, vec![]).unwrap();

            assert_eq!(schedule.coverage(), coverage);
            assert!(schedule.sessions().is_empty());
        }

        #[test]
        fn preserves_the_supplied_sessions_and_their_buffer() {
            let coverage = interval(0, 10);
            let with_break = Session::new("first", vec![interval(1, 2), interval(4, 6)]).unwrap();
            let second = session("second", 7, 9);
            let sessions = vec![with_break, second];
            let sessions_ptr = sessions.as_ptr();
            let schedule = SessionSchedule::new(coverage, sessions).unwrap();

            assert_eq!(schedule.coverage(), coverage);
            assert_eq!(schedule.sessions().as_ptr(), sessions_ptr);
            assert_eq!(schedule.sessions()[0].id(), "first");
            assert_eq!(schedule.sessions()[1].id(), "second");
            assert_eq!(
                schedule.sessions()[0].intervals(),
                [interval(1, 2), interval(4, 6)]
            );
        }

        #[test]
        fn accepts_adjacent_envelopes_at_the_coverage_boundaries() {
            let coverage = interval(0, 10);
            let first = session("first", 0, 5);
            let second = session("second", 5, 10);
            let sessions = vec![first, second];
            let schedule = SessionSchedule::new(coverage, sessions).unwrap();

            assert_eq!(schedule.sessions().len(), 2);
        }

        #[test]
        fn compares_opaque_identifiers_exactly() {
            let coverage = interval(0, 3);
            let venue = session("venue", 0, 1);
            let capitalized = session("Venue", 1, 2);
            let padded = session(" venue ", 2, 3);
            let sessions = vec![venue, capitalized, padded];
            let schedule = SessionSchedule::new(coverage, sessions).unwrap();
            let identifiers: Vec<_> = schedule.sessions().iter().map(Session::id).collect();

            assert_eq!(identifiers, ["venue", "Venue", " venue "]);
        }

        #[test]
        fn supports_overnight_sessions_and_the_full_timestamp_span() {
            let coverage = TimeInterval::new(Timestamp::MIN, Timestamp::MAX).unwrap();
            let before = session("before", i64::MIN, -1);
            let across_midnight = session("across midnight", -1, 1);
            let after = session("after", 1, i64::MAX);
            let sessions = vec![before, across_midnight, after];
            let schedule = SessionSchedule::new(coverage, sessions).unwrap();

            assert_eq!(schedule.coverage(), coverage);
            assert_eq!(schedule.sessions().len(), 3);
            assert_eq!(
                schedule.open_duration(coverage),
                Some(Duration::from_nanos(u64::MAX))
            );
        }

        #[test]
        fn describes_invalid_sequences_with_both_identifiers_and_units() {
            let coverage = interval(0, 10);
            let first = session("first", 0, 3);
            let second = session("second", 2, 5);
            let sessions = vec![first, second];
            let error = SessionSchedule::new(coverage, sessions).unwrap_err();

            assert_eq!(
                error.to_string(),
                "session \"second\" at index 1 starts at 2 ns, \
                 before session \"first\" at index 0 ends at 3 ns",
            );
        }
    }

    mod lookup {
        use super::*;

        #[test]
        fn finds_the_open_session_across_a_break_and_shared_boundary() {
            let coverage = interval(0, 10);
            let split = Session::new("split", vec![interval(1, 3), interval(4, 6)]).unwrap();
            let later = session("later", 6, 9);
            let sessions = vec![split, later];
            let schedule = SessionSchedule::new(coverage, sessions).unwrap();

            let cases = [
                (-1, None),
                (0, None),
                (1, Some("split")),
                (3, None),
                (4, Some("split")),
                (6, Some("later")),
                (9, None),
                (10, None),
            ];

            for (nanos, expected) in cases {
                let session = schedule.session_at(Timestamp::from_unix_nanos(nanos));
                assert_eq!(session.map(Session::id), expected, "{nanos} ns");
            }
        }

        #[test]
        fn distinguishes_open_periods_breaks_gaps_and_uncovered_time() {
            let coverage = interval(0, 10);
            let split = Session::new("split", vec![interval(1, 3), interval(4, 6)]).unwrap();
            let later = session("later", 7, 9);
            let sessions = vec![split, later];
            let schedule = SessionSchedule::new(coverage, sessions).unwrap();

            let cases = [
                (-1, ScheduleStatus::OutsideCoverage),
                (0, ScheduleStatus::Closed),
                (1, ScheduleStatus::Open),
                (3, ScheduleStatus::Closed),
                (4, ScheduleStatus::Open),
                (6, ScheduleStatus::Closed),
                (7, ScheduleStatus::Open),
                (9, ScheduleStatus::Closed),
                (10, ScheduleStatus::OutsideCoverage),
            ];

            for (nanos, expected) in cases {
                assert_eq!(
                    schedule.status_at(Timestamp::from_unix_nanos(nanos)),
                    expected,
                    "{nanos} ns"
                );
            }
        }

        #[test]
        fn stays_open_at_the_boundary_between_adjacent_sessions() {
            let coverage = interval(0, 10);
            let first = session("first", 0, 5);
            let second = session("second", 5, 10);
            let sessions = vec![first, second];
            let schedule = SessionSchedule::new(coverage, sessions).unwrap();

            assert_eq!(
                schedule.status_at(Timestamp::from_unix_nanos(5)),
                ScheduleStatus::Open
            );
            assert_eq!(
                schedule.status_at(Timestamp::from_unix_nanos(10)),
                ScheduleStatus::OutsideCoverage
            );
        }

        #[test]
        fn an_empty_schedule_is_closed_within_coverage() {
            let coverage = interval(0, 10);
            let schedule = SessionSchedule::new(coverage, vec![]).unwrap();

            assert_eq!(
                schedule.status_at(Timestamp::UNIX_EPOCH),
                ScheduleStatus::Closed
            );
        }
    }

    mod navigation {
        use super::*;

        #[test]
        fn navigates_sessions_by_first_open_and_final_close() {
            let coverage = interval(0, 12);
            let split = Session::new("split", vec![interval(3, 5), interval(6, 8)]).unwrap();
            let first = session("first", 1, 3);
            let last = session("last", 9, 11);
            let sessions = vec![first, split, last];
            let schedule = SessionSchedule::new(coverage, sessions).unwrap();

            let cases = [
                (-1, None, Some("first")),
                (1, None, Some("split")),
                (3, Some("first"), Some("last")),
                (5, Some("first"), Some("last")),
                (6, Some("first"), Some("last")),
                (8, Some("split"), Some("last")),
                (9, Some("split"), None),
                (11, Some("last"), None),
                (12, Some("last"), None),
            ];

            for (nanos, expected_previous, expected_next) in cases {
                let timestamp = Timestamp::from_unix_nanos(nanos);
                let previous = schedule.previous_session(timestamp).map(Session::id);
                let next = schedule.next_session(timestamp).map(Session::id);

                assert_eq!(previous, expected_previous, "previous at {nanos} ns");
                assert_eq!(next, expected_next, "next at {nanos} ns");
            }
        }

        #[test]
        fn an_empty_schedule_has_no_previous_or_next_session() {
            let coverage = interval(0, 10);
            let schedule = SessionSchedule::new(coverage, vec![]).unwrap();

            assert!(schedule.previous_session(Timestamp::UNIX_EPOCH).is_none());
            assert!(schedule.next_session(Timestamp::UNIX_EPOCH).is_none());
        }
    }

    mod open_duration {
        use super::*;

        #[test]
        fn sums_open_time_within_covered_ranges() {
            let coverage = interval(-2, 15);
            let split = Session::new("split", vec![interval(-1, 2), interval(4, 7)]).unwrap();
            let adjacent = session("adjacent", 7, 10);
            let last = session("last", 12, 14);
            let sessions = vec![split, adjacent, last];
            let schedule = SessionSchedule::new(coverage, sessions).unwrap();

            let cases = [
                (coverage, 11),
                (interval(1, 8), 5),
                (interval(2, 4), 0),
                (interval(7, 9), 2),
                (interval(10, 12), 0),
                (interval(14, 15), 0),
            ];

            for (range, expected_nanos) in cases {
                assert_eq!(
                    schedule.open_duration(range),
                    Some(Duration::from_nanos(expected_nanos)),
                    "range {range:?}"
                );
            }
        }

        #[test]
        fn open_duration_requires_the_whole_range_to_be_covered() {
            let coverage = interval(0, 10);
            let regular = session("regular", 1, 9);
            let sessions = vec![regular];
            let schedule = SessionSchedule::new(coverage, sessions).unwrap();

            for range in [interval(-1, 5), interval(5, 11), interval(10, 12)] {
                assert_eq!(schedule.open_duration(range), None, "range {range:?}");
            }
        }

        #[test]
        fn open_duration_is_zero_for_a_covered_range_without_sessions() {
            let coverage = interval(0, 10);
            let schedule = SessionSchedule::new(coverage, vec![]).unwrap();

            assert_eq!(schedule.open_duration(coverage), Some(Duration::ZERO));
        }
    }

    #[test]
    fn labels_final_closes_at_shared_boundaries_and_the_end_of_coverage() {
        let coverage = interval(0, 10);
        let split = Session::new("split", vec![interval(1, 3), interval(4, 6)]).unwrap();
        let later = session("later", 6, 10);
        let sessions = vec![split, later];
        let schedule = SessionSchedule::new(coverage, sessions).unwrap();

        let labels: Vec<_> = schedule
            .close_labels()
            .map(|(session, close)| (session.id(), close.as_unix_nanos()))
            .collect();

        assert_eq!(labels, [("split", 6), ("later", 10)]);
    }
}
