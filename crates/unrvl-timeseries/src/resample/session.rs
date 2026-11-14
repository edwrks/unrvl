use std::ops::Range;

use thiserror::Error;

use super::BucketLabel;
use crate::{SeriesView, Session, SessionSchedule, TimeInterval, Timestamp};

/// Invalid analysis bounds or an observation outside the selected open periods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SessionAssignmentError {
    /// Some of the analysis interval lies outside known schedule coverage.
    #[error(
        "analysis interval [{start} ns, {end} ns) is outside schedule coverage \
         [{coverage_start} ns, {coverage_end} ns)",
        start = .bounds.start().as_unix_nanos(),
        end = .bounds.end().as_unix_nanos(),
        coverage_start = .coverage.start().as_unix_nanos(),
        coverage_end = .coverage.end().as_unix_nanos(),
    )]
    BoundsOutsideCoverage {
        /// Supplied analysis interval.
        bounds: TimeInterval,

        /// Coverage declared by the supplied schedule.
        coverage: TimeInterval,
    },

    /// An observation lies outside the half-open analysis interval.
    #[error(
        "observation at index {index} ({timestamp_nanos} ns) is outside \
         analysis interval [{start} ns, {end} ns)",
        timestamp_nanos = .timestamp.as_unix_nanos(),
        start = .bounds.start().as_unix_nanos(),
        end = .bounds.end().as_unix_nanos(),
    )]
    ObservationOutsideBounds {
        /// Index into the supplied series view.
        index: usize,

        /// Original observation timestamp.
        timestamp: Timestamp,

        /// Supplied analysis interval.
        bounds: TimeInterval,
    },

    /// An observation falls in a break or other covered, closed time.
    #[error(
        "observation at index {index} ({timestamp_nanos} ns) is outside \
         the schedule's open periods",
        timestamp_nanos = .timestamp.as_unix_nanos(),
    )]
    ObservationDuringClosure {
        /// Index into the supplied series view.
        index: usize,

        /// Original observation timestamp within the analysis interval.
        timestamp: Timestamp,
    },
}

/// A supplied session and its observations within the analysis interval.
///
/// Eligible observations from all of a session's open periods form one bucket.
/// The intervals below describe its envelope, including any enclosed breaks;
/// open periods remain available through `session.intervals()`. Neither a full
/// envelope nor a nonempty sample establishes sampling completeness.
#[derive(Debug, Clone)]
pub struct SessionBucket<'a, 's> {
    /// Index into the original [`SessionSchedule::sessions`] slice.
    pub session_index: usize,

    /// Original session, including its identifier and open periods.
    pub session: &'s Session,

    /// Full half-open envelope from the first opening to the final close.
    pub interval: TimeInterval,

    /// Intersection of the full envelope with the analysis interval.
    pub selected_interval: TimeInterval,

    /// Contributing observation indices relative to the supplied series view.
    pub source_range: Range<usize>,

    /// Original observations, borrowed without copying or revalidation.
    /// Empty sessions have an empty view at their source insertion position.
    pub observations: SeriesView<'a>,
}

impl SessionBucket<'_, '_> {
    /// Returns the first opening or final close chosen by [`BucketLabel`].
    #[must_use]
    pub const fn label(&self, label: BucketLabel) -> Timestamp {
        match label {
            BucketLabel::Start => self.interval.start(),
            BucketLabel::End => self.interval.end(),
        }
    }

    /// Returns whether analysis bounds clip either end of the session envelope.
    /// This describes interval geometry, not sampling completeness.
    #[must_use]
    pub fn is_partial(&self) -> bool {
        self.selected_interval != self.interval
    }
}

/// Observations grouped by the sessions in a supplied schedule.
///
/// Selects every session whose first-open-to-final-close envelope intersects
/// `bounds`, including empty and partial sessions. All observations must lie
/// within both the analysis interval and a scheduled open period. Breaks and
/// other closed time are rejected. This keeps each session's observations in
/// one contiguous, borrowed value slice.
///
/// Open-period membership includes the start and excludes the end. An
/// observation at a final close does not belong to the ending session; at a
/// shared boundary, it belongs to the opening session. The final close can
/// still label the ending session's output. Exact close-labeled input is
/// checked separately by
/// [`SessionCloseCoverage`](crate::cadence::SessionCloseCoverage).
///
/// Borrows the original series and schedule without allocating an index buffer.
/// It does not generate a calendar, fill missing observations, or establish
/// when a calculated result became available.
///
/// Iterators can outlive the assignment while the source data and schedule
/// remain alive.
///
/// # Examples
///
/// ```
/// # use unrvl_timeseries::{SeriesView, Session, SessionSchedule, TimeInterval, Timestamp};
/// # use unrvl_timeseries::resample::{AggregateOutcome, BucketLabel, SessionAssignment};
/// let coverage = TimeInterval::new(Timestamp::UNIX_EPOCH, Timestamp::from_unix_secs(24 * 3600)?)?;
///
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
/// let schedule = SessionSchedule::new(coverage, vec![session])?;
///
/// let timestamps = [
///     Timestamp::from_unix_secs(10 * 3600)?,
///     Timestamp::from_unix_secs(14 * 3600)?,
/// ];
///
/// let prices = SeriesView::new(&timestamps, &[100.0, 102.0])?;
/// let assigned = SessionAssignment::new(prices, &schedule, coverage)?;
/// let bucket = assigned.iter().next().unwrap();
///
/// assert_eq!(bucket.session.id(), "regular");
/// assert_eq!(bucket.observations.values(), &[100.0, 102.0]);
/// assert_eq!(bucket.label(BucketLabel::End), afternoon.end());
///
/// let last = assigned.last(BucketLabel::End).next().unwrap();
/// assert_eq!(last.observation.unwrap().timestamp, timestamps[1]);
/// assert_eq!(last.timestamp, afternoon.end());
///
/// let average = assigned.mean(BucketLabel::End).next().unwrap();
/// assert_eq!(average.outcome, AggregateOutcome::Value(101.0));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct SessionAssignment<'a, 's> {
    series: SeriesView<'a>,
    schedule: &'s SessionSchedule,
    bounds: TimeInterval,
    session_range: Range<usize>,
}

impl<'a, 's> SessionAssignment<'a, 's> {
    /// Borrows observations for strict assignment to the selected sessions.
    ///
    /// Requires `bounds` to lie wholly within schedule coverage. Every input
    /// observation must lie inside `bounds` and a scheduled open period. Empty
    /// input is valid, including when the selected time is entirely closed.
    /// A range inside a session break yields an empty, partial session bucket.
    ///
    /// Validation locates the selected sessions by binary search, then sweeps
    /// the observations and their open periods in chronological order. Uses
    /// `O(log m + n + p)` time and `O(1)` additional memory for `m` schedule
    /// sessions, `n` observations, and `p` selected open periods.
    ///
    /// # Errors
    ///
    /// Checks [`SessionAssignmentError::BoundsOutsideCoverage`] first. Then
    /// reports the first invalid observation in source order, checking
    /// [`SessionAssignmentError::ObservationOutsideBounds`] before
    /// [`SessionAssignmentError::ObservationDuringClosure`] at each index.
    pub fn new(
        series: SeriesView<'a>,
        schedule: &'s SessionSchedule,
        bounds: TimeInterval,
    ) -> Result<Self, SessionAssignmentError> {
        let coverage = schedule.coverage();
        if bounds.start() < coverage.start() || bounds.end() > coverage.end() {
            return Err(SessionAssignmentError::BoundsOutsideCoverage { bounds, coverage });
        }

        let sessions = schedule.sessions();
        let first = sessions.partition_point(|session| session.final_close() <= bounds.start());
        let last = sessions.partition_point(|session| session.first_open() < bounds.end());
        let session_range = first..last;

        let mut periods = sessions[session_range.clone()]
            .iter()
            .flat_map(Session::intervals)
            .peekable();

        for (index, &timestamp) in series.timestamps().iter().enumerate() {
            if !bounds.contains(timestamp) {
                return Err(SessionAssignmentError::ObservationOutsideBounds {
                    index,
                    timestamp,
                    bounds,
                });
            }

            while periods
                .peek()
                .is_some_and(|period| period.end() <= timestamp)
            {
                periods.next();
            }

            if !periods
                .peek()
                .is_some_and(|period| period.contains(timestamp))
            {
                return Err(SessionAssignmentError::ObservationDuringClosure { index, timestamp });
            }
        }

        Ok(Self { series, schedule, bounds, session_range })
    }

    /// Returns the original supplied series without revalidation.
    #[must_use]
    pub const fn series(&self) -> SeriesView<'a> {
        self.series
    }

    /// Returns the original schedule defining session identity and open
    /// periods.
    #[must_use]
    pub const fn schedule(&self) -> &'s SessionSchedule {
        self.schedule
    }

    /// Returns the selected half-open analysis interval.
    #[must_use]
    pub const fn bounds(&self) -> TimeInterval {
        self.bounds
    }

    /// Returns the number of selected sessions, including empty partial ones.
    #[must_use]
    pub fn session_count(&self) -> usize {
        self.session_range.len()
    }

    /// Yields one borrowed bucket per selected session, in chronological order.
    ///
    /// Original schedule indices and view-relative source ranges are preserved.
    /// A session's open periods contribute to the same value slice because
    /// construction rejects all observations during closure. Iteration uses
    /// `O(n + s)` time and `O(1)` additional memory for `n` observations and
    /// `s` selected sessions.
    #[expect(
        clippy::missing_panics_doc,
        reason = "construction selects only session envelopes intersecting the analysis bounds"
    )]
    pub fn iter(&self) -> impl Iterator<Item = SessionBucket<'a, 's>> + use<'a, 's> {
        let sessions = &self.schedule.sessions()[self.session_range.clone()];
        let mut sessions = sessions.iter().enumerate();
        let first_session = self.session_range.start;
        let series = self.series;
        let bounds = self.bounds;
        let mut next = 0;

        std::iter::from_fn(move || {
            let (offset, session) = sessions.next()?;
            let interval = session.envelope();

            let selected_interval = TimeInterval::new(
                interval.start().max(bounds.start()),
                interval.end().min(bounds.end()),
            )
            .expect("selected session envelopes intersect the analysis interval");

            let first = next;

            while series
                .timestamps()
                .get(next)
                .is_some_and(|&timestamp| timestamp < selected_interval.end())
            {
                next += 1;
            }

            let source_range = first..next;
            let observations = series.slice(source_range.clone());

            Some(SessionBucket {
                session_index: first_session + offset,
                session,
                interval,
                selected_interval,
                source_range,
                observations,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{interval, session, timestamp};

    mod construction {
        use super::*;

        #[test]
        fn rejects_uncovered_bounds_before_inspecting_observations() {
            let coverage = interval(0, 10);
            let regular = session("regular", 1, 9);
            let schedule = SessionSchedule::new(coverage, vec![regular]).unwrap();

            let timestamps = [timestamp(0)];
            let series = SeriesView::new(&timestamps, &[100.0]).unwrap();
            let bounds = [interval(-1, 4), interval(6, 11), interval(-1, 11)];

            for bounds in bounds {
                let error = SessionAssignment::new(series, &schedule, bounds).unwrap_err();

                assert_eq!(
                    error,
                    SessionAssignmentError::BoundsOutsideCoverage { bounds, coverage }
                );
            }
        }

        #[test]
        fn rejects_observations_outside_the_analysis_bounds() {
            let coverage = interval(0, 12);
            let regular = session("regular", 0, 12);
            let schedule = SessionSchedule::new(coverage, vec![regular]).unwrap();
            let bounds = interval(2, 8);
            let cases = [([1, 2], 0, 1), ([2, 8], 1, 8)];

            for (nanos, index, outside) in cases {
                let timestamps = nanos.map(timestamp);
                let series = SeriesView::new(&timestamps, &[100.0, 101.0]).unwrap();

                let error = SessionAssignment::new(series, &schedule, bounds).unwrap_err();

                assert_eq!(
                    error,
                    SessionAssignmentError::ObservationOutsideBounds {
                        index,
                        timestamp: timestamp(outside),
                        bounds,
                    },
                );
            }
        }

        #[test]
        fn rejects_breaks_and_other_known_closed_time() {
            let coverage = interval(0, 14);
            let split = Session::new("split", vec![interval(1, 3), interval(5, 7)]).unwrap();
            let later = session("later", 9, 12);
            let sessions = vec![split, later];
            let schedule = SessionSchedule::new(coverage, sessions).unwrap();
            let closed_times = [0, 3, 4, 7, 8, 12, 13];

            for nanos in closed_times {
                let timestamps = [timestamp(nanos)];
                let series = SeriesView::new(&timestamps, &[100.0]).unwrap();

                let error = SessionAssignment::new(series, &schedule, coverage).unwrap_err();

                assert_eq!(
                    error,
                    SessionAssignmentError::ObservationDuringClosure {
                        index: 0,
                        timestamp: timestamp(nanos),
                    },
                );
            }
        }

        #[test]
        fn reports_the_first_invalid_observation_in_source_order() {
            let coverage = interval(0, 12);
            let regular = session("regular", 1, 3);
            let schedule = SessionSchedule::new(coverage, vec![regular]).unwrap();

            let timestamps = [1, 3, 10].map(timestamp);
            let values = [100.0, 101.0, 102.0];
            let series = SeriesView::new(&timestamps, &values).unwrap();

            let error = SessionAssignment::new(series, &schedule, interval(0, 10)).unwrap_err();

            assert_eq!(
                error,
                SessionAssignmentError::ObservationDuringClosure {
                    index: 1,
                    timestamp: timestamp(3)
                },
            );
        }

        #[test]
        fn accepts_empty_input_during_known_closure() {
            let coverage = interval(0, 10);
            let schedule = SessionSchedule::new(coverage, vec![]).unwrap();

            let series = SeriesView::new(&[], &[]).unwrap();

            let assigned = SessionAssignment::new(series, &schedule, coverage).unwrap();

            assert_eq!(assigned.session_count(), 0);
            assert!(assigned.iter().next().is_none());

            let timestamps = [timestamp(5)];
            let observed = SeriesView::new(&timestamps, &[100.0]).unwrap();

            let error = SessionAssignment::new(observed, &schedule, coverage).unwrap_err();

            assert_eq!(
                error,
                SessionAssignmentError::ObservationDuringClosure {
                    index: 0,
                    timestamp: timestamp(5)
                },
            );
        }

        #[test]
        fn error_messages_include_indices_timestamps_and_ranges() {
            let coverage = interval(0, 10);
            let outside = interval(-1, 5);
            let cases = [
                (
                    SessionAssignmentError::BoundsOutsideCoverage { bounds: outside, coverage },
                    "analysis interval [-1 ns, 5 ns) is outside schedule coverage [0 ns, 10 ns)",
                ),
                (
                    SessionAssignmentError::ObservationOutsideBounds {
                        index: 2,
                        timestamp: timestamp(10),
                        bounds: coverage,
                    },
                    "observation at index 2 (10 ns) is outside analysis interval [0 ns, 10 ns)",
                ),
                (
                    SessionAssignmentError::ObservationDuringClosure {
                        index: 1,
                        timestamp: timestamp(4),
                    },
                    "observation at index 1 (4 ns) is outside the schedule's open periods",
                ),
            ];

            for (error, expected) in cases {
                assert_eq!(error.to_string(), expected);
            }
        }
    }

    mod iteration {
        use std::ptr::eq;
        use std::time::Duration;

        use super::*;

        #[test]
        fn groups_open_periods_and_assigns_a_shared_boundary_to_the_later_session() {
            let coverage = interval(0, 14);
            let split = Session::new("split", vec![interval(1, 3), interval(5, 7)]).unwrap();
            let later = session("later", 7, 11);
            let sessions = vec![split, later];
            let schedule = SessionSchedule::new(coverage, sessions).unwrap();

            let timestamps = [1, 2, 5, 6, 7, 10].map(timestamp);
            let values = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0];
            let series = SeriesView::new(&timestamps, &values).unwrap();

            let assigned = SessionAssignment::new(series, &schedule, coverage).unwrap();
            let rows: Vec<_> = assigned.iter().collect();

            let cases = [
                ("split", interval(1, 7), 0..4),
                ("later", interval(7, 11), 4..6),
            ];

            assert_eq!(rows.len(), cases.len());
            for (index, (row, (id, envelope, source_range))) in rows.iter().zip(cases).enumerate() {
                let expected_session = &schedule.sessions()[index];

                assert_eq!(row.session_index, index);
                assert!(eq(row.session, expected_session));
                assert_eq!(row.session.id(), id);
                assert_eq!(row.interval, envelope);
                assert_eq!(row.selected_interval, envelope);
                assert_eq!(row.label(BucketLabel::Start), envelope.start());
                assert_eq!(row.label(BucketLabel::End), envelope.end());
                assert_eq!(row.source_range, source_range);
                assert_eq!(row.observations.values(), &values[row.source_range.clone()]);
                assert!(!row.is_partial());
            }
        }

        #[test]
        fn keeps_empty_sessions_at_their_source_insertion_positions() {
            let coverage = interval(0, 13);
            let first = session("first", 1, 3);
            let second = session("second", 4, 6);
            let third = session("third", 7, 9);
            let fourth = session("fourth", 10, 12);
            let sessions = vec![first, second, third, fourth];
            let schedule = SessionSchedule::new(coverage, sessions).unwrap();

            let timestamps = [4, 5].map(timestamp);
            let series = SeriesView::new(&timestamps, &[0.0, 7.0]).unwrap();

            let assigned = SessionAssignment::new(series, &schedule, coverage).unwrap();
            let ranges_and_counts: Vec<_> = assigned
                .iter()
                .map(|row| (row.source_range, row.observations.len()))
                .collect();

            assert_eq!(assigned.session_count(), 4);
            assert_eq!(
                ranges_and_counts,
                [(0..0, 0), (0..2, 2), (2..2, 0), (2..2, 0)]
            );
        }

        #[test]
        fn skipping_buckets_preserves_their_source_ranges() {
            let coverage = interval(0, 9);
            let first = session("first", 0, 2);
            let second = session("second", 3, 5);
            let third = session("third", 6, 8);
            let sessions = vec![first, second, third];
            let schedule = SessionSchedule::new(coverage, sessions).unwrap();

            let timestamps = [0, 3, 4, 7].map(timestamp);
            let values = [100.0; 4];
            let series = SeriesView::new(&timestamps, &values).unwrap();

            let assigned = SessionAssignment::new(series, &schedule, coverage).unwrap();
            let second = assigned.iter().nth(1).unwrap();
            let last = assigned.iter().last().unwrap();

            assert_eq!(second.session_index, 1);
            assert_eq!(second.source_range, 1..3);
            assert_eq!(last.session_index, 2);
            assert_eq!(last.source_range, 3..4);
        }

        #[test]
        fn clips_analysis_edges_without_changing_full_session_labels() {
            let coverage = interval(0, 16);
            let split = Session::new("split", vec![interval(1, 3), interval(5, 8)]).unwrap();
            let later = session("later", 10, 14);
            let sessions = vec![split, later];
            let schedule = SessionSchedule::new(coverage, sessions).unwrap();
            let bounds = interval(2, 12);
            let timestamps = [2, 5, 7, 10, 11].map(timestamp);
            let values = [100.0; 5];
            let series = SeriesView::new(&timestamps, &values).unwrap();

            let assigned = SessionAssignment::new(series, &schedule, bounds).unwrap();
            let rows: Vec<_> = assigned.iter().collect();

            let cases = [
                (interval(1, 8), interval(2, 8), 0..3),
                (interval(10, 14), interval(10, 12), 3..5),
            ];

            assert_eq!(rows.len(), cases.len());
            for (row, (full, selected, source_range)) in rows.iter().zip(cases) {
                assert_eq!(row.interval, full);
                assert_eq!(row.selected_interval, selected);
                assert_eq!(row.source_range, source_range);
                assert_eq!(row.label(BucketLabel::Start), full.start());
                assert_eq!(row.label(BucketLabel::End), full.end());
                assert!(row.is_partial());
            }
        }

        #[test]
        fn a_range_inside_a_break_yields_an_empty_partial_session() {
            let coverage = interval(0, 10);
            let split = Session::new("split", vec![interval(1, 3), interval(5, 7)]).unwrap();
            let schedule = SessionSchedule::new(coverage, vec![split]).unwrap();

            let series = SeriesView::new(&[], &[]).unwrap();
            let bounds = interval(3, 5);

            let assigned = SessionAssignment::new(series, &schedule, bounds).unwrap();
            let row = assigned.iter().next().unwrap();

            assert_eq!(assigned.session_count(), 1);
            assert_eq!(row.interval, interval(1, 7));
            assert_eq!(row.selected_interval, bounds);
            assert_eq!(row.source_range, 0..0);
            assert!(row.observations.is_empty());
            assert!(row.is_partial());
        }

        #[test]
        fn touching_envelopes_do_not_select_adjacent_sessions() {
            let coverage = interval(0, 10);
            let earlier = session("earlier", 1, 3);
            let later = session("later", 5, 7);
            let sessions = vec![earlier, later];
            let schedule = SessionSchedule::new(coverage, sessions).unwrap();

            let series = SeriesView::new(&[], &[]).unwrap();

            let assigned = SessionAssignment::new(series, &schedule, interval(3, 5)).unwrap();

            assert_eq!(assigned.session_count(), 0);
            assert!(assigned.iter().next().is_none());
        }

        #[test]
        fn borrows_subviews_and_preserves_original_schedule_indices() {
            let coverage = interval(0, 10);
            let first = session("first", 0, 2);
            let second = session("second", 3, 6);
            let third = session("third", 7, 9);
            let sessions = vec![first, second, third];
            let schedule = SessionSchedule::new(coverage, sessions).unwrap();

            let timestamps = [0, 3, 4, 7].map(timestamp);
            let values = [100.0, 101.0, 102.0, 103.0];
            let original = SeriesView::new(&timestamps, &values).unwrap();
            let bounds = interval(3, 6);
            let series = original.between(bounds).unwrap();
            let rows: Vec<_> = {
                let assigned = SessionAssignment::new(series, &schedule, bounds).unwrap();
                let original_schedule = &schedule;

                assert!(eq(assigned.schedule(), original_schedule));
                assert!(eq(assigned.series().values(), series.values()));
                assert_eq!(assigned.bounds(), bounds);
                assigned.iter().collect()
            };

            assert_eq!(rows.len(), 1);
            let row = &rows[0];
            let expected_timestamps = &timestamps[1..3];
            let expected_values = &values[1..3];

            assert_eq!(row.session_index, 1);
            assert_eq!(row.source_range, 0..2);
            assert!(eq(row.observations.timestamps(), expected_timestamps));
            assert!(eq(row.observations.values(), expected_values));
        }

        #[test]
        fn groups_a_session_across_utc_midnight() {
            let hour = 3600;
            let coverage = TimeInterval::new(
                Timestamp::UNIX_EPOCH,
                Timestamp::from_unix_secs(48 * hour).unwrap(),
            )
            .unwrap();
            let opening = Timestamp::from_unix_secs(22 * hour).unwrap();
            let close = Timestamp::from_unix_secs(26 * hour).unwrap();
            let period = TimeInterval::new(opening, close).unwrap();
            let session = Session::new("overnight", vec![period]).unwrap();
            let schedule = SessionSchedule::new(coverage, vec![session]).unwrap();

            let timestamps = [23, 25].map(|hours| Timestamp::from_unix_secs(hours * hour).unwrap());
            let series = SeriesView::new(&timestamps, &[100.0, 102.0]).unwrap();

            let assigned = SessionAssignment::new(series, &schedule, coverage).unwrap();
            let row = assigned.iter().next().unwrap();

            assert_eq!(assigned.session_count(), 1);
            assert_eq!(row.session.id(), "overnight");
            assert_eq!(row.source_range, 0..2);
            assert_eq!(row.label(BucketLabel::Start), opening);
            assert_eq!(row.label(BucketLabel::End), close);
        }

        #[test]
        fn preserves_boundaries_and_value_bits_across_the_full_timestamp_span() {
            let coverage = TimeInterval::new(Timestamp::MIN, Timestamp::MAX).unwrap();
            let session = Session::new("full", vec![coverage]).unwrap();
            let schedule = SessionSchedule::new(coverage, vec![session]).unwrap();
            let last = Timestamp::MAX.checked_sub(Duration::from_nanos(1)).unwrap();
            let timestamps = [Timestamp::MIN, last];
            let series = SeriesView::new(&timestamps, &[-0.0, 100.0]).unwrap();

            let assigned = SessionAssignment::new(series, &schedule, coverage).unwrap();
            let row = assigned.iter().next().unwrap();

            assert_eq!(row.label(BucketLabel::Start), Timestamp::MIN);
            assert_eq!(row.label(BucketLabel::End), Timestamp::MAX);
            assert_eq!(row.source_range, 0..2);
            assert_eq!(row.observations.values()[0].to_bits(), (-0.0_f64).to_bits());
            assert_eq!(row.observations.timestamps(), timestamps);
        }

        #[test]
        fn adjacent_open_periods_accept_their_shared_boundary() {
            let coverage = interval(0, 8);
            let session = Session::new("adjacent", vec![interval(1, 3), interval(3, 5)]).unwrap();
            let schedule = SessionSchedule::new(coverage, vec![session]).unwrap();

            let timestamps = [1, 3, 4].map(timestamp);
            let values = [100.0, 101.0, 102.0];
            let series = SeriesView::new(&timestamps, &values).unwrap();

            let assigned = SessionAssignment::new(series, &schedule, coverage).unwrap();
            let row = assigned.iter().next().unwrap();

            assert_eq!(row.source_range, 0..3);
            assert_eq!(row.observations.timestamps(), timestamps);
        }
    }
}
