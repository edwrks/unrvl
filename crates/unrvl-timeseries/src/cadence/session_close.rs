use std::ops::{Bound, Range, RangeBounds};

use thiserror::Error;

use crate::{SeriesView, Session, SessionSchedule, TimeInterval, Timestamp};

/// Invalid session-close analysis bounds or an unexpected observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SessionCloseError {
    /// A bounded analysis range starts after it ends.
    #[error(
        "range start ({start_nanos} ns) must not be later than end ({end_nanos} ns)",
        start_nanos = .start.as_unix_nanos(),
        end_nanos = .end.as_unix_nanos(),
    )]
    ReversedRange {
        /// Supplied lower endpoint, whether inclusive or exclusive.
        start: Timestamp,

        /// Supplied upper endpoint, whether inclusive or exclusive.
        end: Timestamp,
    },

    /// An analysis endpoint lies outside the permitted close-label range.
    #[error(
        "session-close analysis bounds extend beyond the permitted close-label range \
         [{start_nanos} ns, {end_nanos} ns]",
        start_nanos = .coverage.start().as_unix_nanos(),
        end_nanos = .coverage.end().as_unix_nanos(),
    )]
    BoundsOutsideCoverage {
        /// Supplied bounds, including any unbounded endpoint.
        bounds: (Bound<Timestamp>, Bound<Timestamp>),

        /// Schedule coverage; its end is a valid close label.
        coverage: TimeInterval,
    },

    /// A supplied observation is outside the selected close-label bounds.
    #[error(
        "observation at index {index} ({timestamp_nanos} ns) is outside \
         the session-close analysis bounds",
        timestamp_nanos = .timestamp.as_unix_nanos(),
    )]
    OutsideBounds {
        /// Index into the supplied series view.
        index: usize,

        /// Original observation timestamp.
        timestamp: Timestamp,

        /// Analysis bounds, with unbounded endpoints resolved to coverage.
        bounds: (Bound<Timestamp>, Bound<Timestamp>),
    },

    /// An observation does not exactly match any selected final session close.
    #[error(
        "observation at index {index} ({timestamp_nanos} ns) does not match \
         a final session close",
        timestamp_nanos = .timestamp.as_unix_nanos(),
    )]
    NotSessionClose {
        /// Index into the supplied series view.
        index: usize,

        /// Original observation timestamp.
        timestamp: Timestamp,
    },
}

/// A scheduled final close and its position in the supplied schedule.
#[derive(Debug, Clone, Copy)]
pub struct SessionClose<'s> {
    /// Index into [`SessionSchedule::sessions`].
    pub session_index: usize,

    /// Session ending at this close, including its caller-supplied identifier.
    pub session: &'s Session,

    /// Final close timestamp, including a close at the end of coverage.
    pub timestamp: Timestamp,
}

/// A supplied observation at its session's exact final close.
#[derive(Debug, Clone, Copy)]
pub struct SessionCloseObservation<'s> {
    /// Index into the supplied series view.
    pub index: usize,

    /// Original observation timestamp, equal to the final close.
    pub timestamp: Timestamp,

    /// Original observation value.
    pub value: f64,

    /// Index into [`SessionSchedule::sessions`].
    pub session_index: usize,

    /// Session ending at this observation's timestamp.
    pub session: &'s Session,
}

/// Checked coverage for one exact closing observation per selected session.
///
/// Borrows the original series and schedule. Missing closes are successful
/// coverage outcomes, reported as counts and consecutive ranges of schedule
/// indices. [`Self::missing`] yields their sessions and close timestamps.
///
/// Bounds select final close labels, independently of open-period membership.
/// A close at a shared session boundary labels the session ending there. A
/// final close at the end of schedule coverage is also a valid label.
#[derive(Debug, Clone)]
pub struct SessionCloseCoverage<'a, 's> {
    series: SeriesView<'a>,
    schedule: &'s SessionSchedule,
    bounds: (Bound<Timestamp>, Bound<Timestamp>),
    expected: Range<usize>,
    session_indices: Vec<usize>,
    gaps: Vec<Range<usize>>,
}

impl<'a, 's> SessionCloseCoverage<'a, 's> {
    /// Checks all supplied observations against selected final session closes.
    ///
    /// Bounds may be inclusive, exclusive, or unbounded. Unbounded endpoints
    /// resolve to the corresponding schedule coverage endpoint, inclusive.
    /// Explicit endpoints must lie between those endpoints, including the
    /// coverage end. This prevents unknown periods from being reported as
    /// complete. [`Self::bounds`] returns the resolved bounds.
    ///
    /// Every observation must exactly match a selected final close. No values
    /// are selected, carried forward, filled, or retimestamped. Select an input
    /// subview first if necessary. Empty series and ranges with no expected
    /// closes are valid; the latter require an empty series.
    ///
    /// Uses `O(log s + n + e)` time and `O(n + g)` additional memory for `s`
    /// supplied sessions, `n` observations, `e` selected sessions, and `g`
    /// missing-session ranges.
    ///
    /// # Errors
    ///
    /// Checks reversed bounded endpoints first, then endpoints outside schedule
    /// coverage. Otherwise reports the first invalid observation in input
    /// order, checking analysis bounds before exact close matching.
    ///
    /// # Examples
    ///
    /// ```
    /// # use unrvl_timeseries::{SeriesView, Session, SessionSchedule, TimeInterval, Timestamp};
    /// # use unrvl_timeseries::cadence::SessionCloseCoverage;
    /// let at = |hour: i64| Timestamp::from_unix_secs(hour * 3600);
    ///
    /// let first = Session::new("first", vec![TimeInterval::new(at(9)?, at(16)?)?])?;
    /// let second = Session::new("second", vec![TimeInterval::new(at(33)?, at(40)?)?])?;
    ///
    /// let schedule = SessionSchedule::new(TimeInterval::new(at(0)?, at(48)?)?, vec![first, second])?;
    ///
    /// let timestamps = [at(16)?];
    /// let prices = SeriesView::new(&timestamps, &[100.0])?;
    /// let coverage = SessionCloseCoverage::new(prices, &schedule, ..)?;
    ///
    /// assert_eq!(coverage.expected_count(), 2);
    /// assert_eq!(coverage.missing_count(), 1);
    /// assert_eq!(coverage.missing().next().unwrap().session.id(), "second");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(
        series: SeriesView<'a>,
        schedule: &'s SessionSchedule,
        bounds: impl RangeBounds<Timestamp>,
    ) -> Result<Self, SessionCloseError> {
        let bounds = (bounds.start_bound().cloned(), bounds.end_bound().cloned());

        if let (
            Bound::Included(start) | Bound::Excluded(start),
            Bound::Included(end) | Bound::Excluded(end),
        ) = bounds
            && start > end
        {
            return Err(SessionCloseError::ReversedRange { start, end });
        }

        let coverage = schedule.coverage();
        for bound in <[_; 2]>::from(bounds) {
            if let Bound::Included(timestamp) | Bound::Excluded(timestamp) = bound
                && (timestamp < coverage.start() || timestamp > coverage.end())
            {
                return Err(SessionCloseError::BoundsOutsideCoverage { bounds, coverage });
            }
        }

        let bounds = (
            match bounds.0 {
                Bound::Unbounded => Bound::Included(coverage.start()),
                bound => bound,
            },
            match bounds.1 {
                Bound::Unbounded => Bound::Included(coverage.end()),
                bound => bound,
            },
        );

        let sessions = schedule.sessions();

        let first = sessions.partition_point(|session| match bounds.0 {
            Bound::Included(start) => session.final_close() < start,
            Bound::Excluded(start) => session.final_close() <= start,
            Bound::Unbounded => false,
        });

        let end = sessions.partition_point(|session| match bounds.1 {
            Bound::Included(end) => session.final_close() <= end,
            Bound::Excluded(end) => session.final_close() < end,
            Bound::Unbounded => true,
        });

        // Equal endpoints with either excluded bound select no close labels.
        let expected = first..end.max(first);
        let mut next = first;
        let mut session_indices = Vec::with_capacity(series.len());
        let mut gaps = Vec::new();

        for (index, &timestamp) in series.timestamps().iter().enumerate() {
            if !bounds.contains(&timestamp) {
                return Err(SessionCloseError::OutsideBounds { index, timestamp, bounds });
            }

            let gap_start = next;
            while next < expected.end && sessions[next].final_close() < timestamp {
                next += 1;
            }

            if gap_start < next {
                gaps.push(gap_start..next);
            }

            if next == expected.end || sessions[next].final_close() != timestamp {
                return Err(SessionCloseError::NotSessionClose { index, timestamp });
            }

            session_indices.push(next);
            next += 1;
        }

        if next < expected.end {
            gaps.push(next..expected.end);
        }

        Ok(Self {
            series,
            schedule,
            bounds,
            expected,
            session_indices,
            gaps,
        })
    }

    /// Returns the original observations without allocation or revalidation.
    #[must_use]
    pub const fn series(&self) -> SeriesView<'a> {
        self.series
    }

    /// Returns the supplied schedule, preserving session identity and coverage.
    #[must_use]
    pub const fn schedule(&self) -> &'s SessionSchedule {
        self.schedule
    }

    /// Returns close-label bounds with unbounded endpoints resolved to
    /// coverage.
    #[must_use]
    pub const fn bounds(&self) -> (Bound<Timestamp>, Bound<Timestamp>) {
        self.bounds
    }

    /// Returns the number of supplied observations, each matching one close.
    #[must_use]
    pub const fn observed_count(&self) -> usize {
        self.series.len()
    }

    /// Returns the number of final session closes selected by the bounds.
    #[must_use]
    pub const fn expected_count(&self) -> usize {
        self.expected.end - self.expected.start
    }

    /// Returns the number of selected session closes without an observation.
    #[must_use]
    pub const fn missing_count(&self) -> usize {
        self.expected_count() - self.observed_count()
    }

    /// Returns whether every selected close has an observation.
    ///
    /// Returns true when no session closes fall within the bounds.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.missing_count() == 0
    }

    /// Returns consecutive missing runs as ranges into the supplied schedule.
    ///
    /// Each range includes its start index and excludes its end index, as in
    /// `schedule.sessions()[range]`. Runs follow session order, regardless of
    /// elapsed gaps between sessions. Use [`Self::missing`] for identifiers
    /// and close timestamps.
    #[must_use]
    pub fn gaps(&self) -> &[Range<usize>] {
        &self.gaps
    }

    /// Yields missing sessions and their final closes in schedule order.
    pub fn missing(&self) -> impl Iterator<Item = SessionClose<'s>> + '_ {
        let sessions = self.schedule.sessions();

        self.gaps
            .iter()
            .flat_map(Clone::clone)
            .map(|session_index| {
                let session = &sessions[session_index];
                SessionClose {
                    session_index,
                    session,
                    timestamp: session.final_close(),
                }
            })
    }

    /// Yields original observations and their sessions in source timestamp
    /// order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = SessionCloseObservation<'s>> + '_ {
        let sessions = self.schedule.sessions();
        self.session_indices
            .iter()
            .enumerate()
            .map(|(index, &session_index)| SessionCloseObservation {
                index,
                timestamp: self.series.timestamps()[index],
                value: self.series.values()[index],
                session_index,
                session: &sessions[session_index],
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{interval, timestamp};

    fn schedule() -> SessionSchedule {
        let coverage = interval(0, 10);
        let sessions = [2, 4, 6, 8, 10]
            .into_iter()
            .enumerate()
            .map(|(index, close)| {
                Session::new(format!("s{index}"), vec![interval(close - 1, close)]).unwrap()
            })
            .collect();

        SessionSchedule::new(coverage, sessions).unwrap()
    }

    mod validation {
        use super::*;

        #[test]
        fn checks_reversed_bounds_before_coverage_or_observations() {
            let schedule = schedule();

            let timestamps = [timestamp(1)];
            let unexpected = SeriesView::new(&timestamps, &[100.0]).unwrap();
            let empty = SeriesView::new(&[], &[]).unwrap();
            let start = timestamp(12);
            let end = timestamp(-1);

            for series in [empty, unexpected] {
                let error = SessionCloseCoverage::new(series, &schedule, start..end).unwrap_err();

                assert_eq!(error, SessionCloseError::ReversedRange { start, end });
            }
        }

        #[test]
        fn rejects_unknown_analysis_periods_even_without_observations() {
            let schedule = schedule();

            let empty = SeriesView::new(&[], &[]).unwrap();
            let bounds = [
                (Bound::Included(timestamp(-1)), Bound::Unbounded),
                (Bound::Excluded(timestamp(-1)), Bound::Unbounded),
                (Bound::Unbounded, Bound::Included(timestamp(11))),
                (Bound::Unbounded, Bound::Excluded(timestamp(11))),
                (
                    Bound::Included(timestamp(11)),
                    Bound::Included(timestamp(11)),
                ),
            ];

            for bounds in bounds {
                let error = SessionCloseCoverage::new(empty, &schedule, bounds).unwrap_err();

                assert_eq!(
                    error,
                    SessionCloseError::BoundsOutsideCoverage {
                        bounds,
                        coverage: schedule.coverage()
                    },
                );
            }
        }

        #[test]
        fn reports_the_first_observation_that_is_not_a_final_close() {
            let coverage = interval(0, 10);

            let split = Session::new("split", vec![interval(1, 3), interval(4, 6)]).unwrap();
            let later = Session::new("later", vec![interval(6, 10)]).unwrap();

            let schedule = SessionSchedule::new(coverage, vec![split, later]).unwrap();

            // The morning close is not the split session's final close.
            let cases = [(vec![3, 6], 0, 3), (vec![6, 7, 8], 1, 7)];

            for (nanos, index, actual) in cases {
                let timestamps: Vec<_> = nanos.into_iter().map(timestamp).collect();
                let values = vec![100.0; timestamps.len()];
                let series = SeriesView::new(&timestamps, &values).unwrap();

                let error = SessionCloseCoverage::new(series, &schedule, ..).unwrap_err();

                assert_eq!(
                    error,
                    SessionCloseError::NotSessionClose { index, timestamp: timestamp(actual) },
                );
            }
        }

        #[test]
        fn rejects_out_of_bounds_observations_instead_of_selecting_them() {
            let schedule = schedule();
            let bounds = (Bound::Included(timestamp(4)), Bound::Excluded(timestamp(8)));
            let timestamps = [4, 8].map(timestamp);
            let series = SeriesView::new(&timestamps, &[100.0, 101.0]).unwrap();

            let error = SessionCloseCoverage::new(series, &schedule, bounds).unwrap_err();

            assert_eq!(
                error,
                SessionCloseError::OutsideBounds { index: 1, timestamp: timestamp(8), bounds },
            );
            assert_eq!(
                error.to_string(),
                "observation at index 1 (8 ns) is outside the session-close analysis bounds",
            );
        }

        #[test]
        fn rejects_observations_during_a_covered_closure() {
            let schedule = SessionSchedule::new(interval(0, 10), vec![]).unwrap();

            let timestamps = [timestamp(5)];
            let series = SeriesView::new(&timestamps, &[100.0]).unwrap();

            let error = SessionCloseCoverage::new(series, &schedule, ..).unwrap_err();

            assert_eq!(
                error,
                SessionCloseError::NotSessionClose { index: 0, timestamp: timestamps[0] },
            );
        }
    }

    mod coverage {
        use super::*;

        #[test]
        fn preserves_sources_and_reports_leading_internal_and_trailing_gaps() {
            let schedule = schedule();

            let timestamps = [4, 8].map(timestamp);
            let values = [100.0, 102.0];
            let series = SeriesView::new(&timestamps, &values).unwrap();

            let coverage = SessionCloseCoverage::new(series, &schedule, ..).unwrap();

            assert!(std::ptr::eq(coverage.series().timestamps(), &timestamps));
            assert!(std::ptr::eq(coverage.series().values(), &values));
            assert!(std::ptr::eq(coverage.schedule(), &raw const schedule));
            assert_eq!(coverage.observed_count(), 2);
            assert_eq!(coverage.expected_count(), 5);
            assert_eq!(coverage.missing_count(), 3);
            assert!(!coverage.is_complete());
            assert_eq!(coverage.gaps(), [0..1, 2..3, 4..5]);
            assert_eq!(coverage.iter().len(), 2);

            let observed: Vec<_> = coverage
                .iter()
                .map(|row| {
                    (
                        row.index,
                        row.timestamp,
                        row.value,
                        row.session_index,
                        row.session.id(),
                    )
                })
                .collect();

            let missing: Vec<_> = coverage
                .missing()
                .map(|close| (close.session_index, close.session.id(), close.timestamp))
                .collect();

            assert_eq!(
                observed,
                [
                    (0, timestamps[0], 100.0, 1, "s1"),
                    (1, timestamps[1], 102.0, 3, "s3")
                ],
            );

            assert_eq!(
                missing,
                [
                    (0, "s0", timestamp(2)),
                    (2, "s2", timestamp(6)),
                    (4, "s4", timestamp(10))
                ],
            );
        }

        #[test]
        fn maps_shared_boundaries_and_the_coverage_end_to_the_ending_session() {
            let coverage = interval(0, 10);

            let split = Session::new("split", vec![interval(1, 3), interval(4, 6)]).unwrap();
            let later = Session::new("later", vec![interval(6, 10)]).unwrap();

            let schedule = SessionSchedule::new(coverage, vec![split, later]).unwrap();

            let timestamps = [6, 10].map(timestamp);
            let series = SeriesView::new(&timestamps, &[100.0, 101.0]).unwrap();

            let coverage = SessionCloseCoverage::new(series, &schedule, ..).unwrap();
            let identifiers: Vec<_> = coverage.iter().map(|row| row.session.id()).collect();

            assert_eq!(identifiers, ["split", "later"]);
            assert!(coverage.is_complete());
            assert_eq!(coverage.gaps(), []);
            assert_eq!(coverage.missing().count(), 0);
        }

        #[test]
        fn an_empty_series_reports_all_selected_sessions_as_one_gap() {
            let schedule = schedule();

            let empty = SeriesView::new(&[], &[]).unwrap();

            let coverage =
                SessionCloseCoverage::new(empty, &schedule, timestamp(4)..=timestamp(8)).unwrap();
            let expected_gap = 1..4;

            assert_eq!(coverage.observed_count(), 0);
            assert_eq!(coverage.expected_count(), 3);
            assert_eq!(coverage.missing_count(), 3);
            assert_eq!(coverage.gaps(), [expected_gap]);
            assert!(coverage.iter().next().is_none());
        }

        #[test]
        fn counts_only_supplied_sessions_across_a_long_closure() {
            let at = |hour: i64| Timestamp::from_unix_secs(hour * 3600).unwrap();

            let first_close = at(16);
            let next_close = at(3 * 24 + 16);

            let first_open = TimeInterval::new(at(9), first_close).unwrap();
            let next_open = TimeInterval::new(at(3 * 24 + 9), next_close).unwrap();

            let before = Session::new("before-break", vec![first_open]).unwrap();
            let after = Session::new("after-break", vec![next_open]).unwrap();

            let coverage = TimeInterval::new(at(0), at(4 * 24)).unwrap();
            let schedule = SessionSchedule::new(coverage, vec![before, after]).unwrap();

            let timestamps = [first_close];
            let series = SeriesView::new(&timestamps, &[100.0]).unwrap();

            let coverage = SessionCloseCoverage::new(series, &schedule, ..).unwrap();

            let missing: Vec<_> = coverage
                .missing()
                .map(|close| (close.session.id(), close.timestamp))
                .collect();

            assert_eq!(coverage.expected_count(), 2);
            assert_eq!(missing, [("after-break", next_close)]);

            let empty = SeriesView::new(&[], &[]).unwrap();

            let closure = SessionCloseCoverage::new(empty, &schedule, at(24)..at(3 * 24)).unwrap();

            assert_eq!(closure.expected_count(), 0);
            assert!(closure.is_complete());
        }

        #[test]
        fn a_covered_closure_has_no_expected_observations() {
            let schedule = SessionSchedule::new(interval(0, 10), vec![]).unwrap();

            let empty = SeriesView::new(&[], &[]).unwrap();

            let coverage = SessionCloseCoverage::new(empty, &schedule, ..).unwrap();

            assert_eq!(coverage.expected_count(), 0);
            assert_eq!(coverage.missing_count(), 0);
            assert!(coverage.is_complete());
            assert_eq!(coverage.gaps(), []);
            assert_eq!(coverage.missing().count(), 0);
        }

        #[test]
        fn indices_are_relative_to_the_supplied_series_view() {
            let schedule = schedule();

            let timestamps = [-2, 2, 4, 8, 12].map(timestamp);
            let values = [99.0, 100.0, 101.0, 102.0, 103.0];
            let series = SeriesView::new(&timestamps, &values).unwrap();
            let bounds = timestamp(2)..=timestamp(8);
            let selected = series.between(bounds.clone()).unwrap();

            let coverage = SessionCloseCoverage::new(selected, &schedule, bounds).unwrap();
            let rows: Vec<_> = coverage
                .iter()
                .map(|row| (row.index, row.timestamp, row.value, row.session_index))
                .collect();

            assert_eq!(
                rows,
                [
                    (0, timestamp(2), 100.0, 0),
                    (1, timestamp(4), 101.0, 1),
                    (2, timestamp(8), 102.0, 3)
                ],
            );
        }
    }

    mod boundaries {
        use std::ops::Bound::{Excluded, Included, Unbounded};

        use super::*;

        #[test]
        fn bounds_select_close_labels_with_each_endpoint_convention() {
            let schedule = schedule();

            let empty = SeriesView::new(&[], &[]).unwrap();
            let start = timestamp(4);
            let end = timestamp(8);
            let cases = [
                ((Included(start), Included(end)), vec![1, 2, 3]),
                ((Excluded(start), Included(end)), vec![2, 3]),
                ((Included(start), Excluded(end)), vec![1, 2]),
                ((Excluded(start), Excluded(end)), vec![2]),
                ((Unbounded, Included(start)), vec![0, 1]),
                ((Excluded(end), Unbounded), vec![4]),
            ];

            for (bounds, expected) in cases {
                let coverage = SessionCloseCoverage::new(empty, &schedule, bounds).unwrap();
                let missing: Vec<_> = coverage
                    .missing()
                    .map(|close| close.session_index)
                    .collect();

                assert_eq!(missing, expected, "{bounds:?}");
                assert_eq!(coverage.expected_count(), expected.len(), "{bounds:?}");
            }
        }

        #[test]
        fn equal_endpoints_select_a_close_only_when_both_are_inclusive() {
            let schedule = schedule();

            let empty = SeriesView::new(&[], &[]).unwrap();
            let close = timestamp(4);
            let cases = [
                ((Included(close), Included(close)), vec![1]),
                ((Excluded(close), Included(close)), vec![]),
                ((Included(close), Excluded(close)), vec![]),
                ((Excluded(close), Excluded(close)), vec![]),
            ];

            for (bounds, expected) in cases {
                let coverage = SessionCloseCoverage::new(empty, &schedule, bounds).unwrap();
                let missing: Vec<_> = coverage
                    .missing()
                    .map(|close| close.session_index)
                    .collect();

                assert_eq!(missing, expected, "{bounds:?}");
                assert_eq!(coverage.expected_count(), expected.len(), "{bounds:?}");
            }
        }

        #[test]
        fn excluding_a_coverage_endpoint_can_select_no_closes() {
            let schedule = schedule();

            let empty = SeriesView::new(&[], &[]).unwrap();
            let start = schedule.coverage().start();
            let end = schedule.coverage().end();
            let cases = [(Unbounded, Excluded(start)), (Excluded(end), Unbounded)];

            for bounds in cases {
                let coverage = SessionCloseCoverage::new(empty, &schedule, bounds).unwrap();

                assert_eq!(coverage.expected_count(), 0, "{bounds:?}");
                assert!(coverage.missing().next().is_none(), "{bounds:?}");
            }
        }

        #[test]
        fn unbounded_ranges_resolve_to_both_coverage_endpoints_inclusive() {
            let schedule = schedule();

            let empty = SeriesView::new(&[], &[]).unwrap();

            let coverage = SessionCloseCoverage::new(empty, &schedule, ..).unwrap();

            assert_eq!(
                coverage.bounds(),
                (Included(timestamp(0)), Included(timestamp(10))),
            );

            let timestamps = [timestamp(-1)];
            let outside = SeriesView::new(&timestamps, &[100.0]).unwrap();

            let error = SessionCloseCoverage::new(outside, &schedule, ..).unwrap_err();

            assert_eq!(
                error,
                SessionCloseError::OutsideBounds {
                    index: 0,
                    timestamp: timestamps[0],
                    bounds: coverage.bounds(),
                },
            );
        }

        #[test]
        fn preserves_exact_labels_across_the_full_timestamp_span() {
            let first_close = Timestamp::from_unix_nanos(-1);
            let coverage = TimeInterval::new(Timestamp::MIN, Timestamp::MAX).unwrap();
            let first_open = TimeInterval::new(Timestamp::MIN, first_close).unwrap();
            let second_open = TimeInterval::new(first_close, Timestamp::MAX).unwrap();
            let first = Session::new("first", vec![first_open]).unwrap();
            let second = Session::new("second", vec![second_open]).unwrap();
            let schedule = SessionSchedule::new(coverage, vec![first, second]).unwrap();

            let timestamps = [first_close, Timestamp::MAX];
            let series = SeriesView::new(&timestamps, &[100.0, 101.0]).unwrap();

            let coverage = SessionCloseCoverage::new(series, &schedule, ..).unwrap();
            let labels: Vec<_> = coverage.iter().map(|row| row.timestamp).collect();

            assert_eq!(labels, timestamps);
            assert!(coverage.is_complete());
        }
    }
}
