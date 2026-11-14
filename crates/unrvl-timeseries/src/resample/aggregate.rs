use std::ops::Range;

use unrvl_numerics::summation::NeumaierSumExt;
use unrvl_stats::moments;

use super::{BucketAssignment, BucketLabel, SessionAssignment};
use crate::{Session, TimeInterval, Timestamp};

/// Outcome of combining the observations in a bucket.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AggregateOutcome {
    /// No observations fall within the selected part of the bucket.
    Empty,

    /// A finite calculated value, including zero.
    Value(f64),

    /// A nonempty bucket's calculation returned a non-finite value.
    ///
    /// The returned value is retained for diagnostics. Its precise numerical
    /// cause is not inferred from the float alone.
    NumericalFailure {
        /// Original non-finite result, preserving infinity or `NaN`.
        result: f64,
    },
}

/// A calculated bucket value and its temporal and source context.
///
/// Every bucket retains its context, including empty buckets and numerical
/// failures. `source_range` identifies all contributing observations; an
/// aggregate does not identify one original observation as its source value.
/// The record owns its value and metadata independently of the input series.
#[derive(Debug, Clone, PartialEq)]
pub struct BucketAggregate {
    /// Full bucket boundary selected by [`BucketLabel`].
    pub timestamp: Timestamp,

    /// Full origin-anchored half-open bucket interval.
    pub interval: TimeInterval,

    /// Portion of the bucket inside the analysis bounds.
    pub selected_interval: TimeInterval,

    /// All contributing observation indices, relative to the supplied view.
    pub source_range: Range<usize>,

    /// Empty, finite, or failed calculation for this bucket.
    pub outcome: AggregateOutcome,
}

impl BucketAggregate {
    /// Returns the number of supplied observations in this bucket.
    #[must_use]
    pub fn observed_count(&self) -> usize {
        self.source_range.len()
    }

    /// Returns whether analysis bounds clip either end of the full bucket.
    /// This describes interval geometry, not sampling completeness.
    #[must_use]
    pub fn is_partial(&self) -> bool {
        self.selected_interval != self.interval
    }
}

/// A calculated session value with its temporal and source context.
///
/// Empty sessions and numerical failures retain the same context as finite
/// results. The original session is borrowed from the schedule; all other
/// fields are owned independently of the source series and assignment.
/// Envelopes can include breaks and do not measure scheduled open time.
#[derive(Debug, Clone)]
pub struct SessionAggregate<'s> {
    /// First opening or final close, selected by [`BucketLabel`].
    pub timestamp: Timestamp,

    /// Index into the original
    /// [`SessionSchedule::sessions`](crate::SessionSchedule::sessions) slice.
    pub session_index: usize,

    /// Original session, including its identifier and open periods.
    pub session: &'s Session,

    /// Full half-open envelope from the first opening to the final close.
    pub interval: TimeInterval,

    /// Intersection of the full envelope with the analysis interval.
    pub selected_interval: TimeInterval,

    /// All contributing observation indices, relative to the supplied view.
    pub source_range: Range<usize>,

    /// Empty, finite, or failed calculation for this session.
    pub outcome: AggregateOutcome,
}

impl SessionAggregate<'_> {
    /// Returns the number of supplied observations in this session bucket.
    #[must_use]
    pub fn observed_count(&self) -> usize {
        self.source_range.len()
    }

    /// Returns whether analysis bounds clip either end of the session envelope.
    /// This describes interval geometry, not sampling completeness.
    #[must_use]
    pub fn is_partial(&self) -> bool {
        self.selected_interval != self.interval
    }
}

impl<'a> BucketAssignment<'a> {
    /// Calculates the arithmetic mean of the observations in each assigned
    /// bucket.
    ///
    /// Each observation has equal weight, regardless of timestamp spacing.
    /// Yields every bucket in chronological order, including empty and
    /// partial buckets. Empty buckets have [`AggregateOutcome::Empty`]; a
    /// nonempty bucket's mean may be zero. No observations are filled or
    /// carried between buckets.
    ///
    /// Uses [`moments::mean`], including its compensated summation, overflow
    /// fallback, and rounding behavior. A non-finite return is recorded as
    /// [`AggregateOutcome::NumericalFailure`] with the original result. Each
    /// row retains its bucket intervals, requested full-boundary label, and
    /// source range. A populated bucket does not establish complete
    /// sampling.
    ///
    /// Iteration is lazy and uses `O(n + b)` time and `O(1)` additional memory
    /// for `n` observations and `b` buckets. Multiple passes within disjoint
    /// buckets remain linear across the series.
    ///
    /// # Panics
    ///
    /// Panics if a nonconstant bucket contains more than `2^53` observations,
    /// matching [`moments::mean`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::time::Duration;
    /// # use unrvl_timeseries::{SeriesView, TimeInterval, Timestamp};
    /// # use unrvl_timeseries::resample::{AggregateOutcome, BucketLabel, FixedBuckets};
    /// let origin = Timestamp::UNIX_EPOCH;
    /// let end = Timestamp::from_unix_secs(2 * 3600)?;
    /// let bounds = TimeInterval::new(origin, end)?;
    /// let buckets = FixedBuckets::new(origin, Duration::from_secs(3600), bounds)?;
    ///
    /// let timestamps = [
    ///     Timestamp::from_unix_secs(5 * 60)?,
    ///     Timestamp::from_unix_secs(20 * 60)?,
    ///     Timestamp::from_unix_secs(55 * 60)?,
    /// ];
    /// let values = [100.0, 110.0, 102.0];
    /// let series = SeriesView::new(&timestamps, &values)?;
    /// let assigned = buckets.assign(series)?;
    /// let rows: Vec<_> = assigned.mean(BucketLabel::End).collect();
    ///
    /// assert_eq!(rows[0].timestamp, Timestamp::from_unix_secs(3600)?);
    /// assert_eq!(rows[0].outcome, AggregateOutcome::Value(104.0));
    /// assert_eq!(rows[0].observed_count(), 3);
    /// assert_eq!(rows[1].outcome, AggregateOutcome::Empty);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn mean(&self, label: BucketLabel) -> impl Iterator<Item = BucketAggregate> + use<'a> {
        aggregate_fixed(self, label, moments::mean)
    }

    /// Adds the observed values in each assigned bucket.
    ///
    /// Suitable for event amounts such as signed liquidity flows. Each
    /// observation contributes once; elapsed time does not weight the values.
    /// Yields every bucket, including empty and partial buckets, with its label
    /// and source context.
    ///
    /// Empty buckets have [`AggregateOutcome::Empty`]. Observed amounts that
    /// cancel have [`AggregateOutcome::Value`] with a total of zero. Empty
    /// means no observations were supplied; it does not prove that no events
    /// occurred.
    ///
    /// Uses [`NeumaierSumExt::neumaier_sum`] for compensated accumulation.
    /// Intermediate overflow can produce a non-finite total even when the exact
    /// sum is representable. A non-finite total is recorded as
    /// [`AggregateOutcome::NumericalFailure`] with the original result; later
    /// buckets are still calculated. Iteration is lazy and uses `O(n + b)` time
    /// and `O(1)` additional memory for `n` observations and `b` buckets.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::time::Duration;
    /// # use unrvl_timeseries::{SeriesView, TimeInterval, Timestamp};
    /// # use unrvl_timeseries::resample::{AggregateOutcome, BucketLabel, FixedBuckets};
    /// let origin = Timestamp::UNIX_EPOCH;
    /// let end = Timestamp::from_unix_secs(2 * 60)?;
    /// let bounds = TimeInterval::new(origin, end)?;
    /// let buckets = FixedBuckets::new(origin, Duration::from_secs(60), bounds)?;
    ///
    /// let timestamps = [
    ///     Timestamp::from_unix_secs(5)?,
    ///     Timestamp::from_unix_secs(20)?,
    /// ];
    /// let flows = SeriesView::new(&timestamps, &[10.0, -10.0])?;
    /// let assigned = buckets.assign(flows)?;
    /// let rows: Vec<_> = assigned.sum(BucketLabel::End).collect();
    ///
    /// assert_eq!(rows[0].outcome, AggregateOutcome::Value(0.0));
    /// assert_eq!(rows[0].observed_count(), 2);
    /// assert_eq!(rows[1].outcome, AggregateOutcome::Empty);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn sum(&self, label: BucketLabel) -> impl Iterator<Item = BucketAggregate> + use<'a> {
        aggregate_fixed(self, label, |values| values.iter().neumaier_sum())
    }
}

impl<'a, 's> SessionAssignment<'a, 's> {
    /// Calculates the arithmetic mean of the observations in each session.
    ///
    /// Uses [`moments::mean`] across the selected open periods. Each
    /// observation has equal weight; elapsed time and session breaks do not
    /// weight values. Empty sessions have [`AggregateOutcome::Empty`].
    /// Non-finite returns are preserved as
    /// [`AggregateOutcome::NumericalFailure`].
    ///
    /// Every result keeps its session identity, full and selected envelopes,
    /// requested label, and source range. A label does not establish when the
    /// result became available. See [`SessionAssignment`] for an example.
    ///
    /// Lazy iteration takes `O(n + s)` time and `O(1)` extra memory for `n`
    /// observations and `s` selected sessions. Numerical rounding and overflow
    /// fallback follow [`moments::mean`].
    ///
    /// # Panics
    ///
    /// Panics if a nonconstant session bucket contains more than `2^53`
    /// observations, matching [`moments::mean`].
    pub fn mean(
        &self,
        label: BucketLabel,
    ) -> impl Iterator<Item = SessionAggregate<'s>> + use<'a, 's> {
        aggregate_sessions(self, label, moments::mean)
    }

    /// Adds the observed values in each session's selected open periods.
    ///
    /// Suitable for signed event amounts. Each observation contributes once;
    /// elapsed time does not weight the values. Empty sessions remain
    /// [`AggregateOutcome::Empty`], distinct from an observed sum of zero.
    ///
    /// Uses [`NeumaierSumExt::neumaier_sum`]. Intermediate overflow can produce
    /// a non-finite total even when the exact sum is representable. Failures
    /// retain the original result and session context, and later sessions are
    /// still calculated. See [`SessionAggregate`] for the result context.
    ///
    /// Lazy iteration takes `O(n + s)` time and `O(1)` extra memory for `n`
    /// observations and `s` selected sessions.
    pub fn sum(
        &self,
        label: BucketLabel,
    ) -> impl Iterator<Item = SessionAggregate<'s>> + use<'a, 's> {
        aggregate_sessions(self, label, |values| values.iter().neumaier_sum())
    }
}

/// Applies a calculation to each nonempty bucket, preserving context and the
/// original result if it is non-finite. Empty buckets bypass the calculation.
fn aggregate_fixed<'a, F>(
    assignment: &BucketAssignment<'a>,
    label: BucketLabel,
    calculate: F,
) -> impl Iterator<Item = BucketAggregate> + use<'a, F>
where
    F: Fn(&[f64]) -> f64,
{
    assignment.iter().map(move |bucket| {
        let outcome = aggregate_outcome(bucket.observations.values(), &calculate);

        BucketAggregate {
            timestamp: bucket.label(label),
            interval: bucket.interval,
            selected_interval: bucket.selected_interval,
            source_range: bucket.source_range,
            outcome,
        }
    })
}

/// Adds session identity to the calculation outcome and its bucket context.
fn aggregate_sessions<'a, 's, F>(
    assignment: &SessionAssignment<'a, 's>,
    label: BucketLabel,
    calculate: F,
) -> impl Iterator<Item = SessionAggregate<'s>> + use<'a, 's, F>
where
    F: Fn(&[f64]) -> f64,
{
    assignment.iter().map(move |bucket| {
        let outcome = aggregate_outcome(bucket.observations.values(), &calculate);

        SessionAggregate {
            timestamp: bucket.label(label),
            session_index: bucket.session_index,
            session: bucket.session,
            interval: bucket.interval,
            selected_interval: bucket.selected_interval,
            source_range: bucket.source_range,
            outcome,
        }
    })
}

/// Empty input bypasses the calculation. Non-finite returns keep their original
/// value without inferring a specific numerical cause.
fn aggregate_outcome(values: &[f64], calculate: impl FnOnce(&[f64]) -> f64) -> AggregateOutcome {
    if values.is_empty() {
        return AggregateOutcome::Empty;
    }

    let result = calculate(values);
    if result.is_finite() {
        AggregateOutcome::Value(result)
    } else {
        AggregateOutcome::NumericalFailure { result }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::SeriesView;
    use crate::resample::FixedBuckets;
    use crate::test_helpers::{interval, timestamp};

    mod fixed {
        use super::*;

        fn buckets(start_nanos: i64, end_nanos: i64) -> FixedBuckets {
            FixedBuckets::new(
                Timestamp::UNIX_EPOCH,
                Duration::from_nanos(4),
                interval(start_nanos, end_nanos),
            )
            .unwrap()
        }

        #[test]
        fn averages_observations_with_equal_weight_and_retains_bucket_context() {
            let rows: Vec<_> = {
                let timestamps = [0, 1, 3, 8, 9, 11].map(timestamp);
                let values = [100.0, 110.0, 102.0, 2.0, -2.0, 0.0];
                let series = SeriesView::new(&timestamps, &values).unwrap();

                let assigned = buckets(0, 12).assign(series).unwrap();

                assigned.mean(BucketLabel::End).collect()
            };
            let cases = [
                (interval(0, 4), 0..3, AggregateOutcome::Value(104.0)),
                (interval(4, 8), 3..3, AggregateOutcome::Empty),
                (interval(8, 12), 3..6, AggregateOutcome::Value(0.0)),
            ];

            assert_eq!(rows.len(), cases.len());
            for (row, (expected_interval, source_range, outcome)) in rows.iter().zip(cases) {
                assert_eq!(row.timestamp, expected_interval.end());
                assert_eq!(row.interval, expected_interval);
                assert_eq!(row.selected_interval, expected_interval);
                assert_eq!(row.observed_count(), source_range.len());
                assert_eq!(row.source_range, source_range);
                assert_eq!(row.outcome, outcome);
                assert!(!row.is_partial());
            }
        }

        #[test]
        fn preserves_subview_ranges_and_partial_edges_independently_of_labels() {
            let timestamps = [-1, 1, 2, 4, 5, 8].map(timestamp);
            let values = [-100.0, 2.0, 4.0, 6.0, 10.0, 100.0];
            let original = SeriesView::new(&timestamps, &values).unwrap();
            let series = original.between(interval(1, 6)).unwrap();

            let assigned = buckets(1, 6).assign(series).unwrap();
            let starts: Vec<_> = assigned.mean(BucketLabel::Start).collect();
            let ends: Vec<_> = assigned.mean(BucketLabel::End).collect();
            let cases = [
                (
                    interval(0, 4),
                    interval(1, 4),
                    0..2,
                    AggregateOutcome::Value(3.0),
                ),
                (
                    interval(4, 8),
                    interval(4, 6),
                    2..4,
                    AggregateOutcome::Value(8.0),
                ),
            ];

            assert_eq!(starts.len(), cases.len());
            assert_eq!(ends.len(), cases.len());
            for ((start, end), (full, selected, source_range, outcome)) in
                starts.iter().zip(&ends).zip(cases)
            {
                assert_eq!(start.timestamp, full.start());
                assert_eq!(end.timestamp, full.end());
                assert_eq!(end.interval, full);
                assert_eq!(end.selected_interval, selected);
                assert_eq!(end.source_range, source_range);
                assert_eq!(end.outcome, outcome);
                assert_eq!(start.outcome, end.outcome);
                assert!(start.is_partial());
                assert!(end.is_partial());
            }
        }

        #[test]
        fn uses_batch_compensation_and_overflow_fallback() {
            let timestamps = [0, 1, 2, 3, 4, 5].map(timestamp);
            // The first bucket cancels large terms. The second has a finite
            // mean even though its direct sum overflows.
            let values = [1.0, 1e16, 1.0, -1e16, f64::MAX, f64::MAX / 2.0];
            let series = SeriesView::new(&timestamps, &values).unwrap();

            let assigned = buckets(0, 8).assign(series).unwrap();
            let outcomes: Vec<_> = assigned
                .mean(BucketLabel::End)
                .map(|row| row.outcome)
                .collect();

            assert_eq!(
                outcomes,
                [
                    AggregateOutcome::Value(0.5),
                    AggregateOutcome::Value(f64::MAX * 0.75)
                ],
            );
        }

        #[test]
        fn sums_signed_amounts_and_distinguishes_empty_buckets_from_zero() {
            let timestamps = [0, 1, 3, 8, 9].map(timestamp);
            let values = [5.0, -2.0, 4.0, 10.0, -10.0];
            let series = SeriesView::new(&timestamps, &values).unwrap();

            let assigned = buckets(0, 12).assign(series).unwrap();
            let outcomes: Vec<_> = assigned
                .sum(BucketLabel::End)
                .map(|row| row.outcome)
                .collect();

            assert_eq!(
                outcomes,
                [
                    AggregateOutcome::Value(7.0),
                    AggregateOutcome::Empty,
                    AggregateOutcome::Value(0.0)
                ],
            );
        }

        #[test]
        fn sum_uses_compensation_and_reports_intermediate_overflow() {
            let timestamps = [0, 1, 2, 3, 4, 5, 6, 8, 9].map(timestamp);
            let maximum = f64::MAX;
            // Compensation recovers the small terms in the first bucket. In the
            // second, intermediate overflow prevents a finite total despite an
            // exact sum of MAX. The last bucket is calculated independently.
            let values = [1.0, 1e16, 1.0, -1e16, maximum, maximum, -maximum, 3.0, -1.0];
            let series = SeriesView::new(&timestamps, &values).unwrap();

            let assigned = buckets(0, 12).assign(series).unwrap();
            let rows: Vec<_> = assigned.sum(BucketLabel::End).collect();

            assert_eq!(rows.len(), 3);
            assert_eq!(rows[0].outcome, AggregateOutcome::Value(2.0));
            assert_eq!(rows[2].outcome, AggregateOutcome::Value(2.0));

            let failed = &rows[1];
            let AggregateOutcome::NumericalFailure { result } = failed.outcome else {
                panic!("expected a numerical failure, got {:?}", failed.outcome);
            };
            assert!(result.is_nan());
            assert_eq!(failed.timestamp, timestamp(8));
            assert_eq!(failed.interval, interval(4, 8));
            assert_eq!(failed.selected_interval, interval(4, 8));
            assert_eq!(failed.source_range, 4..7);
            assert_eq!(failed.observed_count(), 3);
        }

        #[test]
        fn empty_buckets_bypass_the_numerical_calculation() {
            let series = SeriesView::new(&[], &[]).unwrap();

            let assigned = buckets(0, 8).assign(series).unwrap();
            let outcomes: Vec<_> = aggregate_fixed(&assigned, BucketLabel::End, |_| {
                panic!("an empty bucket must not invoke the calculation")
            })
            .map(|row| row.outcome)
            .collect();

            assert_eq!(outcomes, [AggregateOutcome::Empty, AggregateOutcome::Empty]);
        }

        #[test]
        fn nonfinite_results_keep_their_bucket_context_and_allow_later_results() {
            let timestamps = [0, 4, 5].map(timestamp);
            let values = [1.0, 2.0, 4.0];
            let series = SeriesView::new(&timestamps, &values).unwrap();

            let assigned = buckets(0, 8).assign(series).unwrap();

            // Inject failures to check result handling independently of the
            // statistics kernel's numerical behavior.
            for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let rows: Vec<_> = aggregate_fixed(&assigned, BucketLabel::End, |values| {
                    if values.len() == 1 { invalid } else { 3.0 }
                })
                .collect();

                assert_eq!(rows.len(), 2);
                assert_eq!(rows[0].timestamp, timestamp(4));
                assert_eq!(rows[0].interval, interval(0, 4));
                assert_eq!(rows[0].selected_interval, interval(0, 4));
                assert_eq!(rows[0].source_range, 0..1);
                assert_eq!(rows[0].observed_count(), 1);

                let AggregateOutcome::NumericalFailure { result } = rows[0].outcome else {
                    panic!("expected a numerical failure, got {:?}", rows[0].outcome);
                };
                assert_eq!(result.to_bits(), invalid.to_bits());
                assert_eq!(rows[1].outcome, AggregateOutcome::Value(3.0));
            }
        }

        #[test]
        fn a_prefix_of_a_large_plan_is_generated_lazily() {
            let bounds = TimeInterval::new(Timestamp::MIN, Timestamp::MAX).unwrap();
            let buckets =
                FixedBuckets::new(Timestamp::UNIX_EPOCH, Duration::from_nanos(1), bounds).unwrap();
            let series = SeriesView::new(&[], &[]).unwrap();
            let assigned = buckets.assign(series).unwrap();
            let rows: Vec<_> = assigned.mean(BucketLabel::End).take(2).collect();

            assert_eq!(rows.len(), 2);
            assert_eq!(rows[0].interval.start(), Timestamp::MIN);
            assert_eq!(rows[0].interval.end(), rows[1].interval.start());
            assert!(
                rows.iter()
                    .all(|row| row.outcome == AggregateOutcome::Empty)
            );
        }
    }

    mod sessions {
        use super::*;
        use crate::SessionSchedule;

        fn schedule_with_break() -> SessionSchedule {
            let coverage = interval(0, 18);
            let split = Session::new("split", vec![interval(1, 3), interval(5, 8)]).unwrap();
            let empty = Session::new("empty", vec![interval(9, 11)]).unwrap();
            let later = Session::new("later", vec![interval(12, 16)]).unwrap();

            SessionSchedule::new(coverage, vec![split, empty, later]).unwrap()
        }

        #[test]
        fn aggregates_open_periods_and_distinguishes_empty_sessions_from_zero() {
            let schedule = schedule_with_break();

            let timestamps = [1, 2, 5, 7, 12, 15].map(timestamp);
            let values = [100.0, 110.0, 90.0, 100.0, 5.0, -5.0];
            let series = SeriesView::new(&timestamps, &values).unwrap();

            let assigned = SessionAssignment::new(series, &schedule, schedule.coverage()).unwrap();
            let means: Vec<_> = assigned.mean(BucketLabel::End).collect();
            let totals: Vec<_> = assigned
                .sum(BucketLabel::End)
                .map(|row| row.outcome)
                .collect();
            let averages: Vec<_> = means.iter().map(|row| row.outcome).collect();

            assert_eq!(
                averages,
                [
                    AggregateOutcome::Value(100.0),
                    AggregateOutcome::Empty,
                    AggregateOutcome::Value(0.0)
                ],
            );
            assert_eq!(
                totals,
                [
                    AggregateOutcome::Value(400.0),
                    AggregateOutcome::Empty,
                    AggregateOutcome::Value(0.0)
                ],
            );

            let cases = [
                ("split", interval(1, 8), 0..4),
                ("empty", interval(9, 11), 4..4),
                ("later", interval(12, 16), 4..6),
            ];

            assert_eq!(means.len(), cases.len());
            for (index, (row, (id, envelope, source_range))) in means.iter().zip(cases).enumerate()
            {
                assert_eq!(row.session_index, index);
                assert_eq!(row.session.id(), id);
                assert_eq!(row.timestamp, envelope.end());
                assert_eq!(row.interval, envelope);
                assert_eq!(row.selected_interval, envelope);
                assert_eq!(row.observed_count(), source_range.len());
                assert_eq!(row.source_range, source_range);
                assert!(!row.is_partial());
            }
        }

        #[test]
        fn partial_subview_aggregates_outlive_the_source_and_assignment() {
            let schedule = schedule_with_break();
            let bounds = interval(13, 15);
            let (average, total) = {
                let timestamps = [1, 12, 13, 14, 15].map(timestamp);
                let values = [-100.0, 999.0, 2.0, 6.0, 999.0];
                let original = SeriesView::new(&timestamps, &values).unwrap();
                let series = original.between(bounds).unwrap();

                let assigned = SessionAssignment::new(series, &schedule, bounds).unwrap();

                (
                    assigned.mean(BucketLabel::End).next().unwrap(),
                    assigned.sum(BucketLabel::Start).next().unwrap(),
                )
            };

            assert_eq!(average.outcome, AggregateOutcome::Value(4.0));
            assert_eq!(total.outcome, AggregateOutcome::Value(8.0));
            assert_eq!(average.timestamp, timestamp(16));
            assert_eq!(total.timestamp, timestamp(12));

            for row in [&average, &total] {
                assert_eq!(row.session_index, 2);
                assert_eq!(row.session.id(), "later");
                assert_eq!(row.interval, interval(12, 16));
                assert_eq!(row.selected_interval, bounds);
                assert_eq!(row.source_range, 0..2);
                assert_eq!(row.observed_count(), 2);
                assert!(row.is_partial());
            }
        }

        #[test]
        fn mean_uses_batch_compensation_and_overflow_fallback() {
            let schedule = schedule_with_break();

            let timestamps = [1, 2, 5, 7, 12, 15].map(timestamp);
            // The first session cancels large terms across its break. The
            // last has a finite mean even though its direct sum overflows.
            let values = [1.0, 1e16, 1.0, -1e16, f64::MAX, f64::MAX / 2.0];
            let series = SeriesView::new(&timestamps, &values).unwrap();

            let assigned = SessionAssignment::new(series, &schedule, schedule.coverage()).unwrap();
            let outcomes: Vec<_> = assigned
                .mean(BucketLabel::End)
                .map(|row| row.outcome)
                .collect();

            assert_eq!(
                outcomes,
                [
                    AggregateOutcome::Value(0.5),
                    AggregateOutcome::Empty,
                    AggregateOutcome::Value(f64::MAX * 0.75),
                ],
            );
        }

        #[test]
        fn sum_preserves_failure_context_and_calculates_later_sessions() {
            let schedule = schedule_with_break();

            let timestamps = [1, 2, 5, 12, 13, 14, 15].map(timestamp);
            let maximum = f64::MAX;
            let values = [maximum, maximum, -maximum, 1.0, 1e16, 1.0, -1e16];
            let series = SeriesView::new(&timestamps, &values).unwrap();

            let assigned = SessionAssignment::new(series, &schedule, schedule.coverage()).unwrap();
            let rows: Vec<_> = assigned.sum(BucketLabel::End).collect();

            assert_eq!(rows.len(), 3);
            let failed = &rows[0];
            let AggregateOutcome::NumericalFailure { result } = failed.outcome else {
                panic!("expected a numerical failure, got {:?}", failed.outcome);
            };

            assert!(result.is_nan());
            assert_eq!(failed.session_index, 0);
            assert_eq!(failed.session.id(), "split");
            assert_eq!(failed.timestamp, timestamp(8));
            assert_eq!(failed.interval, interval(1, 8));
            assert_eq!(failed.source_range, 0..3);
            assert_eq!(rows[1].outcome, AggregateOutcome::Empty);
            assert_eq!(rows[2].outcome, AggregateOutcome::Value(2.0));
        }
    }
}
