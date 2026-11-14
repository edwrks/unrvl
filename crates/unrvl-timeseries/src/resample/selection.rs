use std::ops::Range;

use unrvl_numerics::extrema::{argmax, argmin};

use super::{BucketAssignment, BucketLabel, SessionAssignment};
use crate::{SeriesView, Session, TimeInterval, Timestamp};

/// An observation selected from a bucket, preserving its source identity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BucketObservation {
    /// Index into the series view supplied to assignment.
    pub index: usize,

    /// Original observation timestamp, independent of the output label.
    pub timestamp: Timestamp,

    /// Original finite observation value.
    pub value: f64,
}

/// A bucket's output label, temporal context, and optional selected
/// observation.
///
/// Empty buckets retain their context with `observation = None`. A populated
/// bucket does not establish sampling completeness. The record owns its values
/// and metadata and can be retained independently of the source series.
///
/// # Examples
///
/// Hourly buckets intersect an analysis interval from 09:15 to 10:45. The first
/// bucket's high occurs at 09:42, while its end label is 10:00.
///
/// ```
/// # use std::time::Duration;
/// # use unrvl_timeseries::{SeriesView, TimeInterval, Timestamp};
/// # use unrvl_timeseries::resample::{BucketLabel, FixedBuckets};
/// let at = |hour: i64, minute: i64| Timestamp::from_unix_secs(hour * 3600 + minute * 60);
/// let bounds = TimeInterval::new(at(9, 15)?, at(10, 45)?)?;
/// let buckets = FixedBuckets::new(at(9, 0)?, Duration::from_secs(3600), bounds)?;
///
/// let timestamps = [at(9, 20)?, at(9, 42)?, at(10, 10)?];
/// let values = [100.0, 103.0, 101.0];
/// let prices = SeriesView::new(&timestamps, &values)?;
///
/// let assigned = buckets.assign(prices)?;
/// let mut highs = assigned.max(BucketLabel::End);
/// let first = highs.next().unwrap();
///
/// assert_eq!(first.interval, TimeInterval::new(at(9, 0)?, at(10, 0)?)?);
/// assert_eq!(
///     first.selected_interval,
///     TimeInterval::new(at(9, 15)?, at(10, 0)?)?
/// );
/// assert_eq!(first.timestamp, at(10, 0)?);
/// assert_eq!(first.source_range, 0..2);
/// assert!(first.is_partial());
///
/// let observation = first.observation.unwrap();
/// assert_eq!(observation.index, 1);
/// assert_eq!(observation.timestamp, at(9, 42)?);
/// assert_eq!(observation.value, 103.0);
///
/// // The last full bucket ends after the selected analysis interval.
/// let last = highs.next().unwrap();
/// assert_eq!(last.timestamp, at(11, 0)?);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct BucketSelection {
    /// Full bucket boundary selected by [`BucketLabel`].
    pub timestamp: Timestamp,

    /// Full origin-anchored half-open bucket interval.
    pub interval: TimeInterval,

    /// Portion of the bucket inside the analysis bounds.
    pub selected_interval: TimeInterval,

    /// All observation indices in this bucket, relative to the supplied view.
    pub source_range: Range<usize>,

    /// Selected source observation, or `None` for an empty bucket.
    pub observation: Option<BucketObservation>,
}

impl BucketSelection {
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

/// A session's output label, context, and optional selected observation.
///
/// Empty sessions retain their context with `observation = None`. The original
/// session, including its open periods, is borrowed from the schedule. All
/// other fields are owned independently of the source series and assignment.
/// Envelopes can include breaks; a populated session does not establish
/// sampling completeness.
#[derive(Debug, Clone)]
pub struct SessionSelection<'s> {
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

    /// Selected source observation, or `None` for an empty session.
    pub observation: Option<BucketObservation>,
}

impl SessionSelection<'_> {
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
    /// Selects the earliest observation in each assigned bucket.
    ///
    /// Yields every bucket in chronological order, including empty and partial
    /// buckets. Empty buckets have no observation. The selected observation
    /// keeps its actual timestamp and view-relative index, independently of
    /// the requested bucket label. See [`BucketSelection`] for the shared
    /// result context.
    ///
    /// Iteration uses `O(n + b)` time and `O(1)` additional memory for `n`
    /// source observations and `b` buckets. Results are generated lazily.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::time::Duration;
    /// # use unrvl_timeseries::{SeriesView, TimeInterval, Timestamp};
    /// # use unrvl_timeseries::resample::{BucketLabel, FixedBuckets};
    /// let origin = Timestamp::UNIX_EPOCH;
    /// let end = Timestamp::from_unix_secs(2 * 60)?;
    /// let bounds = TimeInterval::new(origin, end)?;
    /// let buckets = FixedBuckets::new(origin, Duration::from_secs(60), bounds)?;
    ///
    /// let timestamps = [
    ///     Timestamp::from_unix_secs(5)?,
    ///     Timestamp::from_unix_secs(40)?,
    /// ];
    ///
    /// let series = SeriesView::new(&timestamps, &[100.0, 102.0])?;
    /// let assigned = buckets.assign(series)?;
    /// let rows: Vec<_> = assigned.first(BucketLabel::End).collect();
    ///
    /// assert_eq!(rows[0].timestamp, Timestamp::from_unix_secs(60)?);
    ///
    /// let selected = rows[0].observation.unwrap();
    /// assert_eq!(selected.timestamp, timestamps[0]);
    /// assert_eq!(selected.value, 100.0);
    /// assert!(rows[1].observation.is_none());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn first(&self, label: BucketLabel) -> impl Iterator<Item = BucketSelection> + use<'a> {
        select_fixed(self, label, |_| 0)
    }

    /// Selects the latest observation in each assigned bucket.
    ///
    /// Yields every bucket in chronological order, including empty and partial
    /// buckets. Empty buckets have no observation; values are never carried
    /// from another bucket. Labels use the requested full boundary, while the
    /// selected observation retains its actual timestamp and view-relative
    /// source index.
    ///
    /// Iteration uses `O(n + b)` time and `O(1)` additional memory for `n`
    /// source observations and `b` buckets. Results are generated lazily
    /// without allocating an output array.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::time::Duration;
    /// # use unrvl_timeseries::{SeriesView, TimeInterval, Timestamp};
    /// # use unrvl_timeseries::resample::{BucketLabel, FixedBuckets};
    /// let origin = Timestamp::UNIX_EPOCH;
    /// let end = Timestamp::from_unix_secs(2 * 3600)?;
    /// let bounds = TimeInterval::new(origin, end)?;
    /// let buckets = FixedBuckets::new(origin, Duration::from_secs(3600), bounds)?;
    ///
    /// let timestamps = [
    ///     Timestamp::from_unix_secs(5 * 60)?,
    ///     Timestamp::from_unix_secs(40 * 60)?,
    /// ];
    /// let series = SeriesView::new(&timestamps, &[100.0, 102.0])?;
    /// let assigned = buckets.assign(series)?;
    /// let rows: Vec<_> = assigned.last(BucketLabel::End).collect();
    ///
    /// assert_eq!(rows[0].timestamp, Timestamp::from_unix_secs(3600)?);
    ///
    /// let selected = rows[0].observation.unwrap();
    /// assert_eq!(selected.timestamp, timestamps[1]);
    /// assert_eq!(selected.value, 102.0);
    /// assert_eq!(rows[0].observed_count(), 2);
    /// assert!(rows[1].observation.is_none());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn last(&self, label: BucketLabel) -> impl Iterator<Item = BucketSelection> + use<'a> {
        select_fixed(self, label, |values| values.len() - 1)
    }

    /// Selects the observation with the smallest value in each assigned bucket.
    ///
    /// Equal minima select the earliest observation. Positive and negative zero
    /// tie, preserving the original value's sign. Empty buckets have no
    /// selected observation. Labels, partial edges, and source context
    /// follow [`BucketSelection`].
    ///
    /// Uses [`argmin`] on each nonempty bucket. For `n` observations and `b`
    /// buckets, lazy iteration takes `O(n + b)` time and `O(1)` extra memory.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::time::Duration;
    /// # use unrvl_timeseries::{SeriesView, TimeInterval, Timestamp};
    /// # use unrvl_timeseries::resample::{BucketLabel, FixedBuckets};
    /// let origin = Timestamp::UNIX_EPOCH;
    /// let end = Timestamp::from_unix_secs(2 * 60)?;
    /// let bounds = TimeInterval::new(origin, end)?;
    /// let buckets = FixedBuckets::new(origin, Duration::from_secs(60), bounds)?;
    ///
    /// let timestamps = [
    ///     Timestamp::from_unix_secs(5)?,
    ///     Timestamp::from_unix_secs(20)?,
    ///     Timestamp::from_unix_secs(40)?,
    /// ];
    /// let values = [100.0, 98.0, 103.0];
    /// let series = SeriesView::new(&timestamps, &values)?;
    /// let assigned = buckets.assign(series)?;
    /// let lows: Vec<_> = assigned.min(BucketLabel::End).collect();
    /// let highs: Vec<_> = assigned.max(BucketLabel::End).collect();
    ///
    /// let low = lows[0].observation.unwrap();
    /// assert_eq!(low.value, 98.0);
    /// assert_eq!(low.timestamp, timestamps[1]);
    /// assert_eq!(highs[0].observation.unwrap().value, 103.0);
    /// assert!(lows[1].observation.is_none());
    /// assert!(highs[1].observation.is_none());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn min(&self, label: BucketLabel) -> impl Iterator<Item = BucketSelection> + use<'a> {
        select_fixed(self, label, argmin)
    }

    /// Selects the observation with the largest value in each assigned bucket.
    ///
    /// Equal maxima select the earliest observation. Positive and negative zero
    /// tie, preserving the original value's sign. Empty buckets have no
    /// selected observation. Labels, partial edges, and source context
    /// follow [`BucketSelection`].
    ///
    /// Uses [`argmax`] on each nonempty bucket. For `n` observations and `b`
    /// buckets, lazy iteration takes `O(n + b)` time and `O(1)` extra memory.
    /// See [`Self::min`] for an example using both extrema.
    pub fn max(&self, label: BucketLabel) -> impl Iterator<Item = BucketSelection> + use<'a> {
        select_fixed(self, label, argmax)
    }
}

impl<'a, 's> SessionAssignment<'a, 's> {
    /// Selects the earliest observation in each assigned session.
    ///
    /// All selected open periods contribute to the same session result. Empty
    /// sessions have no observation. The selected value retains its original
    /// timestamp and view-relative index, independently of the requested full
    /// session label. See [`SessionSelection`] for the result context.
    ///
    /// Lazy iteration takes `O(n + s)` time and `O(1)` extra memory for `n`
    /// observations and `s` selected sessions. See [`SessionAssignment`] for
    /// an example using session selection and aggregation.
    pub fn first(
        &self,
        label: BucketLabel,
    ) -> impl Iterator<Item = SessionSelection<'s>> + use<'a, 's> {
        select_sessions(self, label, |_| 0)
    }

    /// Selects the latest observation in each assigned session.
    ///
    /// Empty sessions have no observation; values are never carried from
    /// another session. A final-close output label does not imply that the
    /// selected observation occurred at the close or is an official closing
    /// price. Its original timestamp and view-relative index are preserved.
    ///
    /// Lazy iteration takes `O(n + s)` time and `O(1)` extra memory for `n`
    /// observations and `s` selected sessions. See [`SessionAssignment`] for
    /// an example.
    pub fn last(
        &self,
        label: BucketLabel,
    ) -> impl Iterator<Item = SessionSelection<'s>> + use<'a, 's> {
        select_sessions(self, label, |values| values.len() - 1)
    }

    /// Selects the observation with the smallest value in each session.
    ///
    /// Uses [`argmin`] across the session's selected open periods. Equal
    /// minima, including signed-zero ties, select the earliest observation.
    /// Empty sessions have no observation. Labels and partial envelopes follow
    /// [`SessionSelection`].
    ///
    /// Lazy iteration takes `O(n + s)` time and `O(1)` extra memory for `n`
    /// observations and `s` selected sessions.
    ///
    /// # Examples
    ///
    /// Select the session low and high, retaining each observation's original
    /// timestamp even when the output is labeled with the session close.
    ///
    /// ```
    /// # use unrvl_timeseries::{SeriesView, Session, SessionSchedule, TimeInterval, Timestamp};
    /// # use unrvl_timeseries::resample::{BucketLabel, SessionAssignment};
    /// let open = Timestamp::from_unix_secs(9 * 3600)?;
    /// let close = Timestamp::from_unix_secs(16 * 3600)?;
    /// let coverage = TimeInterval::new(open, close)?;
    /// let regular = Session::new("regular", vec![coverage])?;
    /// let schedule = SessionSchedule::new(coverage, vec![regular])?;
    ///
    /// let timestamps = [
    ///     Timestamp::from_unix_secs(10 * 3600)?,
    ///     Timestamp::from_unix_secs(11 * 3600)?,
    ///     Timestamp::from_unix_secs(14 * 3600)?,
    /// ];
    /// let values = [100.0, 105.0, 98.0];
    /// let prices = SeriesView::new(&timestamps, &values)?;
    ///
    /// let assigned = SessionAssignment::new(prices, &schedule, coverage)?;
    /// let low = assigned.min(BucketLabel::End).next().unwrap();
    /// let high = assigned.max(BucketLabel::End).next().unwrap();
    ///
    /// assert_eq!(low.observation.unwrap().value, 98.0);
    /// assert_eq!(high.observation.unwrap().value, 105.0);
    /// assert_eq!(high.observation.unwrap().timestamp, timestamps[1]);
    /// assert_eq!(high.timestamp, close);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn min(
        &self,
        label: BucketLabel,
    ) -> impl Iterator<Item = SessionSelection<'s>> + use<'a, 's> {
        select_sessions(self, label, argmin)
    }

    /// Selects the observation with the largest value in each session.
    ///
    /// Uses [`argmax`] across the session's selected open periods. Equal
    /// maxima, including signed-zero ties, select the earliest observation.
    /// Empty sessions have no observation. Labels and partial envelopes follow
    /// [`SessionSelection`].
    ///
    /// Lazy iteration takes `O(n + s)` time and `O(1)` extra memory for `n`
    /// observations and `s` selected sessions.
    /// See [`Self::min`] for an example using both extrema.
    pub fn max(
        &self,
        label: BucketLabel,
    ) -> impl Iterator<Item = SessionSelection<'s>> + use<'a, 's> {
        select_sessions(self, label, argmax)
    }
}

/// Builds the shared result context around a bucket-relative selection. Empty
/// buckets bypass the selector; otherwise it must return an index within the
/// nonempty value slice.
fn select_fixed<'a, F>(
    assignment: &BucketAssignment<'a>,
    label: BucketLabel,
    select_index: F,
) -> impl Iterator<Item = BucketSelection> + use<'a, F>
where
    F: Fn(&[f64]) -> usize,
{
    assignment.iter().map(move |bucket| {
        let observation = select_observation(
            bucket.observations,
            bucket.source_range.start,
            &select_index,
        );

        BucketSelection {
            timestamp: bucket.label(label),
            interval: bucket.interval,
            selected_interval: bucket.selected_interval,
            source_range: bucket.source_range,
            observation,
        }
    })
}

/// Adds session identity to the selected observation and its bucket context.
fn select_sessions<'a, 's, F>(
    assignment: &SessionAssignment<'a, 's>,
    label: BucketLabel,
    select_index: F,
) -> impl Iterator<Item = SessionSelection<'s>> + use<'a, 's, F>
where
    F: Fn(&[f64]) -> usize,
{
    assignment.iter().map(move |bucket| {
        let observation = select_observation(
            bucket.observations,
            bucket.source_range.start,
            &select_index,
        );

        SessionSelection {
            timestamp: bucket.label(label),
            session_index: bucket.session_index,
            session: bucket.session,
            interval: bucket.interval,
            selected_interval: bucket.selected_interval,
            source_range: bucket.source_range,
            observation,
        }
    })
}

/// Bypasses the selector for empty input; otherwise its index must be within
/// the value slice. The returned identity is relative to the original view.
fn select_observation(
    observations: SeriesView<'_>,
    source_start: usize,
    select_index: impl FnOnce(&[f64]) -> usize,
) -> Option<BucketObservation> {
    let values = observations.values();
    if values.is_empty() {
        return None;
    }

    let offset = select_index(values);

    Some(BucketObservation {
        index: source_start + offset,
        timestamp: observations.timestamps()[offset],
        value: values[offset],
    })
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
        fn selects_the_earliest_observation_in_each_nonempty_bucket() {
            let timestamps = [0, 1, 3, 8, 10].map(timestamp);
            let values = [103.0, 90.0, 150.0, 110.0, 108.0];
            let series = SeriesView::new(&timestamps, &values).unwrap();

            let assigned = buckets(0, 12).assign(series).unwrap();
            let rows: Vec<_> = assigned.first(BucketLabel::Start).collect();
            let observations: Vec<_> = rows.iter().map(|row| row.observation).collect();
            let labels: Vec<_> = rows.iter().map(|row| row.timestamp).collect();

            assert_eq!(
                observations,
                [
                    Some(BucketObservation { index: 0, timestamp: timestamp(0), value: 103.0 }),
                    None,
                    Some(BucketObservation { index: 3, timestamp: timestamp(8), value: 110.0 }),
                ],
            );
            assert_eq!(labels, [0, 4, 8].map(timestamp));
        }

        #[test]
        fn first_uses_subview_indices_and_selects_within_partial_edges() {
            let timestamps = [-1, 1, 2, 4, 5, 8].map(timestamp);
            let values = [-1.0, 11.0, 12.0, 14.0, 15.0, 18.0];
            let original = SeriesView::new(&timestamps, &values).unwrap();
            let series = original.between(interval(1, 6)).unwrap();

            let assigned = buckets(1, 6).assign(series).unwrap();
            let rows: Vec<_> = assigned.first(BucketLabel::End).collect();

            assert_eq!(rows.len(), 2);
            assert_eq!(
                rows[0].observation,
                Some(BucketObservation { index: 0, timestamp: timestamp(1), value: 11.0 }),
            );
            assert_eq!(
                rows[1].observation,
                Some(BucketObservation { index: 2, timestamp: timestamp(4), value: 14.0 }),
            );
            assert!(rows.iter().all(BucketSelection::is_partial));
        }

        #[test]
        fn selects_the_latest_observation_and_preserves_bucket_context() {
            let timestamps = [0, 1, 3, 8, 10].map(timestamp);
            let values = [100.0, 150.0, 103.0, 108.0, 90.0];
            let series = SeriesView::new(&timestamps, &values).unwrap();

            let assigned = buckets(0, 12).assign(series).unwrap();
            let rows: Vec<_> = assigned.last(BucketLabel::End).collect();
            let observations: Vec<_> = rows.iter().map(|row| row.observation).collect();

            assert_eq!(
                observations,
                [
                    Some(BucketObservation { index: 2, timestamp: timestamp(3), value: 103.0 }),
                    None,
                    Some(BucketObservation { index: 4, timestamp: timestamp(10), value: 90.0 }),
                ],
            );

            let cases = [
                (interval(0, 4), 0..3),
                (interval(4, 8), 3..3),
                (interval(8, 12), 3..5),
            ];

            assert_eq!(rows.len(), cases.len());
            for (row, (expected_interval, source_range)) in rows.iter().zip(cases) {
                assert_eq!(row.timestamp, expected_interval.end());
                assert_eq!(row.interval, expected_interval);
                assert_eq!(row.selected_interval, expected_interval);
                assert_eq!(row.observed_count(), source_range.len());
                assert_eq!(row.source_range, source_range);
                assert!(!row.is_partial());
            }
        }

        #[test]
        fn preserves_partial_edges_and_changes_only_the_requested_label() {
            let timestamps = [1, 2, 4].map(timestamp);
            let values = [10.0, 20.0, 40.0];
            let series = SeriesView::new(&timestamps, &values).unwrap();

            let assigned = buckets(1, 6).assign(series).unwrap();
            let starts: Vec<_> = assigned.last(BucketLabel::Start).collect();
            let ends: Vec<_> = assigned.last(BucketLabel::End).collect();
            let cases = [
                (interval(0, 4), interval(1, 4)),
                (interval(4, 8), interval(4, 6)),
            ];

            assert_eq!(starts.len(), cases.len());
            assert_eq!(ends.len(), cases.len());
            for ((start, end), (full, selected)) in starts.iter().zip(&ends).zip(cases) {
                assert_eq!(start.timestamp, full.start());
                assert_eq!(end.timestamp, full.end());
                assert_eq!(end.interval, full);
                assert_eq!(end.selected_interval, selected);
                assert!(start.is_partial());
                assert!(end.is_partial());
                assert_eq!(start.observation, end.observation);
                assert_eq!(start.source_range, end.source_range);
            }
        }

        #[test]
        fn keeps_empty_buckets_distinct_from_observed_zero_without_carrying_values() {
            let timestamps = [5, 13].map(timestamp);
            let series = SeriesView::new(&timestamps, &[0.0, 7.0]).unwrap();

            let assigned = buckets(0, 20).assign(series).unwrap();
            let values: Vec<_> = assigned
                .last(BucketLabel::End)
                .map(|row| row.observation.map(|observation| observation.value))
                .collect();

            assert_eq!(values, [None, Some(0.0), None, Some(7.0), None]);
        }

        #[test]
        fn empty_input_keeps_every_bucket_empty() {
            let series = SeriesView::new(&[], &[]).unwrap();

            let assigned = buckets(0, 8).assign(series).unwrap();
            let rows: Vec<_> = assigned.last(BucketLabel::End).collect();

            assert_eq!(rows.len(), 2);
            for row in rows {
                assert!(row.observation.is_none());
                assert_eq!(row.observed_count(), 0);
            }
        }

        #[test]
        fn selected_indices_are_relative_to_the_supplied_subview() {
            let timestamps = [-1, 0, 1, 4, 8].map(timestamp);
            let values = [-1.0, 10.0, 11.0, 14.0, 18.0];
            let original = SeriesView::new(&timestamps, &values).unwrap();
            let series = original.between(interval(0, 8)).unwrap();

            let assigned = buckets(0, 8).assign(series).unwrap();
            let rows: Vec<_> = assigned.last(BucketLabel::Start).collect();

            assert_eq!(
                rows[0].observation,
                Some(BucketObservation { index: 1, timestamp: timestamp(1), value: 11.0 }),
            );
            assert_eq!(
                rows[1].observation,
                Some(BucketObservation { index: 2, timestamp: timestamp(4), value: 14.0 }),
            );
        }

        #[test]
        fn preserves_exact_timestamps_and_value_bits_across_the_full_span() {
            let bounds = TimeInterval::new(Timestamp::MIN, Timestamp::MAX).unwrap();
            let buckets =
                FixedBuckets::new(Timestamp::MIN, Duration::from_nanos(u64::MAX), bounds).unwrap();
            let last_time = Timestamp::MAX.checked_sub(Duration::from_nanos(1)).unwrap();
            let timestamps = [Timestamp::MIN, last_time];
            let series = SeriesView::new(&timestamps, &[1.0, -0.0]).unwrap();
            let assigned = buckets.assign(series).unwrap();
            let row = assigned.last(BucketLabel::End).next().unwrap();
            let observation = row.observation.unwrap();

            assert_eq!(row.timestamp, Timestamp::MAX);
            assert_eq!(observation.timestamp, last_time);
            assert_eq!(observation.index, 1);
            assert_eq!(observation.value.to_bits(), (-0.0_f64).to_bits());
        }

        #[test]
        fn a_prefix_of_a_large_plan_is_generated_lazily() {
            let bounds = TimeInterval::new(Timestamp::MIN, Timestamp::MAX).unwrap();
            let buckets =
                FixedBuckets::new(Timestamp::UNIX_EPOCH, Duration::from_nanos(1), bounds).unwrap();
            let series = SeriesView::new(&[], &[]).unwrap();
            let assigned = buckets.assign(series).unwrap();
            let rows: Vec<_> = assigned.last(BucketLabel::End).take(2).collect();

            assert_eq!(rows.len(), 2);
            assert_eq!(rows[0].interval.start(), Timestamp::MIN);
            assert_eq!(rows[0].interval.end(), rows[1].interval.start());
            assert!(rows.iter().all(|row| row.observation.is_none()));
        }

        #[test]
        fn selects_minimum_and_maximum_observations_by_value() {
            let timestamps = [0, 1, 2, 3, 8, 9, 10, 11].map(timestamp);
            let values = [1.0, -15.0, 9.0, 3.0, 7.0, 12.0, -14.0, 1.0];
            let series = SeriesView::new(&timestamps, &values).unwrap();

            let assigned = buckets(0, 12).assign(series).unwrap();
            let lows: Vec<_> = assigned
                .min(BucketLabel::End)
                .map(|row| row.observation)
                .collect();
            let highs: Vec<_> = assigned
                .max(BucketLabel::Start)
                .map(|row| row.observation)
                .collect();

            assert_eq!(
                lows,
                [
                    Some(BucketObservation { index: 1, timestamp: timestamp(1), value: -15.0 }),
                    None,
                    Some(BucketObservation { index: 6, timestamp: timestamp(10), value: -14.0 }),
                ],
            );
            assert_eq!(
                highs,
                [
                    Some(BucketObservation { index: 2, timestamp: timestamp(2), value: 9.0 }),
                    None,
                    Some(BucketObservation { index: 5, timestamp: timestamp(9), value: 12.0 }),
                ],
            );
        }

        #[test]
        fn equal_extrema_keep_the_earliest_observation() {
            let timestamps = [0, 1, 2, 3, 4, 5, 6, 7].map(timestamp);
            let values = [4.0, -2.0, -2.0, 4.0, 7.0, 7.0, 7.0, 7.0];
            let series = SeriesView::new(&timestamps, &values).unwrap();

            let assigned = buckets(0, 8).assign(series).unwrap();
            let low_indices: Vec<_> = assigned
                .min(BucketLabel::End)
                .map(|row| row.observation.unwrap().index)
                .collect();
            let high_indices: Vec<_> = assigned
                .max(BucketLabel::End)
                .map(|row| row.observation.unwrap().index)
                .collect();

            assert_eq!(low_indices, [1, 4]);
            assert_eq!(high_indices, [0, 4]);
        }

        #[test]
        fn signed_zero_ties_preserve_the_earliest_observations_bits() {
            let timestamps = [0, 1, 4, 5].map(timestamp);
            let values = [-0.0, 0.0, 0.0, -0.0];
            let series = SeriesView::new(&timestamps, &values).unwrap();

            let assigned = buckets(0, 8).assign(series).unwrap();
            let identity = |row: BucketSelection| {
                let observation = row.observation.unwrap();
                (observation.index, observation.value.to_bits())
            };
            let lows: Vec<_> = assigned.min(BucketLabel::End).map(identity).collect();
            let highs: Vec<_> = assigned.max(BucketLabel::End).map(identity).collect();
            let expected = [(0, (-0.0_f64).to_bits()), (2, 0.0_f64.to_bits())];

            assert_eq!(lows, expected);
            assert_eq!(highs, expected);
        }

        #[test]
        fn extrema_use_subview_indices_inside_partial_buckets() {
            let timestamps = [-1, 1, 2, 4, 5, 8].map(timestamp);
            let values = [-1.0, 11.0, 10.0, 14.0, 15.0, 18.0];
            let original = SeriesView::new(&timestamps, &values).unwrap();
            let series = original.between(interval(1, 6)).unwrap();

            let assigned = buckets(1, 6).assign(series).unwrap();
            let lows: Vec<_> = assigned.min(BucketLabel::End).collect();
            let highs: Vec<_> = assigned.max(BucketLabel::Start).collect();
            let identity = |row: &BucketSelection| {
                let observation = row.observation.unwrap();
                (observation.index, observation.timestamp)
            };

            assert_eq!(
                lows.iter().map(identity).collect::<Vec<_>>(),
                [(1, timestamp(2)), (2, timestamp(4))],
            );
            assert_eq!(
                highs.iter().map(identity).collect::<Vec<_>>(),
                [(0, timestamp(1)), (3, timestamp(5))],
            );
            assert!(lows.iter().chain(&highs).all(BucketSelection::is_partial));
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
        fn first_and_last_keep_session_context_and_original_observations() {
            let schedule = schedule_with_break();

            let timestamps = [1, 2, 5, 7, 12, 13, 15].map(timestamp);
            let values = [103.0, 90.0, 110.0, 95.0, 105.0, 98.0, 109.0];
            let series = SeriesView::new(&timestamps, &values).unwrap();

            let assigned = SessionAssignment::new(series, &schedule, schedule.coverage()).unwrap();
            let first: Vec<_> = assigned
                .first(BucketLabel::End)
                .map(|row| row.observation)
                .collect();
            let last: Vec<_> = assigned.last(BucketLabel::End).collect();
            let last_observations: Vec<_> = last.iter().map(|row| row.observation).collect();

            assert_eq!(
                first,
                [
                    Some(BucketObservation { index: 0, timestamp: timestamp(1), value: 103.0 }),
                    None,
                    Some(BucketObservation { index: 4, timestamp: timestamp(12), value: 105.0 }),
                ],
            );
            assert_eq!(
                last_observations,
                [
                    Some(BucketObservation { index: 3, timestamp: timestamp(7), value: 95.0 }),
                    None,
                    Some(BucketObservation { index: 6, timestamp: timestamp(15), value: 109.0 }),
                ],
            );

            let cases = [
                ("split", interval(1, 8), 0..4),
                ("empty", interval(9, 11), 4..4),
                ("later", interval(12, 16), 4..7),
            ];

            assert_eq!(last.len(), cases.len());
            for (index, (row, (id, envelope, source_range))) in last.iter().zip(cases).enumerate() {
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
        fn extrema_ties_keep_the_earliest_observation_across_open_periods() {
            let schedule = schedule_with_break();

            let timestamps = [1, 2, 5, 7, 12, 13].map(timestamp);
            let values = [4.0, -2.0, -2.0, 4.0, 7.0, 7.0];
            let series = SeriesView::new(&timestamps, &values).unwrap();

            let assigned = SessionAssignment::new(series, &schedule, schedule.coverage()).unwrap();
            let lows: Vec<_> = assigned
                .min(BucketLabel::End)
                .map(|row| row.observation.map(|observation| observation.index))
                .collect();
            let highs: Vec<_> = assigned
                .max(BucketLabel::End)
                .map(|row| row.observation.map(|observation| observation.index))
                .collect();

            assert_eq!(lows, [Some(1), None, Some(4)]);
            assert_eq!(highs, [Some(0), None, Some(4)]);
        }

        #[test]
        fn signed_zero_ties_preserve_the_earliest_observations_bits() {
            let schedule = schedule_with_break();

            let timestamps = [1, 2, 12, 13].map(timestamp);
            let values = [-0.0, 0.0, 0.0, -0.0];
            let series = SeriesView::new(&timestamps, &values).unwrap();

            let assigned = SessionAssignment::new(series, &schedule, schedule.coverage()).unwrap();
            let identity = |row: SessionSelection<'_>| {
                row.observation
                    .map(|observation| (observation.index, observation.value.to_bits()))
            };
            let lows: Vec<_> = assigned.min(BucketLabel::End).map(identity).collect();
            let highs: Vec<_> = assigned.max(BucketLabel::End).map(identity).collect();
            let expected = [
                Some((0, (-0.0_f64).to_bits())),
                None,
                Some((2, 0.0_f64.to_bits())),
            ];

            assert_eq!(lows, expected);
            assert_eq!(highs, expected);
        }

        #[test]
        fn partial_subview_results_outlive_the_source_and_keep_original_indices() {
            let coverage = interval(0, 16);
            let earlier = Session::new("earlier", vec![interval(0, 2)]).unwrap();
            let split = Session::new("split", vec![interval(3, 5), interval(7, 10)]).unwrap();
            let later = Session::new("later", vec![interval(11, 15)]).unwrap();
            let schedule = SessionSchedule::new(coverage, vec![earlier, split, later]).unwrap();
            let bounds = interval(4, 13);
            let (starts, ends): (Vec<_>, Vec<_>) = {
                let timestamps = [0, 4, 7, 9, 11, 12, 14].map(timestamp);
                let values = [-100.0, 120.0, 90.0, 100.0, 80.0, 85.0, 200.0];
                let original = SeriesView::new(&timestamps, &values).unwrap();
                let series = original.between(bounds).unwrap();

                let assigned = SessionAssignment::new(series, &schedule, bounds).unwrap();

                (
                    assigned.max(BucketLabel::Start).collect(),
                    assigned.max(BucketLabel::End).collect(),
                )
            };
            let observations: Vec<_> = ends.iter().map(|row| row.observation).collect();
            let session_indices: Vec<_> = ends.iter().map(|row| row.session_index).collect();

            assert_eq!(session_indices, [1, 2]);
            assert_eq!(
                observations,
                [
                    Some(BucketObservation { index: 0, timestamp: timestamp(4), value: 120.0 }),
                    Some(BucketObservation { index: 4, timestamp: timestamp(12), value: 85.0 }),
                ],
            );

            let cases = [
                (interval(3, 10), interval(4, 10), 0..3),
                (interval(11, 15), interval(11, 13), 3..5),
            ];

            assert_eq!(starts.len(), cases.len());
            assert_eq!(ends.len(), cases.len());
            for ((start, end), (full, selected, source_range)) in
                starts.iter().zip(&ends).zip(cases)
            {
                assert_eq!(start.timestamp, full.start());
                assert_eq!(end.timestamp, full.end());
                assert_eq!(end.interval, full);
                assert_eq!(end.selected_interval, selected);
                assert_eq!(end.source_range, source_range);
                assert_eq!(start.observation, end.observation);
                assert!(start.is_partial());
                assert!(end.is_partial());
            }
        }
    }
}
