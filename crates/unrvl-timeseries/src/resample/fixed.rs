use std::ops::Range;
use std::time::Duration;

use thiserror::Error;

use super::BucketLabel;
use crate::{SeriesView, TimeInterval, Timestamp};

/// Invalid bucket geometry or an observation outside the analysis interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum BucketError {
    /// The supplied bucket width is zero.
    #[error("bucket width must be greater than zero")]
    ZeroWidth,

    /// A full bucket boundary cannot be represented as a timestamp.
    #[error("bucket boundary ({boundary_nanos} ns) is outside the signed ns range")]
    BoundaryOutOfRange {
        /// Mathematical boundary in Unix nanoseconds.
        boundary_nanos: i128,
    },

    /// An observation lies outside the selected half-open analysis interval.
    #[error(
        "observation at index {index} ({timestamp_nanos} ns) is outside \
         analysis interval [{start_nanos} ns, {end_nanos} ns)",
        timestamp_nanos = .timestamp.as_unix_nanos(),
        start_nanos = .bounds.start().as_unix_nanos(),
        end_nanos = .bounds.end().as_unix_nanos(),
    )]
    ObservationOutsideBounds {
        /// Index into the supplied series view.
        index: usize,

        /// Supplied observation timestamp.
        timestamp: Timestamp,

        /// Selected analysis interval.
        bounds: TimeInterval,
    },
}

/// One bucket and its observations within the selected analysis interval.
///
/// The full bucket may extend beyond the analysis bounds. `selected_interval`
/// records the portion used for assignment; [`Self::is_partial`] identifies
/// clipped edge buckets. Neither a full interval nor a nonempty sample proves
/// that all expected observations were captured.
#[derive(Debug, Clone)]
pub struct Bucket<'a> {
    /// Full origin-anchored interval, including its start and excluding its
    /// end.
    pub interval: TimeInterval,

    /// Intersection of the full bucket with the analysis bounds.
    pub selected_interval: TimeInterval,

    /// Contiguous observation indices relative to the supplied series view.
    pub source_range: Range<usize>,

    /// Original observations in this bucket, borrowed without revalidation.
    /// The view is empty when no supplied observations fall in the bucket.
    pub observations: SeriesView<'a>,
}

impl Bucket<'_> {
    /// Returns the full bucket boundary chosen by [`BucketLabel`].
    #[must_use]
    pub const fn label(&self, label: BucketLabel) -> Timestamp {
        match label {
            BucketLabel::Start => self.interval.start(),
            BucketLabel::End => self.interval.end(),
        }
    }

    /// Returns whether analysis bounds clip either end of this bucket.
    /// This describes interval geometry, not sampling completeness.
    #[must_use]
    pub fn is_partial(&self) -> bool {
        self.selected_interval != self.interval
    }
}

/// Fixed elapsed-time buckets intersecting a finite analysis interval.
///
/// `origin` anchors boundaries at whole multiples of the positive width before
/// and after it. Each bucket includes its start and excludes its end. Analysis
/// bounds may cut through the first or last bucket; full bucket boundaries
/// must remain representable as [`Timestamp`]s.
///
/// Fixed buckets represent interval membership. For nearest-label matching, see
/// [`FixedGrid`](crate::cadence::FixedGrid). Many observations may occupy one
/// bucket. Widths describe elapsed time, not calendar days or trading sessions.
///
/// # Examples
///
/// ```
/// # use std::time::Duration;
/// # use unrvl_timeseries::{SeriesView, TimeInterval, Timestamp};
/// # use unrvl_timeseries::resample::{BucketLabel, FixedBuckets};
/// let origin = Timestamp::UNIX_EPOCH;
/// let end = Timestamp::from_unix_secs(3 * 3600)?;
/// let bounds = TimeInterval::new(origin, end)?;
/// let buckets = FixedBuckets::new(origin, Duration::from_secs(3600), bounds)?;
///
/// let timestamps = [
///     Timestamp::from_unix_secs(5 * 60)?,
///     Timestamp::from_unix_secs(40 * 60)?,
///     Timestamp::from_unix_secs(2 * 3600 + 10 * 60)?,
/// ];
///
/// let values = [100.0, 102.0, 101.0];
/// let series = SeriesView::new(&timestamps, &values)?;
/// let assigned = buckets.assign(series)?;
///
/// let counts: Vec<_> = assigned
///     .iter()
///     .map(|bucket| bucket.observations.len())
///     .collect();
///
/// assert_eq!(counts, [2, 0, 1]);
///
/// let first = assigned.iter().next().unwrap();
///
/// assert_eq!(first.observations.values(), &[100.0, 102.0]);
/// assert_eq!(
///     first.label(BucketLabel::End),
///     Timestamp::from_unix_secs(3600)?
/// );
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedBuckets {
    origin: Timestamp,
    width: Duration,
    bounds: TimeInterval,
    first_start: Timestamp,
    last_end: Timestamp,
}

impl FixedBuckets {
    /// Creates the buckets that intersect `bounds`, including partial edges.
    ///
    /// Uses exact integer arithmetic, including before the Unix epoch. No
    /// bucket array is allocated. The first bucket is the one containing
    /// `bounds.start()`; a bucket starting at `bounds.end()` is excluded.
    ///
    /// # Errors
    ///
    /// Returns [`BucketError::ZeroWidth`] first for zero width. Otherwise
    /// checks the first full bucket's start and the last full bucket's end,
    /// in that order, for [`BucketError::BoundaryOutOfRange`].
    pub fn new(
        origin: Timestamp,
        width: Duration,
        bounds: TimeInterval,
    ) -> Result<Self, BucketError> {
        if width.is_zero() {
            return Err(BucketError::ZeroWidth);
        }

        let anchor = i128::from(origin.as_unix_nanos());
        let width_nanos = duration_nanos(width);
        let start = i128::from(bounds.start().as_unix_nanos());
        let end = i128::from(bounds.end().as_unix_nanos());

        let first_start = start - (start - anchor).rem_euclid(width_nanos);
        let remainder = (end - anchor).rem_euclid(width_nanos);
        let last_end = if remainder == 0 {
            end
        } else {
            end + width_nanos - remainder
        };

        let first_start = checked_boundary(first_start)?;
        let last_end = checked_boundary(last_end)?;

        Ok(Self { origin, width, bounds, first_start, last_end })
    }

    /// Returns the anchor for bucket boundaries before and after it.
    #[must_use]
    pub const fn origin(&self) -> Timestamp {
        self.origin
    }

    /// Returns the positive elapsed width of each full bucket.
    #[must_use]
    pub const fn width(&self) -> Duration {
        self.width
    }

    /// Returns the selected half-open analysis interval.
    #[must_use]
    pub const fn bounds(&self) -> TimeInterval {
        self.bounds
    }

    /// Returns the number of buckets, including empty and partial buckets.
    ///
    /// Uses `u128` so counts do not depend on platform address width. A
    /// nonempty analysis interval always intersects at least one bucket.
    #[must_use]
    pub fn bucket_count(&self) -> u128 {
        let span = self
            .first_start
            .as_unix_nanos()
            .abs_diff(self.last_end.as_unix_nanos());

        u128::from(span) / self.width.as_nanos()
    }

    /// Borrows observations for assignment.
    ///
    /// Every observation must lie inside `bounds()`. Use
    /// [`SeriesView::between`] for an explicit selection before assignment.
    /// Empty input is valid and leaves every bucket empty.
    ///
    /// Validation uses `O(log n)` time and `O(1)` additional memory. Iterating
    /// the result assigns observations in chronological order.
    ///
    /// # Errors
    ///
    /// Returns [`BucketError::ObservationOutsideBounds`] for the first
    /// out-of-bounds observation in the supplied view, including an observation
    /// exactly at the exclusive analysis end.
    pub fn assign<'a>(&self, series: SeriesView<'a>) -> Result<BucketAssignment<'a>, BucketError> {
        let timestamps = series.timestamps();
        let first_outside = if timestamps
            .first()
            .is_some_and(|&timestamp| timestamp < self.bounds.start())
        {
            0
        } else {
            timestamps.partition_point(|&timestamp| timestamp < self.bounds.end())
        };

        if let Some(&timestamp) = timestamps.get(first_outside) {
            return Err(BucketError::ObservationOutsideBounds {
                index: first_outside,
                timestamp,
                bounds: self.bounds,
            });
        }

        Ok(BucketAssignment { series, buckets: *self })
    }

    /// Forms a full bucket and its nonempty analysis intersection. `start` must
    /// be an anchored boundary from `first_start` up to, but not at,
    /// `last_end`.
    fn intervals_at(&self, start: Timestamp) -> (TimeInterval, TimeInterval) {
        let end = start
            .checked_add(self.width)
            .expect("bucket construction validates the last full boundary");

        let interval = TimeInterval::new(start, end).expect("bucket width is positive");

        let selected =
            TimeInterval::new(start.max(self.bounds.start()), end.min(self.bounds.end()))
                .expect("each configured bucket intersects the analysis interval");

        (interval, selected)
    }
}

/// Observations assigned to checked fixed buckets, including empty buckets.
///
/// Retains the source view and bucket configuration without allocating an
/// index buffer. Each iteration walks the buckets and source observations
/// once, using `O(n + b)` time and `O(1)` additional memory for `n`
/// observations and `b` buckets. Large empty ranges are generated lazily.
/// Iterators can outlive the assignment while the source data remains alive.
#[derive(Debug, Clone, Copy)]
pub struct BucketAssignment<'a> {
    series: SeriesView<'a>,
    buckets: FixedBuckets,
}

impl<'a> BucketAssignment<'a> {
    /// Returns the original supplied series without revalidation.
    #[must_use]
    pub const fn series(&self) -> SeriesView<'a> {
        self.series
    }

    /// Returns the bucket configuration and analysis bounds.
    #[must_use]
    pub const fn buckets(&self) -> FixedBuckets {
        self.buckets
    }

    /// Yields every intersecting bucket in chronological order.
    ///
    /// Observations, including their original timestamps and values, are
    /// borrowed. Source ranges are relative to the view passed to `assign`.
    /// Empty buckets have an empty view and an empty range at the insertion
    /// position. Each supplied observation belongs to exactly one bucket.
    pub fn iter(&self) -> impl Iterator<Item = Bucket<'a>> + use<'a> {
        let series = self.series;
        let buckets = self.buckets;
        let mut start = buckets.first_start;
        let mut next = 0;

        std::iter::from_fn(move || {
            if start == buckets.last_end {
                return None;
            }

            let (interval, selected_interval) = buckets.intervals_at(start);
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
            start = interval.end();

            Some(Bucket {
                interval,
                selected_interval,
                source_range,
                observations,
            })
        })
    }
}

/// Converts a [`Duration`] to signed nanoseconds without loss.
fn duration_nanos(duration: Duration) -> i128 {
    i128::from(duration.as_secs()) * 1_000_000_000 + i128::from(duration.subsec_nanos())
}

/// Converts a nanosecond boundary to a timestamp, rejecting values outside the
/// supported range.
fn checked_boundary(boundary_nanos: i128) -> Result<Timestamp, BucketError> {
    i64::try_from(boundary_nanos)
        .map(Timestamp::from_unix_nanos)
        .map_err(|_| BucketError::BoundaryOutOfRange { boundary_nanos })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{interval, timestamp};

    fn buckets(start_nanos: i64, end_nanos: i64, width_nanos: u64) -> FixedBuckets {
        FixedBuckets::new(
            Timestamp::UNIX_EPOCH,
            Duration::from_nanos(width_nanos),
            interval(start_nanos, end_nanos),
        )
        .unwrap()
    }

    mod construction {
        use super::*;

        #[test]
        fn rejects_zero_width_before_checking_boundaries() {
            let bounds = TimeInterval::new(Timestamp::MIN, Timestamp::MAX).unwrap();
            let error =
                FixedBuckets::new(Timestamp::UNIX_EPOCH, Duration::ZERO, bounds).unwrap_err();

            assert_eq!(error, BucketError::ZeroWidth);
        }

        #[test]
        fn preserves_configuration_and_counts_partial_buckets() {
            let origin = timestamp(1);
            let width = Duration::from_nanos(4);
            let bounds = interval(0, 10);
            let buckets = FixedBuckets::new(origin, width, bounds).unwrap();

            assert_eq!(buckets.origin(), origin);
            assert_eq!(buckets.width(), width);
            assert_eq!(buckets.bounds(), bounds);

            // Full buckets are [-3, 1), [1, 5), [5, 9), and [9, 13).
            assert_eq!(buckets.bucket_count(), 4);
        }

        #[test]
        fn rejects_an_unrepresentable_first_boundary() {
            let start = Timestamp::MIN;
            let origin = start.checked_add(Duration::from_nanos(1)).unwrap();
            let bounds = TimeInterval::new(start, origin).unwrap();
            let error = FixedBuckets::new(origin, Duration::from_nanos(2), bounds).unwrap_err();

            // The preceding anchored boundary is one nanosecond below MIN.
            assert_eq!(
                error,
                BucketError::BoundaryOutOfRange { boundary_nanos: i128::from(i64::MIN) - 1 },
            );
        }

        #[test]
        fn rejects_an_unrepresentable_last_boundary() {
            let end = Timestamp::MAX;
            let origin = end.checked_sub(Duration::from_nanos(1)).unwrap();
            let bounds = TimeInterval::new(origin, end).unwrap();
            let error = FixedBuckets::new(origin, Duration::from_nanos(2), bounds).unwrap_err();

            // The full bucket would end one nanosecond above MAX.
            assert_eq!(
                error,
                BucketError::BoundaryOutOfRange { boundary_nanos: i128::from(i64::MAX) + 1 },
            );
        }

        #[test]
        fn rejects_duration_max_without_integer_overflow() {
            let error = FixedBuckets::new(Timestamp::UNIX_EPOCH, Duration::MAX, interval(0, 1))
                .unwrap_err();

            assert_eq!(
                error,
                BucketError::BoundaryOutOfRange {
                    boundary_nanos: i128::try_from(Duration::MAX.as_nanos()).unwrap(),
                },
            );
        }
    }

    mod assignment {
        use super::*;

        #[test]
        fn groups_multiple_observations_and_retains_an_empty_bucket() {
            let timestamps = [0, 1, 3, 8, 9, 11].map(timestamp);
            let values = [10.0, 11.0, 13.0, 18.0, 19.0, 21.0];
            let series = SeriesView::new(&timestamps, &values).unwrap();
            let buckets = buckets(0, 12, 4);
            let assigned = buckets.assign(series).unwrap();
            let rows: Vec<_> = assigned.iter().collect();

            assert_eq!(assigned.buckets(), buckets);
            assert_eq!(assigned.series().timestamps(), timestamps);
            assert_eq!(rows.len(), 3);

            let cases = [
                (interval(0, 4), 0..3),
                (interval(4, 8), 3..3),
                (interval(8, 12), 3..6),
            ];

            for (bucket, (expected_interval, source_range)) in rows.iter().zip(cases) {
                assert_eq!(bucket.interval, expected_interval);
                assert_eq!(bucket.selected_interval, expected_interval);
                assert_eq!(bucket.source_range, source_range);
                assert_eq!(bucket.observations.values(), &values[source_range]);
                assert!(!bucket.is_partial());
            }
        }

        #[test]
        fn shared_boundaries_belong_to_the_later_bucket() {
            let timestamps = [0, 4].map(timestamp);
            let series = SeriesView::new(&timestamps, &[10.0, 20.0]).unwrap();

            let assigned = buckets(0, 8, 4).assign(series).unwrap();
            let ranges: Vec<_> = assigned.iter().map(|bucket| bucket.source_range).collect();

            assert_eq!(ranges, [0..1, 1..2]);
        }

        #[test]
        fn empty_input_leaves_every_bucket_empty() {
            let series = SeriesView::new(&[], &[]).unwrap();

            let assigned = buckets(0, 12, 4).assign(series).unwrap();
            let rows: Vec<_> = assigned.iter().collect();

            assert_eq!(rows.len(), 3);
            for bucket in rows {
                assert_eq!(bucket.source_range, 0..0);
                assert!(bucket.observations.is_empty());
            }
        }

        #[test]
        fn empty_edge_buckets_keep_the_source_insertion_position() {
            let timestamps = [timestamp(4)];
            let series = SeriesView::new(&timestamps, &[100.0]).unwrap();

            let assigned = buckets(0, 12, 4).assign(series).unwrap();
            let ranges: Vec<_> = assigned.iter().map(|bucket| bucket.source_range).collect();

            assert_eq!(ranges, [0..0, 0..1, 1..1]);
        }

        #[test]
        fn rejects_the_first_observation_outside_analysis_bounds() {
            let bounds = interval(1, 9);
            let buckets = buckets(1, 9, 4);
            let cases = [
                ([0, 1, 9], 0, timestamp(0)),
                ([1, 4, 9], 2, timestamp(9)),
                ([9, 10, 11], 0, timestamp(9)),
            ];

            for (nanos, index, ts) in cases {
                let timestamps = nanos.map(timestamp);
                let values = [1.0; 3];
                let series = SeriesView::new(&timestamps, &values).unwrap();
                let error = buckets.assign(series).unwrap_err();

                assert_eq!(
                    error,
                    BucketError::ObservationOutsideBounds { index, timestamp: ts, bounds },
                );
            }
        }

        #[test]
        fn borrows_a_subview_with_indices_relative_to_that_view() {
            use std::ptr::eq;

            let timestamps = [-1, 0, 3, 8, 12].map(timestamp);
            let values = [-10.0, -0.0, 3.0, 8.0, 12.0];
            let original = SeriesView::new(&timestamps, &values).unwrap();
            let series = original.between(interval(0, 12)).unwrap();

            let assigned = buckets(0, 12, 4).assign(series).unwrap();
            let rows: Vec<_> = assigned.iter().collect();
            let first_timestamps = &timestamps[1..3];
            let first_values = &values[1..3];
            let last_values = &values[3..4];

            assert_eq!(rows[0].source_range, 0..2);
            assert_eq!(rows[2].source_range, 2..3);
            assert!(eq(rows[0].observations.timestamps(), first_timestamps));
            assert!(eq(rows[0].observations.values(), first_values));
            assert!(eq(rows[2].observations.values(), last_values));
        }
    }

    mod geometry {
        use super::*;

        #[test]
        fn preserves_full_labels_and_selected_intervals_at_partial_edges() {
            let series = SeriesView::new(&[], &[]).unwrap();

            let assigned = buckets(1, 9, 4).assign(series).unwrap();
            let rows: Vec<_> = assigned.iter().collect();
            let cases = [
                (interval(0, 4), interval(1, 4), true),
                (interval(4, 8), interval(4, 8), false),
                (interval(8, 12), interval(8, 9), true),
            ];

            assert_eq!(rows.len(), cases.len());
            for (bucket, (full, selected, partial)) in rows.iter().zip(cases) {
                assert_eq!(bucket.interval, full);
                assert_eq!(bucket.selected_interval, selected);
                assert_eq!(bucket.is_partial(), partial);
                assert_eq!(bucket.label(BucketLabel::Start), full.start());
                assert_eq!(bucket.label(BucketLabel::End), full.end());
            }
        }

        #[test]
        fn both_analysis_edges_can_fall_within_one_bucket() {
            let timestamps = [timestamp(1)];
            let series = SeriesView::new(&timestamps, &[100.0]).unwrap();

            let assigned = buckets(1, 2, 4).assign(series).unwrap();
            let rows: Vec<_> = assigned.iter().collect();

            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].interval, interval(0, 4));
            assert_eq!(rows[0].selected_interval, interval(1, 2));
            assert_eq!(rows[0].source_range, 0..1);
            assert!(rows[0].is_partial());
        }

        #[test]
        fn anchors_buckets_before_the_epoch_using_floor_division() {
            let timestamps = [-5, -4, -3, 0, 1, 4].map(timestamp);
            let values = [1.0; 6];
            let series = SeriesView::new(&timestamps, &values).unwrap();
            let buckets =
                FixedBuckets::new(timestamp(1), Duration::from_nanos(4), interval(-5, 5)).unwrap();
            let assigned = buckets.assign(series).unwrap();
            let actual: Vec<_> = assigned
                .iter()
                .map(|bucket| (bucket.interval, bucket.source_range))
                .collect();

            assert_eq!(
                actual,
                [
                    (interval(-7, -3), 0..2),
                    (interval(-3, 1), 2..4),
                    (interval(1, 5), 4..6)
                ],
            );
        }

        #[test]
        fn shifting_the_origin_by_whole_widths_preserves_membership() {
            let series = SeriesView::new(&[], &[]).unwrap();
            let bounds = interval(-5, 9);
            let intervals = |origin| {
                let buckets = FixedBuckets::new(origin, Duration::from_nanos(4), bounds).unwrap();
                buckets
                    .assign(series)
                    .unwrap()
                    .iter()
                    .map(|bucket| bucket.interval)
                    .collect::<Vec<_>>()
            };

            assert_eq!(intervals(timestamp(0)), intervals(timestamp(12)));
        }
    }

    mod boundaries {
        use super::*;

        #[test]
        fn one_bucket_can_span_the_full_timestamp_range() {
            let bounds = TimeInterval::new(Timestamp::MIN, Timestamp::MAX).unwrap();
            let width = Duration::from_nanos(u64::MAX);
            let buckets = FixedBuckets::new(Timestamp::MIN, width, bounds).unwrap();
            let timestamps = [Timestamp::MIN, Timestamp::UNIX_EPOCH];
            let series = SeriesView::new(&timestamps, &[1.0, 2.0]).unwrap();
            let assigned = buckets.assign(series).unwrap();
            let rows: Vec<_> = assigned.iter().collect();

            assert_eq!(buckets.bucket_count(), 1);
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].interval, bounds);
            assert_eq!(rows[0].source_range, 0..2);
            assert!(!rows[0].is_partial());
        }

        #[test]
        fn a_full_span_nanosecond_plan_does_not_allocate_or_enumerate_buckets() {
            let bounds = TimeInterval::new(Timestamp::MIN, Timestamp::MAX).unwrap();
            let buckets =
                FixedBuckets::new(Timestamp::UNIX_EPOCH, Duration::from_nanos(1), bounds).unwrap();
            let series = SeriesView::new(&[], &[]).unwrap();
            let assigned = buckets.assign(series).unwrap();
            let first_two: Vec<_> = assigned.iter().take(2).collect();

            // The half-open interval spans 2^64 - 1 nanoseconds.
            assert_eq!(buckets.bucket_count(), u128::from(u64::MAX));
            assert_eq!(first_two.len(), 2);
            assert_eq!(first_two[0].interval.start(), Timestamp::MIN);
            assert_eq!(first_two[0].interval.end(), first_two[1].interval.start());
            assert!(
                first_two
                    .iter()
                    .all(|bucket| bucket.observations.is_empty())
            );
        }

        #[test]
        fn iteration_stops_at_the_last_representable_boundary() {
            let end = Timestamp::MAX;
            let start = end.checked_sub(Duration::from_nanos(1)).unwrap();
            let bounds = TimeInterval::new(start, end).unwrap();
            let buckets = FixedBuckets::new(start, Duration::from_nanos(1), bounds).unwrap();
            let series = SeriesView::new(&[], &[]).unwrap();
            let assigned = buckets.assign(series).unwrap();
            let mut iter = assigned.iter();

            assert_eq!(iter.next().unwrap().interval, bounds);
            assert!(iter.next().is_none());
            assert!(iter.next().is_none());
        }
    }

    #[test]
    fn small_ranges_match_explicit_integer_bucket_enumeration() {
        let cases = [
            // (origin, width, analysis start, analysis end), all in
            // nanoseconds.
            (0, 1, -3, 3),
            (0, 4, 0, 8),
            (1, 4, -5, 5),
            (-2, 3, -4, 6),
            (3, 5, -5, 4),
            (0, 5, 1, 2),
        ];

        for (origin, width, start, end) in cases {
            let bounds = interval(start, end);
            let buckets = FixedBuckets::new(
                timestamp(origin),
                Duration::from_nanos(u64::try_from(width).unwrap()),
                bounds,
            )
            .unwrap();

            let timestamps: Vec<_> = (start..end)
                .filter(|nanos| nanos.rem_euclid(3) != 1)
                .map(timestamp)
                .collect();

            let values = vec![1.0; timestamps.len()];
            let series = SeriesView::new(&timestamps, &values).unwrap();
            let assigned = buckets.assign(series).unwrap();
            let actual: Vec<_> = assigned.iter().collect();

            // Enumerate anchored intervals directly, without division or
            // nearest-slot arithmetic. These multiples cover every case above.
            let expected: Vec<_> = (-10..=10)
                .map(|multiple| origin + multiple * width)
                .filter(|&first| first < end && first + width > start)
                .map(|first| interval(first, first + width))
                .collect();

            assert_eq!(
                actual.len(),
                expected.len(),
                "origin={origin}, width={width}"
            );

            assert_eq!(buckets.bucket_count(), expected.len() as u128);

            for (bucket, full) in actual.iter().zip(expected) {
                let selected = TimeInterval::new(
                    full.start().max(bounds.start()),
                    full.end().min(bounds.end()),
                )
                .unwrap();

                let first = timestamps
                    .iter()
                    .filter(|&&time| time < selected.start())
                    .count();

                let count = timestamps
                    .iter()
                    .filter(|&&time| selected.contains(time))
                    .count();

                assert_eq!(bucket.interval, full);
                assert_eq!(bucket.selected_interval, selected);
                assert_eq!(bucket.source_range, first..first + count);
                assert_eq!(
                    bucket.observations.timestamps(),
                    &timestamps[first..first + count]
                );
            }
        }
    }
}
