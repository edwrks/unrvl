use std::ops::{Bound, RangeBounds};
use std::time::Duration;

use thiserror::Error;

use super::FixedGrid;
use crate::{SeriesView, Timestamp};

/// Invalid analysis bounds or observations that cannot occupy distinct grid
/// slots within those bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum CadenceError {
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

    /// An observation is too far from its nearest grid timestamp.
    #[error(
        "timestamp at index {index} ({actual_nanos} ns) is outside tolerance \
         ({tolerance_nanos} ns) of nearest grid timestamp ({nearest_nanos} ns)",
        actual_nanos = .timestamp.as_unix_nanos(),
        tolerance_nanos = .tolerance.as_nanos(),
    )]
    OutsideTolerance {
        /// Index into the supplied series view.
        index: usize,

        /// Supplied observation timestamp.
        timestamp: Timestamp,

        /// Nearest mathematical grid timestamp in Unix nanoseconds.
        /// Equidistant timestamps use the earlier grid timestamp.
        nearest_nanos: i128,

        /// Inclusive maximum permitted deviation from a grid timestamp.
        tolerance: Duration,
    },

    /// The matching grid timestamp is outside the signed nanosecond range.
    #[error(
        "grid timestamp ({slot_nanos} ns) for observation at index {index} \
         ({actual_nanos} ns) is outside the signed ns range",
        actual_nanos = .timestamp.as_unix_nanos(),
    )]
    SlotOutOfRange {
        /// Index into the supplied series view.
        index: usize,

        /// Supplied observation timestamp.
        timestamp: Timestamp,

        /// Matching mathematical grid timestamp in Unix nanoseconds.
        slot_nanos: i128,
    },

    /// An observation matches a grid label outside the analysis bounds.
    #[error(
        "grid slot ({slot_nanos} ns) for observation at index {index} \
         ({actual_nanos} ns) is outside the analysis bounds",
        slot_nanos = .slot.as_unix_nanos(),
        actual_nanos = .timestamp.as_unix_nanos(),
    )]
    OutsideBounds {
        /// Index into the supplied series view.
        index: usize,

        /// Supplied observation timestamp.
        timestamp: Timestamp,

        /// Matching grid timestamp.
        slot: Timestamp,

        /// Supplied bounds on expected grid labels.
        bounds: (Bound<Timestamp>, Bound<Timestamp>),
    },

    /// Two observations match the same grid timestamp.
    #[error(
        "observations at indices {previous_index} and {index} map to \
         the same grid slot ({slot_nanos} ns)",
        slot_nanos = .slot.as_unix_nanos(),
    )]
    DuplicateSlot {
        /// Index of the preceding observation in the supplied view.
        previous_index: usize,

        /// Index of the second observation in the supplied view.
        index: usize,

        /// Timestamp of the preceding observation.
        previous: Timestamp,

        /// Timestamp of the second observation.
        timestamp: Timestamp,

        /// Grid timestamp matched by both observations.
        slot: Timestamp,
    },
}

/// A supplied observation and its matching grid timestamp.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridObservation {
    /// Index into the supplied series view.
    pub index: usize,

    /// Original observation timestamp.
    pub timestamp: Timestamp,

    /// Original observation value.
    pub value: f64,

    /// Expected grid timestamp matched by this observation.
    pub slot: Timestamp,

    /// Absolute elapsed deviation from the grid timestamp.
    pub deviation: Duration,
}

/// A consecutive run of missing grid slots, with both endpoints included.
///
/// Endpoints are expected observation labels. The gap describes missing
/// samples; it cannot establish what happened between observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotGap {
    /// First missing grid timestamp.
    pub first: Timestamp,

    /// Last missing grid timestamp, equal to `first` for a single missing slot.
    pub last: Timestamp,

    /// Number of missing grid slots, including both endpoints.
    pub count: u128,
}

/// Checked observation-to-grid mapping and missing-slot coverage.
///
/// Borrows the supplied series, retains its original observations, and stores
/// one matching grid label per observation. Missing slots are allowed and
/// reported as consecutive ranges. [`Self::is_complete`] indicates whether
/// every expected slot has an observation.
///
/// Analysis bounds select expected grid labels. An observation may lie just
/// outside those bounds if it matches a label inside them within tolerance.
/// Every supplied observation must match a distinct label inside the bounds.
///
/// Expected and missing counts use `u128` to support all `2^64` labels of a
/// one-nanosecond grid across the full timestamp range without enumerating
/// them.
#[derive(Debug, Clone)]
pub struct GridCoverage<'a> {
    series: SeriesView<'a>,
    grid: FixedGrid,
    bounds: (Bound<Timestamp>, Bound<Timestamp>),
    slots: Vec<Timestamp>,
    expected: u128,
    gaps: Vec<SlotGap>,
}

impl<'a> GridCoverage<'a> {
    /// Maps all supplied observations to distinct grid slots within `bounds`.
    ///
    /// Accepts inclusive, exclusive, and unbounded endpoints through
    /// [`RangeBounds`]. Bounds apply to grid labels, including at tolerance
    /// boundaries. No observations are dropped, filled, or retimestamped.
    /// Missing slots produce a successful result with gaps and counts.
    ///
    /// Empty series and ranges containing no expected slots are valid. An empty
    /// range requires an empty series. Equal endpoints select a grid label
    /// only when both endpoints are inclusive.
    ///
    /// Uses `O(n + g)` time and additional memory for `n` observations and `g`
    /// gap ranges. Expected slots are counted without materializing them.
    ///
    /// # Errors
    ///
    /// Returns [`CadenceError::ReversedRange`] first for reversed bounded
    /// endpoints. Otherwise reports the first invalid observation in input
    /// order: outside tolerance, an unrepresentable grid timestamp, a label
    /// outside the bounds, or a duplicate slot.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::time::Duration;
    /// # use unrvl_timeseries::{SeriesView, Timestamp};
    /// # use unrvl_timeseries::cadence::{FixedGrid, GridCoverage};
    /// let timestamps = [
    ///     Timestamp::from_unix_secs(0)?,
    ///     Timestamp::from_unix_secs(60)?,
    ///     Timestamp::from_unix_secs(180)?,
    /// ];
    ///
    /// let prices = &[100.0, 101.0, 103.0];
    ///
    /// let series = SeriesView::new(&timestamps, prices)?;
    ///
    /// let grid = FixedGrid::new(
    ///     Timestamp::UNIX_EPOCH,
    ///     Duration::from_secs(60),
    ///     Duration::ZERO,
    /// )?;
    ///
    /// let coverage = GridCoverage::new(series, grid, timestamps[0]..=timestamps[2])?;
    ///
    /// assert_eq!(coverage.expected_count(), 4);
    /// assert_eq!(coverage.missing_count(), 1);
    /// assert_eq!(coverage.gaps()[0].first, Timestamp::from_unix_secs(120)?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(
        series: SeriesView<'a>,
        grid: FixedGrid,
        bounds: impl RangeBounds<Timestamp>,
    ) -> Result<Self, CadenceError> {
        let bounds = (bounds.start_bound().cloned(), bounds.end_bound().cloned());

        if let (
            Bound::Included(start) | Bound::Excluded(start),
            Bound::Included(end) | Bound::Excluded(end),
        ) = bounds
            && start > end
        {
            return Err(CadenceError::ReversedRange { start, end });
        }

        let origin = i128::from(grid.origin().as_unix_nanos());

        // Convert losslessly through seconds: even Duration::MAX fits in signed
        // 128-bit nanoseconds.
        let spacing = grid.spacing();
        let spacing =
            i128::from(spacing.as_secs()) * 1_000_000_000 + i128::from(spacing.subsec_nanos());

        let limits = expected_limits(origin, spacing, bounds);
        let expected = limits.map_or(0, |(first, last)| slot_count(first, last, spacing));
        let mut slots = Vec::with_capacity(series.len());

        for (index, &timestamp) in series.timestamps().iter().enumerate() {
            let (slot_nanos, deviation) = nearest_slot(origin, spacing, timestamp);
            if deviation > grid.tolerance().as_nanos() {
                return Err(CadenceError::OutsideTolerance {
                    index,
                    timestamp,
                    nearest_nanos: slot_nanos,
                    tolerance: grid.tolerance(),
                });
            }

            let slot = i64::try_from(slot_nanos)
                .map(Timestamp::from_unix_nanos)
                .map_err(|_| CadenceError::SlotOutOfRange { index, timestamp, slot_nanos })?;

            if !bounds.contains(&slot) {
                return Err(CadenceError::OutsideBounds { index, timestamp, slot, bounds });
            }

            // Increasing source timestamps give nondecreasing slot labels, so
            // any duplicate must be adjacent to its first observation.
            if slots.last() == Some(&slot) {
                let previous_index = index - 1;
                return Err(CadenceError::DuplicateSlot {
                    previous_index,
                    index,
                    previous: series.timestamps()[previous_index],
                    timestamp,
                    slot,
                });
            }

            slots.push(slot);
        }

        let gaps = missing_ranges(&slots, limits, spacing);

        Ok(Self { series, grid, bounds, slots, expected, gaps })
    }

    /// Returns the original observations without allocation or revalidation.
    #[must_use]
    pub const fn series(&self) -> SeriesView<'a> {
        self.series
    }

    /// Returns the sampling grid used for this mapping.
    #[must_use]
    pub const fn grid(&self) -> FixedGrid {
        self.grid
    }

    /// Returns the supplied bounds on expected grid labels.
    #[must_use]
    pub const fn bounds(&self) -> (Bound<Timestamp>, Bound<Timestamp>) {
        self.bounds
    }

    /// Returns the number of supplied observations, each occupying one slot.
    #[must_use]
    pub const fn observed_count(&self) -> usize {
        self.series.len()
    }

    /// Returns the number of grid labels within the analysis bounds.
    #[must_use]
    pub const fn expected_count(&self) -> u128 {
        self.expected
    }

    /// Returns the number of expected slots without an observation.
    #[must_use]
    pub const fn missing_count(&self) -> u128 {
        self.expected - self.observed_count() as u128
    }

    /// Returns whether every expected slot has one observation.
    ///
    /// Returns true when no grid labels fall within the analysis bounds.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.missing_count() == 0
    }

    /// Returns consecutive missing-slot ranges in grid timestamp order.
    #[must_use]
    pub fn gaps(&self) -> &[SlotGap] {
        &self.gaps
    }

    /// Yields observations and their grid labels in source timestamp order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = GridObservation> + '_ {
        self.slots.iter().enumerate().map(|(index, &slot)| {
            let timestamp = self.series.timestamps()[index];
            let deviation =
                Duration::from_nanos(timestamp.as_unix_nanos().abs_diff(slot.as_unix_nanos()));

            GridObservation {
                index,
                timestamp,
                value: self.series.values()[index],
                slot,
                deviation,
            }
        })
    }

    /// Borrows the checked slot labels for internal alignment operations.
    pub(crate) const fn slot_labels(&self) -> &[Timestamp] {
        self.slots.as_slice()
    }
}

/// Returns the nearest grid label in Unix nanoseconds and its absolute
/// deviation in nanoseconds. Requires positive `spacing` and breaks ties toward
/// the earlier label. The label may lie outside [`Timestamp`]'s range; the
/// caller checks it.
///
/// Origin 0, spacing 4, and timestamp -2 give (-4, 2).
fn nearest_slot(origin: i128, spacing: i128, timestamp: Timestamp) -> (i128, u128) {
    let actual = i128::from(timestamp.as_unix_nanos());

    // Euclidean remainder also locates the preceding slot before the origin.
    let remainder = (actual - origin).rem_euclid(spacing);
    let until_next = spacing - remainder;

    if remainder <= until_next {
        (actual - remainder, remainder.unsigned_abs())
    } else {
        (actual + until_next, until_next.unsigned_abs())
    }
}

/// Returns the first and last grid labels selected by `bounds`, both inclusive.
/// Returns `None` when no labels fit. Uses nanosecond units and requires
/// positive `spacing`. Unbounded endpoints use the limits of [`Timestamp`].
///
/// Origin 1, spacing 4, and bounds 0..9 give Some((1, 5)): the selected labels
/// are 1 and 5; the upper bound excludes 9.
fn expected_limits(
    origin: i128,
    spacing: i128,
    bounds: (Bound<Timestamp>, Bound<Timestamp>),
) -> Option<(i128, i128)> {
    // Wider arithmetic permits exclusive endpoints at either timestamp limit.
    let start = match bounds.0 {
        Bound::Included(timestamp) => i128::from(timestamp.as_unix_nanos()),
        Bound::Excluded(timestamp) => i128::from(timestamp.as_unix_nanos()) + 1,
        Bound::Unbounded => i128::from(i64::MIN),
    };

    let end = match bounds.1 {
        Bound::Included(timestamp) => i128::from(timestamp.as_unix_nanos()),
        Bound::Excluded(timestamp) => i128::from(timestamp.as_unix_nanos()) - 1,
        Bound::Unbounded => i128::from(i64::MAX),
    };

    if start > end {
        return None;
    }

    let remainder = (start - origin).rem_euclid(spacing);
    let first = if remainder == 0 {
        start
    } else {
        start + spacing - remainder
    };

    let last = end - (end - origin).rem_euclid(spacing);

    (first <= last).then_some((first, last))
}

/// Counts grid labels from `first` through `last`, including both endpoints.
/// Requires positive spacing and ordered endpoints on the same grid.
///
/// First 1, last 9, and spacing 4 give three labels: 1, 5, and 9.
const fn slot_count(first: i128, last: i128, spacing: i128) -> u128 {
    (last - first).unsigned_abs() / spacing.unsigned_abs() + 1
}

/// Groups missing grid labels into consecutive ranges with inclusive endpoints.
/// Requires positive spacing and strictly increasing occupied `slots` on the
/// grid within the inclusive `limits`. Returns no gaps when `limits` is `None`.
///
/// Limits (1, 13), spacing 4, and occupied slots [1, 13] give one gap from 5 to
/// 9, containing two missing labels.
fn missing_ranges(
    slots: &[Timestamp],
    limits: Option<(i128, i128)>,
    spacing: i128,
) -> Vec<SlotGap> {
    let mut gaps = Vec::new();

    let Some((first, last)) = limits else {
        return gaps;
    };

    let mut next = first;

    for &slot in slots {
        let slot = i128::from(slot.as_unix_nanos());

        if next < slot {
            gaps.push(gap(next, slot - spacing, spacing));
        }

        next = slot + spacing;
    }

    if next <= last {
        gaps.push(gap(next, last, spacing));
    }

    gaps
}

/// Constructs an inclusive gap from ordered, aligned grid endpoints that fit in
/// [`Timestamp`], using the positive grid spacing.
fn gap(first: i128, last: i128, spacing: i128) -> SlotGap {
    SlotGap {
        first: Timestamp::from_unix_nanos(
            i64::try_from(first).expect("bounded grid timestamp fits"),
        ),
        last: Timestamp::from_unix_nanos(i64::try_from(last).expect("bounded grid timestamp fits")),
        count: slot_count(first, last, spacing),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn timestamp(nanos: i64) -> Timestamp {
        Timestamp::from_unix_nanos(nanos)
    }

    fn grid(origin_nanos: i64, spacing_nanos: u64, tolerance_nanos: u64) -> FixedGrid {
        FixedGrid::new(
            timestamp(origin_nanos),
            Duration::from_nanos(spacing_nanos),
            Duration::from_nanos(tolerance_nanos),
        )
        .unwrap()
    }

    mod validation {
        use super::*;

        #[test]
        fn rejects_reversed_bounds_before_inspecting_observations() {
            let empty = SeriesView::new(&[], &[]).unwrap();
            let timestamps = [timestamp(1)];
            let off_grid = SeriesView::new(&timestamps, &[100.0]).unwrap();
            let grid = grid(0, 4, 0);
            let start = timestamp(2);
            let end = timestamp(0);

            for series in [empty, off_grid] {
                let error = GridCoverage::new(series, grid, start..end).unwrap_err();

                assert_eq!(error, CadenceError::ReversedRange { start, end });
            }
        }

        #[test]
        fn reports_the_first_observation_outside_tolerance() {
            let grid = grid(1, 10, 2);
            // 14 is one tick beyond tolerance of 11; 6 is midway between 1 and
            // 11.
            let cases = [(14, 11), (6, 1)];

            for (actual, nearest_nanos) in cases {
                let timestamps = [-9, 1, actual, 26].map(timestamp);
                let series = SeriesView::new(&timestamps, &[100.0; 4]).unwrap();
                let error = GridCoverage::new(series, grid, ..).unwrap_err();

                assert_eq!(
                    error,
                    CadenceError::OutsideTolerance {
                        index: 2,
                        timestamp: timestamp(actual),
                        nearest_nanos,
                        tolerance: Duration::from_nanos(2),
                    },
                );
            }
        }

        #[test]
        fn rejects_a_label_outside_bounds_without_dropping_the_observation() {
            let timestamps = [0, 9].map(timestamp);
            let series = SeriesView::new(&timestamps, &[100.0, 101.0]).unwrap();
            let bounds = (
                Bound::Included(timestamp(0)),
                Bound::Excluded(timestamp(10)),
            );
            let error = GridCoverage::new(series, grid(0, 10, 2), bounds).unwrap_err();

            // The actual timestamp is inside the range, but its slot label is
            // not.
            assert_eq!(
                error,
                CadenceError::OutsideBounds {
                    index: 1,
                    timestamp: timestamp(9),
                    slot: timestamp(10),
                    bounds,
                },
            );
        }

        #[test]
        fn rejects_two_observations_in_the_same_slot_with_both_indices() {
            let timestamps = [-1, 0].map(timestamp);
            let series = SeriesView::new(&timestamps, &[100.0, 101.0]).unwrap();
            let error = GridCoverage::new(series, grid(0, 10, 2), ..).unwrap_err();

            assert_eq!(
                error,
                CadenceError::DuplicateSlot {
                    previous_index: 0,
                    index: 1,
                    previous: timestamp(-1),
                    timestamp: timestamp(0),
                    slot: timestamp(0),
                },
            );

            assert_eq!(
                error.to_string(),
                "observations at indices 0 and 1 map to the same grid slot (0 ns)",
            );
        }

        #[test]
        fn rejects_matching_slots_beyond_either_timestamp_limit() {
            let cases = [
                (Timestamp::MAX, Timestamp::MIN, i128::from(i64::MIN) - 1),
                (Timestamp::MIN, Timestamp::MAX, i128::from(i64::MAX) + 1),
            ];

            for (origin, actual, slot_nanos) in cases {
                let grid = FixedGrid::new(origin, Duration::from_nanos(4), Duration::from_nanos(1))
                    .unwrap();
                let timestamps = [actual];
                let series = SeriesView::new(&timestamps, &[100.0]).unwrap();
                let error = GridCoverage::new(series, grid, ..).unwrap_err();

                assert_eq!(
                    error,
                    CadenceError::SlotOutOfRange { index: 0, timestamp: actual, slot_nanos }
                );
            }
        }
    }

    mod coverage {
        use super::*;

        #[test]
        fn preserves_observations_and_reports_leading_internal_and_trailing_gaps() {
            let timestamps = [-11, 23].map(timestamp);
            let values = [100.0, 200.0];
            let series = SeriesView::new(&timestamps, &values).unwrap();
            let grid = grid(1, 10, 2);
            let bounds = (
                Bound::Included(timestamp(-20)),
                Bound::Included(timestamp(32)),
            );
            let coverage = GridCoverage::new(series, grid, bounds).unwrap();

            assert!(std::ptr::eq(coverage.series().timestamps(), &timestamps));
            assert!(std::ptr::eq(coverage.series().values(), &values));
            assert_eq!(coverage.grid(), grid);
            assert_eq!(coverage.bounds(), bounds);
            assert_eq!(coverage.observed_count(), 2);
            assert_eq!(coverage.expected_count(), 6);
            assert_eq!(coverage.missing_count(), 4);
            assert!(!coverage.is_complete());
            assert_eq!(coverage.iter().len(), 2);

            assert_eq!(
                coverage.iter().collect::<Vec<_>>(),
                [
                    GridObservation {
                        index: 0,
                        timestamp: timestamps[0],
                        value: 100.0,
                        slot: timestamp(-9),
                        deviation: Duration::from_nanos(2),
                    },
                    GridObservation {
                        index: 1,
                        timestamp: timestamps[1],
                        value: 200.0,
                        slot: timestamp(21),
                        deviation: Duration::from_nanos(2),
                    },
                ],
            );
            assert_eq!(
                coverage.gaps(),
                [
                    SlotGap {
                        first: timestamp(-19),
                        last: timestamp(-19),
                        count: 1
                    },
                    SlotGap { first: timestamp(1), last: timestamp(11), count: 2 },
                    SlotGap {
                        first: timestamp(31),
                        last: timestamp(31),
                        count: 1
                    },
                ],
            );
        }

        #[test]
        fn bounds_select_grid_labels_and_allow_jitter_on_both_sides() {
            let timestamps = [-2, 12].map(timestamp);
            let series = SeriesView::new(&timestamps, &[100.0, 101.0]).unwrap();
            let coverage =
                GridCoverage::new(series, grid(0, 10, 2), timestamp(0)..=timestamp(10)).unwrap();

            assert_eq!(coverage.expected_count(), 2);
            assert_eq!(coverage.missing_count(), 0);
            assert!(coverage.is_complete());
            assert_eq!(coverage.gaps(), []);
        }

        #[test]
        fn an_empty_series_reports_all_expected_slots_as_one_gap() {
            let series = SeriesView::new(&[], &[]).unwrap();
            let coverage =
                GridCoverage::new(series, grid(1, 10, 0), timestamp(-15)..timestamp(25)).unwrap();

            assert_eq!(coverage.observed_count(), 0);
            assert_eq!(coverage.expected_count(), 4);
            assert_eq!(coverage.missing_count(), 4);
            assert_eq!(coverage.iter().next(), None);
            assert_eq!(
                coverage.gaps(),
                [SlotGap {
                    first: timestamp(-9),
                    last: timestamp(21),
                    count: 4
                }]
            );
        }

        #[test]
        fn indices_are_relative_to_the_supplied_view() {
            let timestamps = [-10, 0, 10].map(timestamp);
            let series = SeriesView::new(&timestamps, &[90.0, 100.0, 110.0]).unwrap();
            let selected = series.between(timestamp(0)..).unwrap();
            let coverage =
                GridCoverage::new(selected, grid(0, 10, 0), timestamp(0)..=timestamp(10)).unwrap();

            assert_eq!(
                coverage.iter().next(),
                Some(GridObservation {
                    index: 0,
                    timestamp: timestamp(0),
                    value: 100.0,
                    slot: timestamp(0),
                    deviation: Duration::ZERO,
                }),
            );
        }
    }

    mod boundaries {
        use super::*;

        #[test]
        fn nearest_slot_breaks_negative_ties_toward_the_earlier_label() {
            let origin = 0;
            let spacing = 4;
            let observed = timestamp(-2);

            // The observation is halfway between the grid labels -4 and 0.
            assert_eq!(nearest_slot(origin, spacing, observed), (-4, 2));
        }

        #[test]
        fn counts_labels_with_inclusive_and_exclusive_endpoints() {
            let series = SeriesView::new(&[], &[]).unwrap();
            let start = timestamp(-4);
            let end = timestamp(4);
            let cases = [
                ((Bound::Included(start), Bound::Included(end)), 3),
                ((Bound::Included(start), Bound::Excluded(end)), 2),
                ((Bound::Excluded(start), Bound::Included(end)), 2),
                ((Bound::Excluded(start), Bound::Excluded(end)), 1),
            ];

            for (bounds, expected) in cases {
                let coverage = GridCoverage::new(series, grid(0, 4, 0), bounds).unwrap();

                assert_eq!(coverage.expected_count(), expected);
                assert_eq!(coverage.missing_count(), expected);
            }
        }

        #[test]
        fn equal_bounds_select_a_single_label_only_when_both_are_inclusive() {
            let series = SeriesView::new(&[], &[]).unwrap();
            let epoch = Timestamp::UNIX_EPOCH;
            let cases = [
                ((Bound::Included(epoch), Bound::Included(epoch)), 1),
                ((Bound::Included(epoch), Bound::Excluded(epoch)), 0),
                ((Bound::Excluded(epoch), Bound::Included(epoch)), 0),
                ((Bound::Excluded(epoch), Bound::Excluded(epoch)), 0),
            ];

            for (bounds, expected) in cases {
                let coverage = GridCoverage::new(series, grid(0, 4, 0), bounds).unwrap();

                assert_eq!(coverage.expected_count(), expected);
                assert_eq!(coverage.is_complete(), expected == 0);
            }

            let between_labels =
                GridCoverage::new(series, grid(0, 4, 0), timestamp(1)..timestamp(4)).unwrap();
            assert_eq!(between_labels.expected_count(), 0);
            assert_eq!(between_labels.gaps(), []);
        }

        #[test]
        fn excluding_a_timestamp_limit_can_produce_an_empty_range() {
            let series = SeriesView::new(&[], &[]).unwrap();
            let cases = [
                (Bound::Excluded(Timestamp::MAX), Bound::Unbounded),
                (Bound::Unbounded, Bound::Excluded(Timestamp::MIN)),
            ];

            for bounds in cases {
                let coverage = GridCoverage::new(series, grid(0, 1, 0), bounds).unwrap();

                assert_eq!(coverage.expected_count(), 0);
                assert!(coverage.is_complete());
                assert_eq!(coverage.gaps(), []);
            }
        }

        #[test]
        fn reports_the_full_one_nanosecond_grid_without_materializing_its_slots() {
            let series = SeriesView::new(&[], &[]).unwrap();
            let coverage = GridCoverage::new(series, grid(0, 1, 0), ..).unwrap();

            // Both endpoints are included: one more label than u64::MAX.
            let full_count = 1_u128 << 64;

            assert_eq!(coverage.expected_count(), full_count);
            assert_eq!(coverage.missing_count(), full_count);
            assert_eq!(
                coverage.gaps(),
                [SlotGap {
                    first: Timestamp::MIN,
                    last: Timestamp::MAX,
                    count: full_count
                }],
            );
        }

        #[test]
        fn supports_spacing_across_the_full_timestamp_span_and_beyond() {
            let timestamps = [Timestamp::MIN, Timestamp::MAX];
            let series = SeriesView::new(&timestamps, &[100.0, 101.0]).unwrap();
            let spacing = Duration::from_nanos(u64::MAX);
            let grid = FixedGrid::new(Timestamp::MIN, spacing, Duration::ZERO).unwrap();
            let coverage = GridCoverage::new(series, grid, ..).unwrap();

            assert_eq!(coverage.expected_count(), 2);
            assert!(coverage.is_complete());

            // Spacing can exceed the timestamp span; only the origin is in
            // range.
            let series = SeriesView::new(&[Timestamp::UNIX_EPOCH], &[100.0]).unwrap();
            let grid =
                FixedGrid::new(Timestamp::UNIX_EPOCH, Duration::MAX, Duration::ZERO).unwrap();
            let coverage = GridCoverage::new(series, grid, ..).unwrap();

            assert_eq!(coverage.expected_count(), 1);
            assert!(coverage.is_complete());
        }

        #[test]
        fn preserves_adjacent_nanoseconds_beyond_float_precision() {
            let distant = (1_i64 << 53) + 1;
            let timestamps = [distant, distant + 2].map(timestamp);
            let series = SeriesView::new(&timestamps, &[100.0, 102.0]).unwrap();
            let coverage =
                GridCoverage::new(series, grid(distant, 1, 0), timestamps[0]..=timestamps[1])
                    .unwrap();

            assert_eq!(coverage.expected_count(), 3);
            assert_eq!(
                coverage.gaps(),
                [SlotGap {
                    first: timestamp(distant + 1),
                    last: timestamp(distant + 1),
                    count: 1
                }],
            );
        }
    }

    #[test]
    fn agrees_with_direct_matching_and_enumeration_on_a_small_grid() {
        let grid = grid(1, 6, 2);
        let labels = [-17, -11, -5, 1, 7, 13, 19].map(timestamp);
        let cases = [
            vec![],
            vec![-19, -4, 7, 21],
            vec![-17, -12, -3, 0, 9, 13, 18],
        ];

        for nanos in cases {
            let timestamps: Vec<_> = nanos.into_iter().map(timestamp).collect();
            let values = vec![100.0; timestamps.len()];
            let series = SeriesView::new(&timestamps, &values).unwrap();
            let coverage = GridCoverage::new(series, grid, timestamp(-20)..=timestamp(20)).unwrap();

            // Match against explicit labels rather than repeating grid
            // arithmetic.
            let expected_slots: Vec<_> = timestamps
                .iter()
                .map(|actual| {
                    labels
                        .iter()
                        .copied()
                        .find(|label| actual.as_unix_nanos().abs_diff(label.as_unix_nanos()) <= 2)
                        .unwrap()
                })
                .collect();

            let expected_missing: Vec<_> = labels
                .iter()
                .copied()
                .filter(|label| !expected_slots.contains(label))
                .collect();

            let actual_slots: Vec<_> = coverage.iter().map(|row| row.slot).collect();

            let actual_missing: Vec<_> = coverage
                .gaps()
                .iter()
                .flat_map(|gap| (gap.first.as_unix_nanos()..=gap.last.as_unix_nanos()).step_by(6))
                .map(timestamp)
                .collect();

            let gap_count: u128 = coverage.gaps().iter().map(|gap| gap.count).sum();

            assert_eq!(actual_slots, expected_slots);
            assert_eq!(actual_missing, expected_missing);
            assert_eq!(coverage.missing_count(), expected_missing.len() as u128);
            assert_eq!(gap_count, coverage.missing_count());
        }
    }
}
