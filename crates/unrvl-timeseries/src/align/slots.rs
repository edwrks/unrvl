use std::time::Duration;

use thiserror::Error;

use super::merge::matching_indices;
use crate::cadence::GridCoverage;
use crate::{SeriesView, Timestamp};

/// Sampling grids that cannot be aligned by shared slot labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SlotAlignmentError {
    /// The grids have different elapsed-time spacing.
    #[error(
        "grid spacings must match, got left={left_nanos} ns and right={right_nanos} ns",
        left_nanos = .left.as_nanos(),
        right_nanos = .right.as_nanos(),
    )]
    SpacingMismatch {
        /// Spacing of the left grid.
        left: Duration,

        /// Spacing of the right grid.
        right: Duration,
    },

    /// The origins are not separated by a whole number of grid steps.
    #[error(
        "grid origins must differ by whole multiples of {spacing_nanos} ns, \
         got left={left_nanos} ns and right={right_nanos} ns",
        spacing_nanos = .spacing.as_nanos(),
        left_nanos = .left_origin.as_unix_nanos(),
        right_nanos = .right_origin.as_unix_nanos(),
    )]
    PhaseMismatch {
        /// Origin of the left grid.
        left_origin: Timestamp,

        /// Origin of the right grid.
        right_origin: Timestamp,

        /// Shared spacing between expected timestamps.
        spacing: Duration,
    },
}

/// Two observations matched to the same expected grid timestamp.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlotPair {
    /// Grid label shared by both observations.
    pub slot: Timestamp,

    /// Index into the supplied left series view.
    pub left_index: usize,

    /// Original timestamp from the left series.
    pub left_timestamp: Timestamp,

    /// Original value from the left series.
    pub left_value: f64,

    /// Absolute deviation of the left timestamp from the shared slot label.
    pub left_deviation: Duration,

    /// Index into the supplied right series view.
    pub right_index: usize,

    /// Original timestamp from the right series.
    pub right_timestamp: Timestamp,

    /// Original value from the right series.
    pub right_value: f64,

    /// Absolute deviation of the right timestamp from the shared slot label.
    pub right_deviation: Duration,
}

/// Result of matching occupied grid slots in two checked series.
///
/// Borrows the source series and stores one slot label and index pair per
/// match. The coverage inputs are needed only during construction. An
/// alignment with no shared occupied slots is valid.
#[derive(Debug, Clone)]
pub struct SlotAlignment<'left, 'right> {
    left: SeriesView<'left>,
    right: SeriesView<'right>,
    matches: Vec<Match>,
}

impl SlotAlignment<'_, '_> {
    /// Returns the number of matched pairs.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.matches.len()
    }

    /// Returns whether there are no shared occupied slots.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.matches.is_empty()
    }

    /// Returns the number of unmatched observations in the supplied left view.
    #[must_use]
    pub const fn unmatched_left(&self) -> usize {
        self.left.len() - self.len()
    }

    /// Returns the number of unmatched observations in the supplied right view.
    #[must_use]
    pub const fn unmatched_right(&self) -> usize {
        self.right.len() - self.len()
    }

    /// Yields matched pairs in grid timestamp order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = SlotPair> + '_ {
        self.matches.iter().map(|matched| {
            let left_timestamp = self.left.timestamps()[matched.left_index];
            let right_timestamp = self.right.timestamps()[matched.right_index];
            let slot_nanos = matched.slot.as_unix_nanos();

            SlotPair {
                slot: matched.slot,
                left_index: matched.left_index,
                left_timestamp,
                left_value: self.left.values()[matched.left_index],
                left_deviation: Duration::from_nanos(
                    left_timestamp.as_unix_nanos().abs_diff(slot_nanos),
                ),
                right_index: matched.right_index,
                right_timestamp,
                right_value: self.right.values()[matched.right_index],
                right_deviation: Duration::from_nanos(
                    right_timestamp.as_unix_nanos().abs_diff(slot_nanos),
                ),
            }
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct Match {
    slot: Timestamp,
    left_index: usize,
    right_index: usize,
}

/// Pairs observations whose checked grid labels are equal.
///
/// Grids must have the same spacing and origins separated by a whole number
/// of grid steps. Each source may use its own tolerance and analysis bounds.
/// A match lies within both inputs' bounds; unmatched counts include all
/// supplied observations, including labels outside the other input's bounds.
///
/// Preserves both source timestamps, values, view-relative indices, and
/// deviations from the shared label. Observations may be earlier or later than
/// that label; information availability is assessed separately. Missing slots
/// and an empty intersection are valid.
///
/// Uses `O(n + m)` time for `n` left and `m` right observations, and `O(k)`
/// additional memory for `k` matches. Source data and coverage are borrowed
/// without copying or revalidation.
///
/// # Errors
///
/// Returns [`SlotAlignmentError::SpacingMismatch`] first if the grid spacings
/// differ. Otherwise returns [`SlotAlignmentError::PhaseMismatch`] when the
/// origins do not describe the same grid. Compatibility is checked even when
/// either source is empty.
///
/// # Examples
///
/// ```
/// # use std::time::Duration;
/// # use unrvl_timeseries::{SeriesView, Timestamp, align};
/// # use unrvl_timeseries::cadence::{FixedGrid, GridCoverage};
/// let left_times = [Timestamp::from_unix_secs(59)?];
/// let right_times = [Timestamp::from_unix_secs(61)?];
///
/// let left = SeriesView::new(&left_times, &[100.0])?;
/// let right = SeriesView::new(&right_times, &[101.0])?;
///
/// let grid = FixedGrid::new(
///     Timestamp::UNIX_EPOCH,
///     Duration::from_secs(60),
///     Duration::from_secs(2),
/// )?;
///
/// let label = Timestamp::from_unix_secs(60)?;
///
/// let left = GridCoverage::new(left, grid, label..=label)?;
/// let right = GridCoverage::new(right, grid, label..=label)?;
///
/// let aligned = align::inner_slots(&left, &right)?;
/// let pair = aligned.iter().next().unwrap();
///
/// assert_eq!(pair.slot, label);
/// assert_eq!(pair.left_timestamp, left_times[0]);
/// assert_eq!(pair.right_timestamp, right_times[0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn inner_slots<'left, 'right>(
    left: &GridCoverage<'left>,
    right: &GridCoverage<'right>,
) -> Result<SlotAlignment<'left, 'right>, SlotAlignmentError> {
    let left_grid = left.grid();
    let right_grid = right.grid();

    if left_grid.spacing() != right_grid.spacing() {
        return Err(SlotAlignmentError::SpacingMismatch {
            left: left_grid.spacing(),
            right: right_grid.spacing(),
        });
    }

    let origin_distance = u128::from(
        left_grid
            .origin()
            .as_unix_nanos()
            .abs_diff(right_grid.origin().as_unix_nanos()),
    );

    if !origin_distance.is_multiple_of(left_grid.spacing().as_nanos()) {
        return Err(SlotAlignmentError::PhaseMismatch {
            left_origin: left_grid.origin(),
            right_origin: right_grid.origin(),
            spacing: left_grid.spacing(),
        });
    }

    let left_slots = left.slot_labels();
    let right_slots = right.slot_labels();
    let matches = matching_indices(left_slots, right_slots)
        .map(|(left_index, right_index)| Match {
            slot: left_slots[left_index],
            left_index,
            right_index,
        })
        .collect();

    Ok(SlotAlignment {
        left: left.series(),
        right: right.series(),
        matches,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cadence::FixedGrid;

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

    #[test]
    fn rejects_different_spacing_before_checking_phase_even_for_empty_sources() {
        let empty = SeriesView::new(&[], &[]).unwrap();
        let left = GridCoverage::new(empty, grid(0, 10, 0), ..).unwrap();
        let right = GridCoverage::new(empty, grid(1, 20, 0), ..).unwrap();
        let error = inner_slots(&left, &right).unwrap_err();

        assert_eq!(
            error,
            SlotAlignmentError::SpacingMismatch {
                left: Duration::from_nanos(10),
                right: Duration::from_nanos(20),
            },
        );
        assert_eq!(
            error.to_string(),
            "grid spacings must match, got left=10 ns and right=20 ns",
        );
    }

    #[test]
    fn rejects_different_phases_including_spacing_beyond_the_timestamp_span() {
        let empty = SeriesView::new(&[], &[]).unwrap();
        let left_origin = timestamp(0);
        let right_origin = timestamp(1);

        for spacing in [Duration::from_nanos(10), Duration::MAX] {
            let left_grid = FixedGrid::new(left_origin, spacing, Duration::ZERO).unwrap();
            let right_grid = FixedGrid::new(right_origin, spacing, Duration::ZERO).unwrap();
            let left = GridCoverage::new(empty, left_grid, ..).unwrap();
            let right = GridCoverage::new(empty, right_grid, ..).unwrap();
            let error = inner_slots(&left, &right).unwrap_err();

            assert_eq!(
                error,
                SlotAlignmentError::PhaseMismatch { left_origin, right_origin, spacing }
            );
        }
    }

    #[test]
    fn pairs_equivalent_grid_labels_and_preserves_both_sources() {
        let left_times = [-9, 1, 19, 31].map(timestamp);
        let right_times = [-1, 11, 20, 39, 51].map(timestamp);
        let left = SeriesView::new(&left_times, &[91.0, 101.0, 119.0, 131.0]).unwrap();
        let right = SeriesView::new(&right_times, &[200.0, 210.0, 220.0, 240.0, 250.0]).unwrap();

        // Origins differ by three steps; tolerances and analysis ranges differ.
        let left =
            GridCoverage::new(left, grid(20, 10, 2), timestamp(-10)..=timestamp(30)).unwrap();

        let right =
            GridCoverage::new(right, grid(-10, 10, 1), timestamp(0)..=timestamp(50)).unwrap();
        let aligned = inner_slots(&left, &right).unwrap();

        assert_eq!(aligned.len(), 2);
        assert_eq!(aligned.iter().len(), 2);
        assert_eq!(aligned.unmatched_left(), 2);
        assert_eq!(aligned.unmatched_right(), 3);
        assert_eq!(
            aligned.iter().collect::<Vec<_>>(),
            [
                SlotPair {
                    slot: timestamp(0),
                    left_index: 1,
                    left_timestamp: timestamp(1),
                    left_value: 101.0,
                    left_deviation: Duration::from_nanos(1),
                    right_index: 0,
                    right_timestamp: timestamp(-1),
                    right_value: 200.0,
                    right_deviation: Duration::from_nanos(1),
                },
                SlotPair {
                    slot: timestamp(20),
                    left_index: 2,
                    left_timestamp: timestamp(19),
                    left_value: 119.0,
                    left_deviation: Duration::from_nanos(1),
                    right_index: 2,
                    right_timestamp: timestamp(20),
                    right_value: 220.0,
                    right_deviation: Duration::ZERO,
                },
            ],
        );
    }

    #[test]
    fn empty_or_disjoint_coverage_produces_an_empty_alignment() {
        let grid = grid(0, 10, 0);
        let empty = SeriesView::new(&[], &[]).unwrap();
        let left_times = [timestamp(0)];
        let right_times = [timestamp(10)];
        let left = SeriesView::new(&left_times, &[100.0]).unwrap();
        let right = SeriesView::new(&right_times, &[110.0]).unwrap();
        let empty = GridCoverage::new(empty, grid, timestamp(0)..=timestamp(10)).unwrap();
        let left = GridCoverage::new(left, grid, timestamp(0)..=timestamp(0)).unwrap();
        let right = GridCoverage::new(right, grid, timestamp(10)..=timestamp(10)).unwrap();
        let cases = [
            (&empty, &empty),
            (&empty, &right),
            (&left, &empty),
            (&left, &right),
            (&right, &left),
        ];

        for (left, right) in cases {
            let aligned = inner_slots(left, right).unwrap();

            assert!(aligned.is_empty());
            assert_eq!(aligned.iter().next(), None);
            assert_eq!(aligned.unmatched_left(), left.observed_count());
            assert_eq!(aligned.unmatched_right(), right.observed_count());
        }
    }

    #[test]
    fn indices_are_relative_to_the_supplied_series_views() {
        let left_times = [-11, -1, 11].map(timestamp);
        let right_times = [-1, 9].map(timestamp);

        let left = SeriesView::new(&left_times, &[90.0, 100.0, 120.0]).unwrap();
        let right = SeriesView::new(&right_times, &[200.0, 210.0]).unwrap();

        let left = left.between(timestamp(-1)..).unwrap();
        let right = right.between(timestamp(0)..).unwrap();

        let grid = grid(0, 10, 1);
        let left = GridCoverage::new(left, grid, timestamp(0)..=timestamp(10)).unwrap();
        let right = GridCoverage::new(right, grid, timestamp(0)..=timestamp(10)).unwrap();

        let aligned = inner_slots(&left, &right).unwrap();
        let pair = aligned.iter().next().unwrap();

        assert_eq!((pair.left_index, pair.right_index), (1, 0));
        assert_eq!(
            (pair.left_timestamp, pair.right_timestamp),
            (timestamp(11), timestamp(9))
        );
        assert_eq!((pair.left_value, pair.right_value), (120.0, 210.0));
    }

    #[test]
    fn recognizes_equivalent_origins_across_the_full_timestamp_span() {
        let timestamps = [Timestamp::MIN, Timestamp::MAX];

        let left = SeriesView::new(&timestamps, &[100.0, 101.0]).unwrap();
        let right = SeriesView::new(&timestamps, &[200.0, 201.0]).unwrap();

        let spacing = Duration::from_nanos(u64::MAX);
        let left_grid = FixedGrid::new(Timestamp::MIN, spacing, Duration::ZERO).unwrap();
        let right_grid = FixedGrid::new(Timestamp::MAX, spacing, Duration::ZERO).unwrap();

        let left = GridCoverage::new(left, left_grid, ..).unwrap();
        let right = GridCoverage::new(right, right_grid, ..).unwrap();

        let aligned = inner_slots(&left, &right).unwrap();
        let labels: Vec<_> = aligned.iter().map(|pair| pair.slot).collect();

        assert_eq!(labels, timestamps);
        assert_eq!(aligned.unmatched_left(), 0);
        assert_eq!(aligned.unmatched_right(), 0);
    }
}
