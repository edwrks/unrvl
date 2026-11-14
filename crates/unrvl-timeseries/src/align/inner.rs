use super::merge::matching_indices;
use crate::{SeriesView, Timestamp};

/// Two observations with the same timestamp.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlignedPair {
    /// Timestamp shared by both observations.
    pub timestamp: Timestamp,

    /// Index into the supplied left view.
    pub left_index: usize,

    /// Index into the supplied right view.
    pub right_index: usize,

    /// Value from the left view.
    pub left_value: f64,

    /// Value from the right view.
    pub right_value: f64,
}

/// Result of matching equal timestamps in two series.
///
/// Borrows the source data and stores one index pair per match. An alignment
/// can be empty.
#[derive(Debug, Clone)]
pub struct InnerAlignment<'left, 'right> {
    left: SeriesView<'left>,
    right: SeriesView<'right>,
    matches: Vec<(usize, usize)>,
}

impl InnerAlignment<'_, '_> {
    /// Returns the number of matched pairs.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.matches.len()
    }

    /// Returns whether the two views share zero timestamps.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.matches.is_empty()
    }

    /// Returns the number of unmatched observations in the left view.
    #[must_use]
    pub const fn unmatched_left(&self) -> usize {
        self.left.len() - self.len()
    }

    /// Returns the number of unmatched observations in the right view.
    #[must_use]
    pub const fn unmatched_right(&self) -> usize {
        self.right.len() - self.len()
    }

    /// Yields the matched pairs in timestamp order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = AlignedPair> + '_ {
        self.matches
            .iter()
            .map(|&(left_index, right_index)| AlignedPair {
                timestamp: self.left.timestamps()[left_index],
                left_index,
                right_index,
                left_value: self.left.values()[left_index],
                right_value: self.right.values()[right_index],
            })
    }
}

/// Pairs observations whose timestamps are exactly equal.
///
/// Both views may contain irregularly spaced observations. Indices in the
/// result are relative to the supplied views, including any range selections.
///
/// Uses `O(n + m)` time for input lengths `n` and `m`, and `O(k)` additional
/// memory for `k` matches.
///
/// # Examples
/// ```
/// # use unrvl_timeseries::{SeriesError, SeriesView, Timestamp, align};
/// let left_times = [1, 3, 5].map(Timestamp::from_unix_nanos);
/// let right_times = [3, 4, 5].map(Timestamp::from_unix_nanos);
/// let left_values = [10.0, 30.0, 50.0];
/// let right_values = [300.0, 400.0, 500.0];
/// let left = SeriesView::new(&left_times, &left_values)?;
/// let right = SeriesView::new(&right_times, &right_values)?;
///
/// let aligned = align::inner(left, right);
/// let first = aligned.iter().next().unwrap();
/// assert_eq!(aligned.len(), 2);
/// assert_eq!(first.timestamp, left_times[1]);
/// assert_eq!((first.left_index, first.right_index), (1, 0));
/// assert_eq!((first.left_value, first.right_value), (30.0, 300.0));
/// # Ok::<(), SeriesError>(())
/// ```
#[must_use]
pub fn inner<'left, 'right>(
    left: SeriesView<'left>,
    right: SeriesView<'right>,
) -> InnerAlignment<'left, 'right> {
    let matches = matching_indices(left.timestamps(), right.timestamps()).collect();

    InnerAlignment { left, right, matches }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairs_equal_timestamps_and_counts_unmatched_observations() {
        let left_times = [-2, 1, 3, 5, 9].map(Timestamp::from_unix_nanos);
        let right_times = [0, 3, 4, 5, 8, 10].map(Timestamp::from_unix_nanos);
        let left_values = [-20.0, 10.0, 30.0, 50.0, 90.0];
        let right_values = [0.0, 300.0, 400.0, 500.0, 800.0, 1000.0];
        let left = SeriesView::new(&left_times, &left_values).unwrap();
        let right = SeriesView::new(&right_times, &right_values).unwrap();

        let aligned = inner(left, right);
        let pairs: Vec<_> = aligned.iter().collect();

        assert_eq!(aligned.len(), 2);
        assert_eq!(aligned.unmatched_left(), 3);
        assert_eq!(aligned.unmatched_right(), 4);
        assert_eq!(
            pairs,
            [
                AlignedPair {
                    timestamp: Timestamp::from_unix_nanos(3),
                    left_index: 2,
                    right_index: 1,
                    left_value: 30.0,
                    right_value: 300.0,
                },
                AlignedPair {
                    timestamp: Timestamp::from_unix_nanos(5),
                    left_index: 3,
                    right_index: 3,
                    left_value: 50.0,
                    right_value: 500.0,
                },
            ]
        );
    }

    #[test]
    fn empty_or_disjoint_views_produce_an_empty_alignment() {
        let empty = SeriesView::new(&[], &[]).unwrap();
        let left = SeriesView::new(&[Timestamp::MIN], &[1.0]).unwrap();
        let right = SeriesView::new(&[Timestamp::MAX], &[2.0]).unwrap();

        for (left, right) in [
            (empty, empty),
            (empty, right),
            (left, empty),
            (left, right),
            (right, left),
        ] {
            let aligned = inner(left, right);

            assert!(aligned.is_empty());
            assert_eq!(aligned.iter().next(), None);
            assert_eq!(aligned.unmatched_left(), left.len());
            assert_eq!(aligned.unmatched_right(), right.len());
        }
    }

    #[test]
    fn distinguishes_adjacent_nanoseconds_beyond_float_precision() {
        let distant = 1_i64 << 53;

        let left_times = [i64::MIN, distant, distant + 1, i64::MAX].map(Timestamp::from_unix_nanos);
        let right_times = [i64::MIN, distant + 1, i64::MAX].map(Timestamp::from_unix_nanos);

        let left_values = [1.0, 2.0, 3.0, 4.0];
        let right_values = [10.0, 30.0, 40.0];
        let left = SeriesView::new(&left_times, &left_values).unwrap();
        let right = SeriesView::new(&right_times, &right_values).unwrap();

        let aligned = inner(left, right);
        let left_indices: Vec<_> = aligned.iter().map(|pair| pair.left_index).collect();

        assert_eq!(left_indices, [0, 2, 3]);
        assert_eq!(aligned.unmatched_left(), 1);
        assert_eq!(aligned.unmatched_right(), 0);
    }

    #[test]
    fn reports_indices_relative_to_the_supplied_views() {
        let left_times = [1, 3, 5].map(Timestamp::from_unix_nanos);
        let right_times = [2, 4, 5].map(Timestamp::from_unix_nanos);

        let left_values = [10.0, 30.0, 50.0];
        let right_values = [20.0, 40.0, 50.0];
        let left = SeriesView::new(&left_times, &left_values).unwrap();
        let right = SeriesView::new(&right_times, &right_values).unwrap();

        let left = left.between(left_times[1]..).unwrap();
        let right = right.between(right_times[2]..).unwrap();

        let aligned = inner(left, right);
        let pair = aligned.iter().next().unwrap();

        assert_eq!((pair.left_index, pair.right_index), (1, 0));
    }
}
