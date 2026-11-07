use std::ops::{Bound, RangeBounds};

use thiserror::Error;

use crate::Timestamp;

/// Invalid series input or time-range bounds.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum SeriesError {
    /// Timestamp and value slices have different lengths.
    #[error("inputs must have equal lengths, got timestamps={timestamps} and values={values}")]
    LengthMismatch {
        /// Number of supplied timestamps.
        timestamps: usize,

        /// Number of supplied values.
        values: usize,
    },

    /// An observed value is NaN or infinite.
    #[error(
        "value at index {index} must be finite, got {value} at {nanos} ns",
        nanos = .timestamp.as_unix_nanos(),
    )]
    NonFiniteValue {
        /// Zero-based observation index.
        index: usize,

        /// Timestamp paired with the non-finite value.
        timestamp: Timestamp,

        /// The non-finite value.
        value: f64,
    },

    /// An observation's timestamp is equal to or earlier than its predecessor.
    #[error(
        "timestamp at index {index} ({current_nanos} ns) must be later than \
         index {previous_index} ({previous_nanos} ns)",
        current_nanos = .current.as_unix_nanos(),
        previous_nanos = .previous.as_unix_nanos(),
    )]
    NonIncreasingTimestamp {
        /// Zero-based index of the preceding observation.
        previous_index: usize,

        /// Zero-based index of the offending observation.
        index: usize,

        /// Timestamp of the preceding observation.
        previous: Timestamp,

        /// Timestamp that must be later than `previous`.
        current: Timestamp,
    },

    /// A bounded time range starts after it ends.
    #[error(
        "range start ({start_nanos} ns) must not be later than \
         end ({end_nanos} ns)",
        start_nanos = .start.as_unix_nanos(),
        end_nanos = .end.as_unix_nanos(),
    )]
    ReversedRange {
        /// Supplied lower endpoint, whether inclusive or exclusive.
        start: Timestamp,

        /// Supplied upper endpoint, whether inclusive or exclusive.
        end: Timestamp,
    },
}

/// An owned numerical series with finite values and strictly increasing
/// timestamps.
///
/// Owns corresponding timestamp and value vectors, paired by index. Empty and
/// singleton series are valid, as are signed values and irregular observation
/// intervals. Access is immutable to preserve the validation guarantees.
///
/// [`Self::view`] borrows the stored observations without allocation or
/// revalidation. Cloning a series copies both vectors.
///
/// # Examples
///
/// ```
/// # use unrvl_timeseries::{Series, SeriesError, Timestamp};
/// let timestamps = [-1, 0, 2].map(Timestamp::from_unix_nanos).to_vec();
/// let values = vec![-2.0, 0.0, 5.0];
/// let series = Series::new(timestamps, values)?;
/// let view = series.view();
///
/// assert_eq!(view.len(), 3);
/// assert_eq!(view.values(), &[-2.0, 0.0, 5.0]);
/// # Ok::<(), SeriesError>(())
/// ```
#[derive(Debug, Clone)]
pub struct Series {
    timestamps: Vec<Timestamp>,
    values: Vec<f64>,
}

impl Series {
    /// Validates and takes ownership of corresponding timestamp and value
    /// vectors.
    ///
    /// Requires equal lengths, finite values, and strictly increasing
    /// timestamps. Empty and singleton samples are valid. Observations are
    /// neither sorted, dropped, nor filled.
    ///
    /// Uses `O(n)` time and `O(1)` additional memory for `n` observations,
    /// beyond the supplied vectors. Takes their buffers without copying or
    /// reallocating them.
    ///
    /// # Errors
    ///
    /// Uses the same validation and [`SeriesError`] variants as
    /// [`SeriesView::new`]. Length mismatches are reported first; otherwise
    /// reports the first invalid observation in index order.
    pub fn new(timestamps: Vec<Timestamp>, values: Vec<f64>) -> Result<Self, SeriesError> {
        validate(&timestamps, &values)?;

        Ok(Self { timestamps, values })
    }

    /// Returns a view of the stored observations without allocation or
    /// revalidation.
    #[must_use]
    pub const fn view(&self) -> SeriesView<'_> {
        SeriesView {
            timestamps: self.timestamps(),
            values: self.values(),
        }
    }

    /// Returns the original, strictly increasing timestamp slice.
    #[must_use]
    pub const fn timestamps(&self) -> &[Timestamp] {
        self.timestamps.as_slice()
    }

    /// Returns the original finite value slice, paired with timestamps by
    /// index.
    #[must_use]
    pub const fn values(&self) -> &[f64] {
        self.values.as_slice()
    }

    /// Returns the number of timestamp/value pairs.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether there are no observations.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// A borrowed numerical series with finite values and strictly increasing
/// timestamps.
///
/// Timestamps and values are paired by index. Empty and singleton views are
/// valid, as are signed values and irregular observation intervals. The view
/// borrows both slices without copying or allocating and exposes immutable
/// access to preserve its validation guarantees.
///
/// Structural validation does not establish sampling cadence, temporal
/// coverage, or information availability.
///
/// # Examples
///
/// ```
/// # use unrvl_timeseries::{SeriesError, SeriesView, Timestamp};
/// let timestamps = [-1, 0, 2].map(Timestamp::from_unix_nanos);
/// let values = [-2.0, 0.0, 5.0];
/// let series = SeriesView::new(&timestamps, &values)?;
///
/// assert_eq!(series.len(), 3);
/// assert_eq!(series.values(), &values);
/// # Ok::<(), SeriesError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SeriesView<'a> {
    timestamps: &'a [Timestamp],
    values: &'a [f64],
}

impl<'a> SeriesView<'a> {
    /// Validates and borrows corresponding timestamp and value slices.
    ///
    /// Requires equal lengths, finite values, and strictly increasing
    /// timestamps. Empty and singleton samples are valid. Observations are
    /// neither sorted, dropped, nor filled.
    ///
    /// Uses `O(n)` time and `O(1)` additional memory for `n` observations.
    ///
    /// # Errors
    ///
    /// Returns [`SeriesError::LengthMismatch`] before inspecting observations
    /// if the lengths differ. Otherwise reports the first invalid observation
    /// in index order: a [`SeriesError::NonFiniteValue`] or
    /// [`SeriesError::NonIncreasingTimestamp`].
    pub fn new(timestamps: &'a [Timestamp], values: &'a [f64]) -> Result<Self, SeriesError> {
        validate(timestamps, values)?;

        Ok(Self { timestamps, values })
    }

    /// Borrows the observations whose timestamps satisfy the supplied range.
    ///
    /// Accepts inclusive, exclusive, and unbounded endpoints through
    /// [`RangeBounds`]. For example, `start..end` excludes `end`, while
    /// `start..=end` includes it. Equal endpoints select a matching observation
    /// only when both are inclusive; otherwise the result is empty.
    ///
    /// A valid range containing no observations returns an empty view. Bounds
    /// may extend beyond the stored timestamps. The result borrows the
    /// corresponding slices without allocation or structural revalidation.
    ///
    /// Uses `O(log n)` time and `O(1)` additional memory for `n` observations.
    ///
    /// # Errors
    ///
    /// Returns [`SeriesError::ReversedRange`] if both endpoints are bounded
    /// and the start is later than the end, including for an empty source.
    pub fn between(&self, range: impl RangeBounds<Timestamp>) -> Result<Self, SeriesError> {
        let start_bound = range.start_bound();
        let end_bound = range.end_bound();

        // Only explicitly bounded endpoints can form a reversed range.
        if let (
            Bound::Included(start) | Bound::Excluded(start),
            Bound::Included(end) | Bound::Excluded(end),
        ) = (start_bound, end_bound)
            && start > end
        {
            return Err(SeriesError::ReversedRange { start: *start, end: *end });
        }

        let start_index = match start_bound {
            Bound::Included(start) => self
                .timestamps
                .partition_point(|timestamp| timestamp < start),
            Bound::Excluded(start) => self
                .timestamps
                .partition_point(|timestamp| timestamp <= start),
            Bound::Unbounded => 0,
        };

        let timestamps = &self.timestamps[start_index..];
        let values = &self.values[start_index..];

        // Search the remaining slice so equal excluded endpoints naturally
        // produce an empty view rather than inverted slice indices.
        let end_index = match end_bound {
            Bound::Included(end) => timestamps.partition_point(|timestamp| timestamp <= end),
            Bound::Excluded(end) => timestamps.partition_point(|timestamp| timestamp < end),
            Bound::Unbounded => timestamps.len(),
        };

        Ok(Self {
            timestamps: &timestamps[..end_index],
            values: &values[..end_index],
        })
    }

    /// Returns the original, strictly increasing timestamp slice.
    #[must_use]
    pub const fn timestamps(&self) -> &'a [Timestamp] {
        self.timestamps
    }

    /// Returns the original finite value slice, paired with timestamps by
    /// index.
    #[must_use]
    pub const fn values(&self) -> &'a [f64] {
        self.values
    }

    /// Returns the number of timestamp/value pairs.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether there are no observations.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Checks matching lengths, then reports the first invalid observation in index
/// order. At each index, a non-finite value takes precedence over timestamp
/// ordering.
fn validate(timestamps: &[Timestamp], values: &[f64]) -> Result<(), SeriesError> {
    if timestamps.len() != values.len() {
        return Err(SeriesError::LengthMismatch {
            timestamps: timestamps.len(),
            values: values.len(),
        });
    }

    for (index, (&timestamp, &value)) in timestamps.iter().zip(values).enumerate() {
        if !value.is_finite() {
            return Err(SeriesError::NonFiniteValue { index, timestamp, value });
        }

        if index > 0 {
            let previous_index = index - 1;
            let previous = timestamps[previous_index];

            if timestamp <= previous {
                return Err(SeriesError::NonIncreasingTimestamp {
                    previous_index,
                    index,
                    previous,
                    current: timestamp,
                });
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    mod series {
        use super::*;

        #[test]
        fn rejects_invalid_input_with_the_shared_error_context() {
            let epoch = Timestamp::UNIX_EPOCH;
            let error = Series::new(vec![epoch], vec![f64::INFINITY]).unwrap_err();

            assert_eq!(
                error,
                SeriesError::NonFiniteValue { index: 0, timestamp: epoch, value: f64::INFINITY },
            );
        }

        #[test]
        fn accepts_empty_and_singleton_series() {
            let cases = [(vec![], vec![]), (vec![Timestamp::UNIX_EPOCH], vec![0.0])];

            for (timestamps, values) in cases {
                let expected_len = values.len();
                let series = Series::new(timestamps, values).unwrap();

                assert_eq!(series.len(), expected_len);
                assert_eq!(series.is_empty(), expected_len == 0);
            }
        }

        #[test]
        fn preserves_supplied_buffers_and_observations() {
            let expected_timestamps = [-1, 0, 2, 5].map(Timestamp::from_unix_nanos);
            let expected_values = [-2.0, -0.0, 0.0, 5.0];

            let timestamps = expected_timestamps.to_vec();
            let values = expected_values.to_vec();

            let timestamps_ptr = timestamps.as_ptr();
            let values_ptr = values.as_ptr();

            let series = Series::new(timestamps, values).unwrap();

            assert_eq!(series.timestamps().as_ptr(), timestamps_ptr);
            assert_eq!(series.values().as_ptr(), values_ptr);
            assert_eq!(series.timestamps(), expected_timestamps);

            // Compare bits so that the sign of zero is checked too.
            let actual_bits: Vec<_> = series
                .values()
                .iter()
                .map(|value| value.to_bits())
                .collect();

            assert_eq!(actual_bits, expected_values.map(f64::to_bits));
        }

        #[test]
        fn view_borrows_the_owned_buffers() {
            let timestamps = [0, 1, 3].map(Timestamp::from_unix_nanos).to_vec();
            let values = vec![-2.0, 0.0, 5.0];
            let series = Series::new(timestamps, values).unwrap();
            let view = series.view();

            assert!(std::ptr::eq(view.timestamps(), series.timestamps()));
            assert!(std::ptr::eq(view.values(), series.values()));
            assert_eq!(view.len(), series.len());
            assert_eq!(view.is_empty(), series.is_empty());
        }
    }

    mod series_view {
        use super::*;

        #[test]
        fn rejects_nonfinite_values_with_observation_context() {
            let timestamps = [-1, 0, 2].map(Timestamp::from_unix_nanos);
            let invalid_values = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];

            for value in invalid_values {
                for index in 0..timestamps.len() {
                    let mut values = [-2.0, 0.0, 5.0];
                    values[index] = value;

                    let error = SeriesView::new(&timestamps, &values).unwrap_err();

                    let SeriesError::NonFiniteValue {
                        index: actual_index,
                        timestamp,
                        value: actual_value,
                    } = error
                    else {
                        panic!("unexpected error: {error:?}");
                    };

                    assert_eq!(actual_index, index);
                    assert_eq!(timestamp, timestamps[index]);
                    assert_eq!(actual_value.to_bits(), value.to_bits());
                }
            }
        }

        #[test]
        fn rejects_unequal_lengths_including_one_empty_slice() {
            let timestamps = [0, 1].map(Timestamp::from_unix_nanos);

            // Length errors take precedence even when a value is non-finite.
            let values = [0.0, f64::NAN];
            let cases = [(0, 1), (1, 0), (1, 2), (2, 1)];

            for (timestamp_count, value_count) in cases {
                let error = SeriesView::new(&timestamps[..timestamp_count], &values[..value_count])
                    .unwrap_err();

                assert_eq!(
                    error,
                    SeriesError::LengthMismatch {
                        timestamps: timestamp_count,
                        values: value_count
                    },
                );
            }
        }

        #[test]
        fn rejects_non_increasing_timestamps() {
            let cases = [
                ([-2, -1, -1, 1], -1, -1),
                ([-2, 1, 0, 2], 1, 0),
                ([i64::MIN, i64::MAX, i64::MIN, i64::MAX], i64::MAX, i64::MIN),
            ];
            let values = [0.0; 4];

            for (timestamps, previous, current) in cases {
                let timestamps = timestamps.map(Timestamp::from_unix_nanos);
                let error = SeriesView::new(&timestamps, &values).unwrap_err();

                assert_eq!(
                    error,
                    SeriesError::NonIncreasingTimestamp {
                        previous_index: 1,
                        index: 2,
                        previous: Timestamp::from_unix_nanos(previous),
                        current: Timestamp::from_unix_nanos(current),
                    },
                );
            }
        }

        #[test]
        fn reports_the_first_invalid_observation_across_validation_rules() {
            let epoch = Timestamp::UNIX_EPOCH;
            let next = Timestamp::from_unix_nanos(1);
            let cases = [
                (
                    [epoch, epoch, next],
                    [0.0, 0.0, f64::INFINITY],
                    SeriesError::NonIncreasingTimestamp {
                        previous_index: 0,
                        index: 1,
                        previous: epoch,
                        current: epoch,
                    },
                ),
                (
                    [epoch, next, epoch],
                    [0.0, f64::INFINITY, 0.0],
                    SeriesError::NonFiniteValue { index: 1, timestamp: next, value: f64::INFINITY },
                ),
            ];

            for (timestamps, values, expected) in cases {
                let error = SeriesView::new(&timestamps, &values).unwrap_err();

                assert_eq!(error, expected);
            }
        }

        #[test]
        fn accepts_empty_slices() {
            let series = SeriesView::new(&[], &[]).unwrap();

            assert_eq!(series.len(), 0);
            assert!(series.is_empty());
            assert_eq!(series.timestamps(), []);
            assert_eq!(series.values(), []);
        }

        #[test]
        fn accepts_a_single_observation() {
            let timestamps = [Timestamp::UNIX_EPOCH];
            let values = [-2.0];
            let series = SeriesView::new(&timestamps, &values).unwrap();

            assert_eq!(series.len(), 1);
            assert!(!series.is_empty());
        }

        #[test]
        fn borrows_irregular_observations_without_copying() {
            let timestamps = [-1, 0, 2].map(Timestamp::from_unix_nanos);
            let values = [-2.0, 0.0, 5.0];
            let series = SeriesView::new(&timestamps, &values).unwrap();

            assert!(std::ptr::eq(series.timestamps(), timestamps.as_slice()));
            assert!(std::ptr::eq(series.values(), values.as_slice()));
            assert_eq!(series.len(), values.len());
            assert!(!series.is_empty());
        }
    }

    mod between {
        use std::ops::Bound::{Excluded, Included, Unbounded};

        use super::*;

        #[test]
        fn rejects_reversed_bounds_even_for_empty_views() {
            let timestamps = [-1, 0, 3].map(Timestamp::from_unix_nanos);
            let values = [-2.0, 0.0, 5.0];
            let series = SeriesView::new(&timestamps, &values).unwrap();
            let start = timestamps[2];
            let end = timestamps[0];
            let expected = SeriesError::ReversedRange { start, end };
            let cases = [
                (Included(start), Included(end)),
                (Included(start), Excluded(end)),
                (Excluded(start), Included(end)),
                (Excluded(start), Excluded(end)),
            ];

            for bounds in cases {
                assert_eq!(series.between(bounds).unwrap_err(), expected);
            }

            let empty = SeriesView::new(&[], &[]).unwrap();

            assert_eq!(empty.between(start..end).unwrap_err(), expected);
        }

        #[test]
        fn respects_included_excluded_and_unbounded_endpoints() {
            let timestamps = [-5, -1, 0, 3, 8].map(Timestamp::from_unix_nanos);
            let values = [-50.0, -10.0, 0.0, 30.0, 80.0];
            let series = SeriesView::new(&timestamps, &values).unwrap();
            let lower = timestamps[1];
            let upper = timestamps[3];
            let cases = [
                ((Included(lower), Included(upper)), 1..4),
                ((Included(lower), Excluded(upper)), 1..3),
                ((Excluded(lower), Included(upper)), 2..4),
                ((Excluded(lower), Excluded(upper)), 2..3),
                ((Included(lower), Unbounded), 1..5),
                ((Excluded(lower), Unbounded), 2..5),
                ((Unbounded, Included(upper)), 0..4),
                ((Unbounded, Excluded(upper)), 0..3),
                ((Unbounded, Unbounded), 0..5),
            ];

            for (bounds, expected_indices) in cases {
                let selected = series.between(bounds).unwrap();

                assert_eq!(selected.timestamps(), &timestamps[expected_indices.clone()]);
                assert_eq!(selected.values(), &values[expected_indices]);
            }
        }

        #[test]
        fn selects_by_bounds_when_endpoints_are_not_observed() {
            let timestamps = [-5, -1, 0, 3, 8].map(Timestamp::from_unix_nanos);
            let values = [-50.0, -10.0, 0.0, 30.0, 80.0];
            let series = SeriesView::new(&timestamps, &values).unwrap();
            let cases = [
                (-3, 5, 1..4),
                (-10, 1, 0..3),
                (1, 10, 3..5),
                (-10, 10, 0..5),
                // Empty selections in a gap, before the data, and after it.
                (4, 7, 4..4),
                (-10, -6, 0..0),
                (9, 12, 5..5),
            ];

            for (start, end, expected_indices) in cases {
                let start = Timestamp::from_unix_nanos(start);
                let end = Timestamp::from_unix_nanos(end);
                let selected = series.between(start..end).unwrap();

                assert_eq!(selected.timestamps(), &timestamps[expected_indices.clone()]);
                assert_eq!(selected.values(), &values[expected_indices]);
            }
        }

        #[test]
        fn equal_endpoints_select_a_point_only_when_both_are_inclusive() {
            let timestamps = [-1, 0, 2].map(Timestamp::from_unix_nanos);
            let values = [-2.0, 0.0, 5.0];
            let series = SeriesView::new(&timestamps, &values).unwrap();
            let epoch = Timestamp::UNIX_EPOCH;
            let cases: [(_, &[f64]); 4] = [
                ((Included(epoch), Included(epoch)), &[0.0]),
                ((Included(epoch), Excluded(epoch)), &[]),
                ((Excluded(epoch), Included(epoch)), &[]),
                ((Excluded(epoch), Excluded(epoch)), &[]),
            ];

            for (bounds, expected_values) in cases {
                let selected = series.between(bounds).unwrap();

                assert_eq!(selected.values(), expected_values);
            }

            let missing = Timestamp::from_unix_nanos(1);

            assert!(series.between(missing..=missing).unwrap().is_empty());
        }

        #[test]
        fn selects_from_empty_and_singleton_views() {
            let empty = SeriesView::new(&[], &[]).unwrap();

            assert!(empty.between(..).unwrap().is_empty());

            let epoch = Timestamp::UNIX_EPOCH;
            let timestamps = [epoch];
            let values = [-2.0];
            let single = SeriesView::new(&timestamps, &values).unwrap();

            assert_eq!(single.between(epoch..=epoch).unwrap().values(), values);
            assert!(single.between(..epoch).unwrap().is_empty());
            assert!(
                single
                    .between((Excluded(epoch), Unbounded))
                    .unwrap()
                    .is_empty()
            );
        }

        #[test]
        fn handles_timestamp_limits_without_endpoint_arithmetic() {
            let timestamps =
                [i64::MIN, i64::MIN + 1, 0, i64::MAX - 1, i64::MAX].map(Timestamp::from_unix_nanos);
            let values = [-2.0, -1.0, 0.0, 1.0, 2.0];
            let series = SeriesView::new(&timestamps, &values).unwrap();
            let cases = [
                ((Unbounded, Excluded(Timestamp::MIN)), 0..0),
                ((Unbounded, Included(Timestamp::MIN)), 0..1),
                ((Included(Timestamp::MIN), Included(Timestamp::MAX)), 0..5),
                ((Excluded(Timestamp::MIN), Excluded(Timestamp::MAX)), 1..4),
                ((Included(Timestamp::MAX), Unbounded), 4..5),
                ((Excluded(Timestamp::MAX), Unbounded), 5..5),
            ];

            for (bounds, expected_indices) in cases {
                let selected = series.between(bounds).unwrap();

                assert_eq!(selected.timestamps(), &timestamps[expected_indices.clone()]);
                assert_eq!(selected.values(), &values[expected_indices]);
            }
        }

        #[test]
        fn selections_borrow_the_source_slices_and_can_be_narrowed() {
            let timestamps = [-1, 0, 3, 8].map(Timestamp::from_unix_nanos);
            let values = [-2.0, 0.0, 5.0, 7.0];
            let selected = {
                let series = SeriesView::new(&timestamps, &values).unwrap();

                series.between(timestamps[1]..).unwrap()
            };
            let narrowed = selected.between(..=timestamps[2]).unwrap();
            let expected_timestamps = &timestamps[1..3];
            let expected_values = &values[1..3];

            assert!(std::ptr::eq(narrowed.timestamps(), expected_timestamps));
            assert!(std::ptr::eq(narrowed.values(), expected_values));
        }
    }
}
