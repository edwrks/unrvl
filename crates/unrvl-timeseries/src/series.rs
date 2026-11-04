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
        "value at index {index} must be finite, got {value} at {nanos} nanoseconds",
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
        "timestamp at index {index} ({current_nanos} nanoseconds) must be later than \
         index {previous_index} ({previous_nanos} nanoseconds)",
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
        "range start ({start_nanos} nanoseconds) must not be later than \
         end ({end_nanos} nanoseconds)",
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
    ///
    /// Uses `O(1)` time and additional memory. The view borrows from this
    /// series.
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
mod tests;
