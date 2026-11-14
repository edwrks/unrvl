use std::ops::RangeBounds;

use thiserror::Error;

use crate::{SeriesError, SeriesView, Timestamp};

/// Invalid availability timestamps for a series.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum AvailabilityError {
    /// Observation and availability counts differ.
    #[error(
        "inputs must have equal lengths, got observations={observations} and availability={availability}"
    )]
    LengthMismatch {
        /// Number of observations in the series.
        observations: usize,

        /// Number of supplied availability timestamps.
        availability: usize,
    },

    /// An observation is marked available before it was observed.
    #[error(
        "availability at index {index} ({available_nanos} ns) must be at or after \
         observation time ({observed_nanos} ns)",
        available_nanos = .available_at.as_unix_nanos(),
        observed_nanos = .observed_at.as_unix_nanos(),
    )]
    BeforeObservation {
        /// Zero-based observation index.
        index: usize,

        /// Observation timestamp from the series.
        observed_at: Timestamp,

        /// Supplied availability timestamp.
        available_at: Timestamp,
    },
}

/// A borrowed series paired with when each observation became available.
///
/// The series timestamps record when observations occurred. The corresponding
/// `available_at` timestamps record when those observations became usable, such
/// as when a provider published them or an application received them. Callers
/// supply these times using one consistent meaning.
///
/// Each availability timestamp must be at or after its observation timestamp.
/// Availability times may repeat or be out of order: an older observation can
/// arrive after a newer one. Construction preserves the supplied order and
/// borrows the data without allocation.
///
/// # Examples
///
/// ```
/// # use unrvl_timeseries::{AvailabilityView, SeriesView, Timestamp};
/// let observed_at = [Timestamp::from_unix_secs(10 * 60 * 60)?];
/// let available_at = [Timestamp::from_unix_secs(10 * 60 * 60 + 7 * 60)?];
/// let prices = SeriesView::new(&observed_at, &[100.0])?;
/// let availability = AvailabilityView::new(prices, &available_at)?;
///
/// // The 10:00 price became available at 10:07.
/// assert_eq!(availability.series().values(), &[100.0]);
/// assert_eq!(availability.available_at(), &available_at);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AvailabilityView<'a> {
    series: SeriesView<'a>,
    available_at: &'a [Timestamp],
}

impl<'a> AvailabilityView<'a> {
    /// Pairs a validated series with caller-supplied availability timestamps.
    ///
    /// Requires matching lengths and `available_at >= observed_at` for each
    /// observation. Empty views are valid. Availability times need not be
    /// sorted or unique. The series is borrowed without revalidation.
    ///
    /// Uses `O(n)` time and `O(1)` additional memory for `n` observations.
    ///
    /// # Errors
    ///
    /// Returns [`AvailabilityError::LengthMismatch`] before inspecting
    /// timestamps if the lengths differ. Otherwise reports the first
    /// [`AvailabilityError::BeforeObservation`] in series index order.
    pub fn new(
        series: SeriesView<'a>,
        available_at: &'a [Timestamp],
    ) -> Result<Self, AvailabilityError> {
        if series.len() != available_at.len() {
            return Err(AvailabilityError::LengthMismatch {
                observations: series.len(),
                availability: available_at.len(),
            });
        }

        for (index, (&observed_at, &available_at)) in
            series.timestamps().iter().zip(available_at).enumerate()
        {
            if available_at < observed_at {
                return Err(AvailabilityError::BeforeObservation {
                    index,
                    observed_at,
                    available_at,
                });
            }
        }

        Ok(Self { series, available_at })
    }

    /// Borrows observations in the supplied range, keeping availability times
    /// paired with them.
    ///
    /// Selects by observation time, using the same inclusive, exclusive, and
    /// unbounded endpoints as [`SeriesView::between`]. Availability times are
    /// preserved even when they lie outside the range. A valid range with no
    /// observations returns an empty view.
    ///
    /// Borrows the corresponding slices without allocation or revalidation.
    /// Uses `O(log n)` time and `O(1)` additional memory for `n` observations.
    ///
    /// # Errors
    ///
    /// Returns [`SeriesError::ReversedRange`] if both endpoints are bounded
    /// and the start is later than the end, including for an empty source.
    ///
    /// # Examples
    ///
    /// ```
    /// # use unrvl_timeseries::{AvailabilityView, SeriesView, Timestamp};
    /// let observed_at = [
    ///     Timestamp::from_unix_secs(10 * 3600)?,
    ///     Timestamp::from_unix_secs(10 * 3600 + 2 * 60)?,
    /// ];
    /// let available_at = [
    ///     Timestamp::from_unix_secs(10 * 3600 + 7 * 60)?,
    ///     Timestamp::from_unix_secs(10 * 3600 + 8 * 60)?,
    /// ];
    /// let prices = SeriesView::new(&observed_at, &[100.0, 102.0])?;
    /// let availability = AvailabilityView::new(prices, &available_at)?;
    ///
    /// let end = Timestamp::from_unix_secs(10 * 3600 + 5 * 60)?;
    /// let selected = availability.between(observed_at[1]..end)?;
    ///
    /// assert_eq!(selected.series().values(), &[102.0]);
    /// assert_eq!(selected.available_at(), &available_at[1..]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn between(&self, range: impl RangeBounds<Timestamp>) -> Result<Self, SeriesError> {
        let indices = self.series.indices_between(range)?;

        Ok(Self {
            series: self.series.slice(indices.clone()),
            available_at: &self.available_at[indices],
        })
    }

    /// Returns the underlying series without allocation or revalidation.
    #[must_use]
    pub const fn series(&self) -> SeriesView<'a> {
        self.series
    }

    /// Returns availability timestamps paired with observations by index.
    #[must_use]
    pub const fn available_at(&self) -> &'a [Timestamp] {
        self.available_at
    }

    /// Returns the number of observations.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.series.len()
    }

    /// Returns whether there are no observations.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.series.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::timestamp;

    #[test]
    fn rejects_unequal_lengths_before_inspecting_timestamps() {
        let observed_at = [0, 2].map(Timestamp::from_unix_nanos);
        let values = [100.0, 101.0];

        // This availability time is invalid too, but lengths are checked first.
        let available_at = [-1, 2].map(Timestamp::from_unix_nanos);
        let cases = [(0, 1), (1, 0), (1, 2), (2, 1)];

        for (observation_count, availability_count) in cases {
            let series = SeriesView::new(
                &observed_at[..observation_count],
                &values[..observation_count],
            )
            .unwrap();

            let error =
                AvailabilityView::new(series, &available_at[..availability_count]).unwrap_err();

            assert_eq!(
                error,
                AvailabilityError::LengthMismatch {
                    observations: observation_count,
                    availability: availability_count,
                },
            );
        }
    }

    #[test]
    fn reports_the_first_early_availability_with_observation_context() {
        let observed_at = [-4, -1, 2, 5].map(Timestamp::from_unix_nanos);
        let available_at = [-4, -2, 0, 5].map(Timestamp::from_unix_nanos);
        let values = [100.0; 4];
        let series = SeriesView::new(&observed_at, &values).unwrap();

        let error = AvailabilityView::new(series, &available_at).unwrap_err();

        assert_eq!(
            error,
            AvailabilityError::BeforeObservation {
                index: 1,
                observed_at: observed_at[1],
                available_at: available_at[1],
            },
        );
        assert_eq!(
            error.to_string(),
            "availability at index 1 (-2 ns) must be at or after observation time (-1 ns)",
        );
    }

    #[test]
    fn borrows_the_original_data_with_repeated_and_unordered_availability() {
        use std::ptr::eq;

        let observed_at = [-3, 0, 2, 5].map(Timestamp::from_unix_nanos);
        let available_at = [5, 4, 4, 5].map(Timestamp::from_unix_nanos);
        let values = [-2.0, 0.0, 5.0, 4.0];
        let series = SeriesView::new(&observed_at, &values).unwrap();

        let availability = AvailabilityView::new(series, &available_at).unwrap();

        assert!(eq(availability.series().timestamps(), &observed_at));
        assert!(eq(availability.series().values(), &values));
        assert!(eq(availability.available_at(), &available_at));
        assert_eq!(availability.len(), 4);
        assert!(!availability.is_empty());
    }

    #[test]
    fn accepts_empty_and_singleton_views() {
        let empty = SeriesView::new(&[], &[]).unwrap();

        let availability = AvailabilityView::new(empty, &[]).unwrap();

        assert_eq!(availability.len(), 0);
        assert!(availability.is_empty());

        let epoch = [Timestamp::UNIX_EPOCH];
        let singleton = SeriesView::new(&epoch, &[100.0]).unwrap();

        let availability = AvailabilityView::new(singleton, &epoch).unwrap();

        assert_eq!(availability.len(), 1);
        assert!(!availability.is_empty());
    }

    #[test]
    fn validates_the_full_timestamp_span() {
        let observed_at = [Timestamp::MIN, Timestamp::MAX];
        let available_at = [Timestamp::MAX, Timestamp::MAX];
        let series = SeriesView::new(&observed_at, &[100.0, 101.0]).unwrap();

        let availability = AvailabilityView::new(series, &available_at).unwrap();

        assert_eq!(availability.available_at(), &available_at);

        let early = [Timestamp::MIN, Timestamp::MIN];

        let error = AvailabilityView::new(series, &early).unwrap_err();

        assert_eq!(
            error,
            AvailabilityError::BeforeObservation {
                index: 1,
                observed_at: Timestamp::MAX,
                available_at: Timestamp::MIN,
            },
        );
    }

    #[test]
    fn range_selection_preserves_paired_slices_and_outlives_the_original_view() {
        use std::ptr::eq;

        let observed_at = [-3, 0, 2, 5].map(timestamp);
        let available_at = [7, 6, 6, 5].map(timestamp);
        let values = [-2.0, 0.0, 5.0, 4.0];
        let series = SeriesView::new(&observed_at, &values).unwrap();

        let selected = {
            let availability = AvailabilityView::new(series, &available_at).unwrap();
            let before_close = availability.between(..timestamp(5)).unwrap();

            before_close.between(timestamp(0)..=timestamp(2)).unwrap()
        };
        let expected_timestamps = &observed_at[1..3];
        let expected_values = &values[1..3];
        let expected_availability = &available_at[1..3];

        assert!(eq(selected.series().timestamps(), expected_timestamps));
        assert!(eq(selected.series().values(), expected_values));
        assert!(eq(selected.available_at(), expected_availability));
    }

    #[test]
    fn empty_ranges_keep_the_paired_slice_position() {
        use std::ops::Bound::{Excluded, Included, Unbounded};
        use std::ptr::eq;

        let observed_at = [0, 2, 4].map(timestamp);
        let available_at = [5, 6, 4].map(timestamp);
        let values = [100.0, 102.0, 104.0];
        let series = SeriesView::new(&observed_at, &values).unwrap();
        let availability = AvailabilityView::new(series, &available_at).unwrap();
        let middle = timestamp(2);
        let cases = [
            ((Unbounded, Excluded(timestamp(0))), 0),
            ((Included(middle), Excluded(middle)), 1),
            ((Excluded(middle), Included(middle)), 2),
            ((Excluded(timestamp(4)), Unbounded), 3),
        ];

        for (bounds, index) in cases {
            let selected = availability.between(bounds).unwrap();
            let expected_timestamps = &observed_at[index..index];
            let expected_values = &values[index..index];
            let expected_availability = &available_at[index..index];

            assert!(selected.is_empty(), "{bounds:?}");
            assert!(eq(selected.series().timestamps(), expected_timestamps));
            assert!(eq(selected.series().values(), expected_values));
            assert!(eq(selected.available_at(), expected_availability));
        }
    }

    #[test]
    fn range_selection_rejects_reversed_bounds_even_for_empty_input() {
        let observed_at = [Timestamp::UNIX_EPOCH];
        let series = SeriesView::new(&observed_at, &[100.0]).unwrap();
        let populated = AvailabilityView::new(series, &observed_at).unwrap();
        let empty = AvailabilityView::new(SeriesView::new(&[], &[]).unwrap(), &[]).unwrap();
        let start = timestamp(2);
        let end = timestamp(0);

        for availability in [populated, empty] {
            let error = availability.between(start..end).unwrap_err();

            assert_eq!(error, SeriesError::ReversedRange { start, end });
        }
    }
}
