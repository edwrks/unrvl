use thiserror::Error;

use crate::{SeriesView, Timestamp};

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
        let series = SeriesView::new(&observed_at, &[100.0; 4]).unwrap();
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
}
