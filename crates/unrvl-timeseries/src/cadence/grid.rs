use std::time::Duration;

use thiserror::Error;

use crate::Timestamp;

/// Invalid fixed-grid spacing or tolerance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum GridError {
    /// Spacing between expected timestamps is zero.
    #[error("grid spacing must be greater than zero")]
    ZeroSpacing,

    /// Tolerance windows would meet or overlap between neighboring slots.
    #[error(
        "grid tolerance ({tolerance_nanos} ns) must be less than half \
         the spacing ({spacing_nanos} ns)",
        tolerance_nanos = .tolerance.as_nanos(),
        spacing_nanos = .spacing.as_nanos(),
    )]
    AmbiguousTolerance {
        /// Supplied spacing between expected timestamps.
        spacing: Duration,

        /// Supplied maximum deviation from an expected timestamp.
        tolerance: Duration,
    },
}

/// A fixed elapsed-time sampling grid with an inclusive timestamp tolerance.
///
/// The origin anchors expected timestamps at whole multiples of the spacing
/// before and after it. For example, an origin of 09:00 with hourly spacing
/// also describes 08:00 and 10:00.
///
/// Tolerance is the maximum absolute deviation from an expected timestamp.
/// It must be strictly less than half the spacing, so an accepted timestamp
/// can belong to only one slot. Zero tolerance requires exact timestamps.
///
/// The grid records sampling expectations. Constructing it validates the
/// configuration without examining observations or establishing completeness.
///
/// # Examples
///
/// ```
/// # use std::time::Duration;
/// # use unrvl_timeseries::{Timestamp, cadence::FixedGrid};
/// let origin = Timestamp::from_unix_secs(9 * 3600)?;
/// let grid = FixedGrid::new(
///     origin,
///     Duration::from_secs(3600),
///     Duration::from_secs(2 * 60),
/// )?;
///
/// assert_eq!(grid.origin(), origin);
/// assert_eq!(grid.tolerance(), Duration::from_secs(120));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FixedGrid {
    origin: Timestamp,
    spacing: Duration,
    tolerance: Duration,
}

impl FixedGrid {
    /// Creates a grid anchored at `origin` with positive elapsed-time spacing.
    ///
    /// Tolerance is inclusive and must be strictly less than half the spacing.
    /// Both durations retain nanosecond precision and accept the full
    /// [`Duration`] range when they satisfy these rules.
    ///
    /// # Errors
    ///
    /// Returns [`GridError::ZeroSpacing`] first if spacing is zero. Otherwise
    /// returns [`GridError::AmbiguousTolerance`] if tolerance is at least half
    /// the spacing, including the supplied durations.
    pub fn new(
        origin: Timestamp,
        spacing: Duration,
        tolerance: Duration,
    ) -> Result<Self, GridError> {
        if spacing.is_zero() {
            return Err(GridError::ZeroSpacing);
        }

        // If doubling tolerance exceeds Duration's range, it also exceeds
        // every representable spacing.
        if tolerance
            .checked_mul(2)
            .is_none_or(|width| width >= spacing)
        {
            return Err(GridError::AmbiguousTolerance { spacing, tolerance });
        }

        Ok(Self { origin, spacing, tolerance })
    }

    /// Returns the anchor for expected timestamps before and after it.
    #[must_use]
    pub const fn origin(&self) -> Timestamp {
        self.origin
    }

    /// Returns the positive elapsed duration between expected timestamps.
    #[must_use]
    pub const fn spacing(&self) -> Duration {
        self.spacing
    }

    /// Returns the inclusive maximum deviation from an expected timestamp.
    #[must_use]
    pub const fn tolerance(&self) -> Duration {
        self.tolerance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_spacing_before_checking_tolerance() {
        let error =
            FixedGrid::new(Timestamp::UNIX_EPOCH, Duration::ZERO, Duration::MAX).unwrap_err();

        assert_eq!(error, GridError::ZeroSpacing);
    }

    #[test]
    fn rejects_tolerance_at_or_above_half_the_spacing() {
        let cases = [(4, 2), (5, 3), (4, 5)];

        for (spacing_nanos, tolerance_nanos) in cases {
            let spacing = Duration::from_nanos(spacing_nanos);
            let tolerance = Duration::from_nanos(tolerance_nanos);
            let error = FixedGrid::new(Timestamp::UNIX_EPOCH, spacing, tolerance).unwrap_err();

            assert_eq!(error, GridError::AmbiguousTolerance { spacing, tolerance });
        }
    }

    #[test]
    fn preserves_valid_configuration_with_nanosecond_precision() {
        let cases = [
            (Timestamp::MIN, 1, 0),
            (Timestamp::UNIX_EPOCH, 4, 1),
            // Two nanoseconds is strictly below half of five nanoseconds.
            (Timestamp::MAX, 5, 2),
        ];

        for (origin, spacing_nanos, tolerance_nanos) in cases {
            let spacing = Duration::from_nanos(spacing_nanos);
            let tolerance = Duration::from_nanos(tolerance_nanos);
            let grid = FixedGrid::new(origin, spacing, tolerance).unwrap();

            assert_eq!(grid.origin(), origin);
            assert_eq!(grid.spacing(), spacing);
            assert_eq!(grid.tolerance(), tolerance);
        }
    }

    #[test]
    fn validates_the_full_duration_range_without_overflow() {
        // Duration::MAX is an odd number of nanoseconds. Rounding half down is
        // valid; adding one nanosecond crosses the half-spacing boundary.
        let spacing = Duration::MAX;
        let below_half = spacing / 2;
        let at_or_above_half = below_half + Duration::from_nanos(1);
        let origin = Timestamp::UNIX_EPOCH;

        let grid = FixedGrid::new(origin, spacing, below_half).unwrap();
        assert_eq!(grid.spacing(), spacing);
        assert_eq!(grid.tolerance(), below_half);

        let error = FixedGrid::new(origin, spacing, at_or_above_half).unwrap_err();
        assert_eq!(
            error,
            GridError::AmbiguousTolerance { spacing, tolerance: at_or_above_half },
        );
    }

    #[test]
    fn describes_ambiguous_tolerance_with_durations_and_units() {
        let spacing = Duration::from_nanos(4);
        let tolerance = Duration::from_nanos(2);
        let error = FixedGrid::new(Timestamp::UNIX_EPOCH, spacing, tolerance).unwrap_err();

        assert_eq!(
            error.to_string(),
            "grid tolerance (2 ns) must be less than half the spacing (4 ns)",
        );
    }
}
