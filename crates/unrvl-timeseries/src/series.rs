//! Validated numerical series and observation availability.

mod availability;
mod observations;

pub use availability::{AvailabilityError, AvailabilityView};
pub use observations::{Series, SeriesError, SeriesView};
