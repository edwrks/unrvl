#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

pub mod align;
pub mod cadence;
pub mod resample;

mod series;
mod session;
mod time;

#[cfg(test)]
mod test_helpers;

pub use series::{AvailabilityError, AvailabilityView, Series, SeriesError, SeriesView};
pub use session::{ScheduleError, ScheduleStatus, Session, SessionError, SessionSchedule};
pub use time::{IntervalError, TimeInterval, Timestamp, TimestampError};
