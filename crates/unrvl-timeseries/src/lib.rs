#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

pub mod align;
pub mod cadence;

mod series;
mod session;
mod time;

pub use series::{AvailabilityError, AvailabilityView, Series, SeriesError, SeriesView};
pub use session::{ScheduleError, ScheduleStatus, Session, SessionError, SessionSchedule};
pub use time::{IntervalError, TimeInterval, Timestamp, TimestampError};
