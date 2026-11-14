//! Supplied sessions and schedules with explicit coverage.

mod definition;
mod schedule;

pub use definition::{Session, SessionError};
pub use schedule::{ScheduleError, ScheduleStatus, SessionSchedule};
