//! Exact timestamps and nonempty time intervals.

mod interval;
mod timestamp;

pub use interval::{IntervalError, TimeInterval};
pub use timestamp::{Timestamp, TimestampError};
