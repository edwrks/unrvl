#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

mod series;
mod timestamp;

pub use series::{Series, SeriesError, SeriesView};
pub use timestamp::{Timestamp, TimestampError};
