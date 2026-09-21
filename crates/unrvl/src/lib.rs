//! Numerical, statistical, time-series, and risk analysis tools for Rust.
//!
//! Enable the `numerics`, `stats`, `timeseries`, or `risk` feature to expose
//! the corresponding module. The `full` feature enables all four. No features
//! are enabled by default.

#[cfg(feature = "numerics")]
pub mod numerics;

#[cfg(feature = "risk")]
pub mod risk;

#[cfg(feature = "stats")]
pub mod stats;

#[cfg(feature = "timeseries")]
pub mod timeseries;
