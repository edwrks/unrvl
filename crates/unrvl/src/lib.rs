//! Quantitative analysis tools for Rust.

#[cfg(feature = "numerics")]
pub mod numerics;

#[cfg(feature = "risk")]
pub mod risk;

#[cfg(feature = "stats")]
pub mod stats;

#[cfg(feature = "timeseries")]
pub mod timeseries;
