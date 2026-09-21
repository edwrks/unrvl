//! Quantiles and medians.
//!
//! Quantiles use R's default Type 7 definition. Input slices need not be
//! sorted.

mod extension;
mod internal;
mod median;
mod quantiles;
mod single;

pub use extension::QuantileExt;
pub use median::median;
pub use quantiles::quantiles;
pub use single::quantile;
