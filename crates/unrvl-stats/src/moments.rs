//! Moments and shape statistics.
//!
//! This module contains mean, sample variance, bias-corrected sample skewness,
//! and bias-corrected sample excess kurtosis.

mod extension;
mod kurtosis;
mod mean;
mod skewness;
mod var;

pub use extension::MomentsExt;
pub use kurtosis::excess_kurtosis;
pub use mean::mean;
pub use skewness::skewness;
pub use var::var;
