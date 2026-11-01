//! Arithmetic mean, variance, and distribution shape statistics.

mod estimator;
mod extension;
mod kurtosis;
mod mean;
mod skewness;
mod var;

pub use estimator::ShapeEstimator;
pub use extension::MomentsExt;
pub use kurtosis::{Kurtosis, kurtosis};
pub use mean::mean;
pub use skewness::skewness;
pub use var::var;
