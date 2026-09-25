//! Measures of spread for samples of floating-point values.

mod cv;
mod extension;
mod iqr;
mod mad;
mod range;
mod std_dev;

pub use cv::cv;
pub use extension::DispersionExt;
pub use iqr::iqr;
pub use mad::mad;
pub use range::range;
pub use std_dev::std_dev;
