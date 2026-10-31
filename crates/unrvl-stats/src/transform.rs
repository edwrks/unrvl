//! Statistical transformations that preserve observation order.

mod demean;
mod extension;
mod ranks;
mod z_score;

pub use demean::demean;
pub use extension::TransformExt;
pub use ranks::ranks;
pub use z_score::z_score;
