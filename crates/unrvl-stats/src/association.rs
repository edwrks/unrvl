//! Association statistics for paired observations and observation matrices.

mod covariance;
mod extension;
mod kendall;
mod matrix;
mod pearson;
mod spearman;

pub use covariance::covariance;
pub use extension::AssociationExt;
pub use kendall::kendall;
pub use matrix::{MatrixError, correlation_matrix, covariance_matrix};
pub use pearson::pearson;
pub use spearman::spearman;
