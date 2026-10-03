//! R conformance fixtures and references for association statistics.
//!
//! [`paired`] loads two aligned observation vectors and scalar covariance,
//! Pearson, Spearman, and Kendall tau-b references. [`matrix`] loads named
//! variables and indexed covariance and Pearson correlation matrix entries.
//!
//! Each module provides its own `Dataset`, `Reference`, and `Statistics` types.
//! Fixtures and reference values are embedded; loading does not run R or
//! require network access.

pub mod matrix;
pub mod paired;
