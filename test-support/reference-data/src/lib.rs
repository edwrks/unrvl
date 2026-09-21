//! Bundled datasets and reference statistics for numerical and statistical
//! tests.
//!
//! [`nist`] supplies published datasets with certified statistics.
//! [`r`] supplies generated references for statistical-definition conformance.
//!
//! Loading these references does not run external statistical software.

#![warn(missing_docs)]

pub mod nist;
pub mod r;
