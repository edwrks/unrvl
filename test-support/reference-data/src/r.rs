//! Committed R references for statistical-definition conformance.
//!
//! These references use small, well-conditioned fixtures to check estimator
//! conventions against R and `e1071`.
//!
//! Loading references does not invoke R or Docker. Undefined-input policies
//! and floating-point stress cases are tested separately in Rust.

pub mod moments;
pub mod quantiles;
