//! Shared testing utilities used across the workspace.
//!
//! This crate contains reusable helpers that do not belong to any production
//! crate, including:
//!
//! - approximate floating-point comparison utilities;
//! - assertion helpers; and
//! - property-testing strategies for numerical inputs.

pub mod approx;
pub mod strategies;
