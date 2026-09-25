//! Floating-point primitives for building numerical algorithms.
//!
//! This module provides compensated [`summation`] and [`dot`] operations,
//! checked integer-to-floating-point [`convert`] helpers, and [`extrema`]
//! operations for values and indices.
//!
//! Import [`prelude`] for iterator extension methods, or use the individual
//! modules directly.
//!
//! Input requirements and non-finite-value behavior vary by operation.
//! Compensation reduces rounding error but does not extend the range of `f64`.

pub use unrvl_numerics::*;
