//! Empirical distribution functions for repeated sample queries.

use thiserror::Error;
use unrvl_numerics::convert::usize_to_f64;

/// An empirical distribution backed by a sorted copy of a finite sample.
///
/// Construction validates and sorts the observations once. CDF and survival
/// queries use binary search; inverse CDF queries select an observation without
/// interpolation. Queries do not allocate.
///
/// ```
/// # use unrvl_stats::distribution::empirical::Ecdf;
/// let distribution = Ecdf::new(&[10.0, 0.0, 10.0, 20.0])?;
///
/// assert_eq!(distribution.cdf(10.0), 0.75);
/// assert_eq!(distribution.sf(10.0), 0.25);
/// assert_eq!(distribution.inverse_cdf(0.5), 10.0);
/// # Ok::<(), unrvl_stats::distribution::empirical::EcdfError>(())
/// ```
#[derive(Debug, Clone)]
pub struct Ecdf {
    sorted: Vec<f64>,
}

/// An invalid sample supplied to [`Ecdf::new`].
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum EcdfError {
    /// No observations were provided.
    #[error("sample must contain at least one observation")]
    EmptySample,

    /// An observation is NaN or infinite.
    #[error("observation at index {index} must be finite, got {value}")]
    NonFiniteObservation {
        /// Zero-based index in the original sample.
        index: usize,

        /// The non-finite observation.
        value: f64,
    },
}

impl Ecdf {
    /// Creates an empirical distribution from a nonempty, finite sample.
    ///
    /// Copies and sorts the observations. For `n` observations, uses
    /// `O(n log n)` time and `O(n)` additional memory, including the retained
    /// sorted sample.
    ///
    /// # Errors
    ///
    /// Returns [`EcdfError::EmptySample`] if `xs` is empty, or
    /// [`EcdfError::NonFiniteObservation`] for the first non-finite observation
    /// in the original sample.
    ///
    /// # Panics
    ///
    /// Panics if a finite sample contains more than `2^53` observations.
    pub fn new(xs: &[f64]) -> Result<Self, EcdfError> {
        if xs.is_empty() {
            return Err(EcdfError::EmptySample);
        }

        for (index, &value) in xs.iter().enumerate() {
            if !value.is_finite() {
                return Err(EcdfError::NonFiniteObservation { index, value });
            }
        }

        // Establish exact representability of all counts before copying.
        let _ = usize_to_f64(xs.len());

        let mut sorted = xs.to_vec();
        sorted.sort_unstable_by(f64::total_cmp);

        Ok(Self { sorted })
    }

    /// Returns the fraction of observations less than or equal to `query`.
    ///
    /// Includes all observations tied at the query. Negative infinity returns
    /// zero, positive infinity returns one, and NaN returns `NaN`.
    ///
    /// Uses `O(log n)` time and `O(1)` additional memory per query, where `n`
    /// is the number of stored observations.
    #[must_use]
    pub fn cdf(&self, query: f64) -> f64 {
        if query.is_nan() {
            return f64::NAN;
        }

        let count = self.sorted.partition_point(|&x| x <= query);

        usize_to_f64(count) / usize_to_f64(self.sorted.len())
    }

    /// Returns the fraction of observations strictly greater than `query`.
    ///
    /// Excludes observations tied at the query. Negative infinity returns one,
    /// positive infinity returns zero, and NaN returns `NaN`.
    ///
    /// Computes the exceedance count before division, avoiding subtraction
    /// from a rounded cumulative probability.
    ///
    /// Uses `O(log n)` time and `O(1)` additional memory per query, where `n`
    /// is the number of stored observations.
    #[must_use]
    pub fn sf(&self, query: f64) -> f64 {
        if query.is_nan() {
            return f64::NAN;
        }

        let count = self.sorted.partition_point(|&x| x <= query);
        let exceedances = self.sorted.len() - count;

        usize_to_f64(exceedances) / usize_to_f64(self.sorted.len())
    }

    /// Returns the inverse empirical CDF at `p` using Hyndman–Fan Type 1
    /// quantiles.
    ///
    /// For `0 < p <= 1`, selects the observation at one-based rank
    /// `ceil(n * p)`, where `n` is the sample length. Values are selected
    /// without interpolation.
    ///
    /// Probabilities zero and one return the minimum and maximum, respectively.
    /// A single observation is returned for every valid probability. Returns
    /// `NaN` if `p` is non-finite or outside `[0, 1]`.
    ///
    /// The rank is computed in floating-point arithmetic. Rounding near a
    /// cumulative jump can affect which observation is selected.
    ///
    /// Uses `O(1)` time and `O(1)` additional memory per query.
    #[must_use]
    pub fn inverse_cdf(&self, p: f64) -> f64 {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return f64::NAN;
        }

        if p == 0.0 {
            return self.sorted[0];
        }

        let n = usize_to_f64(self.sorted.len());
        let h = (n * p).ceil();

        #[expect(
            clippy::cast_sign_loss,
            reason = "position is positive for a valid nonzero probability"
        )]
        #[expect(
            clippy::cast_possible_truncation,
            reason = "position is an integer within the sample length"
        )]
        let rank = h as usize;

        self.sorted[rank - 1]
    }
}

#[cfg(test)]
mod tests;
