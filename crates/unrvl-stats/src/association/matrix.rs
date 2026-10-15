use thiserror::Error;
use unrvl_numerics::convert::usize_to_f64;

use crate::internal::deviations::centered_scaled;

mod correlation;
mod covariance;

pub use correlation::correlation_matrix;
pub use covariance::covariance_matrix;

/// Invalid aligned series supplied to an association matrix statistic.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum MatrixError {
    /// The first series contains fewer than two observations.
    #[error("each series must contain at least two observations, got {count}")]
    InsufficientObservations {
        /// Number of observations in the first series.
        count: usize,
    },

    /// No variable series were provided.
    #[error("at least one variable series must be provided")]
    NoVariables,

    /// A series has a different observation count from the first series.
    #[error("series {series} must contain {expected} observations, got {actual}")]
    UnequalSeriesLength {
        /// Zero-based input series index.
        series: usize,

        /// Number of observations in the first series.
        expected: usize,

        /// Number of observations in this series.
        actual: usize,
    },

    /// An observation is NaN or infinite.
    #[error("observation {observation} in series {series} must be finite, got {value}")]
    NonFiniteObservation {
        /// Zero-based input series index.
        series: usize,

        /// Zero-based observation index within the series.
        observation: usize,

        /// The non-finite observation.
        value: f64,
    },
}

/// A series's reusable centered deviations, expressed in its scaling unit.
struct PreparedSeries {
    unit: f64,
    deviations: Vec<f64>,
}

/// Validates aligned series, then centers and scales each series once.
fn prepared_series<S: AsRef<[f64]>>(series: &[S]) -> Result<Vec<PreparedSeries>, MatrixError> {
    validate_series(series)?;

    Ok(series
        .iter()
        .map(|values| {
            let (unit, deviations) =
                centered_scaled(values.as_ref()).expect("should be nonempty and finite");
            PreparedSeries { unit, deviations: deviations.collect() }
        })
        .collect())
}

/// Validates series without copying the observations.
fn validate_series<S: AsRef<[f64]>>(series: &[S]) -> Result<(), MatrixError> {
    let first = series.first().ok_or(MatrixError::NoVariables)?;
    let n = first.as_ref().len();
    if n < 2 {
        return Err(MatrixError::InsufficientObservations { count: n });
    }

    for (series, values) in series.iter().enumerate() {
        let values = values.as_ref();
        if values.len() != n {
            return Err(MatrixError::UnequalSeriesLength {
                series,
                expected: n,
                actual: values.len(),
            });
        }

        for (observation, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(MatrixError::NonFiniteObservation { series, observation, value });
            }
        }
    }

    // Check the observation-count bound before allocating deviations.
    let _ = usize_to_f64(n);

    Ok(())
}
