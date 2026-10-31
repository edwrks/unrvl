use unrvl_numerics::summation::NeumaierSumExt;

use super::deviations::CenteredSample;

/// A compensated sum of squared deviations in scaled units.
///
/// Keeps squared deviations in scaled units so callers can normalize before
/// restoring units or forming a dimensionless ratio.
#[derive(Debug, Clone, Copy)]
pub struct SecondMoment {
    unit: f64,
    sum_squares: f64,
}

impl SecondMoment {
    /// Prepares a nonempty finite sample in `O(n)` time and `O(1)` additional
    /// memory, using corrected centering and compensated summation.
    ///
    /// Returns `None` for empty or non-finite input.
    ///
    /// # Panics
    ///
    /// Panics if a finite sample contains more than `2^53` observations.
    pub fn new(values: &[f64]) -> Option<Self> {
        let sample = CenteredSample::new(values)?;

        Some(Self::from_centered(&sample))
    }

    /// Accumulates an already prepared sample in one pass over its deviations.
    /// Reuses its centering without copying observations or allocating.
    pub fn from_centered(sample: &CenteredSample<'_>) -> Self {
        let unit = sample.centering().unit();

        let sum_squares = sample
            .deviations()
            .map(|deviation| deviation * deviation)
            .neumaier_sum();

        Self { unit, sum_squares }
    }

    /// Positive, finite power-of-two scaling unit.
    pub const fn unit(&self) -> f64 {
        self.unit
    }

    /// Sum of squared deviations in scaled units, before normalization.
    pub const fn scaled_sum_squares(&self) -> f64 {
        self.sum_squares
    }
}
