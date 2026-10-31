/// Denominator used for unweighted variance, standard deviation, CV,
/// covariance, and z-score standardization.
///
/// For `n` observations, the divisor is `n - ddof`. These statistics
/// are undefined when `n <= ddof`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Normalization {
    /// Population normalization: use divisor `n`, equivalent to `Ddof(0)`.
    Population,

    /// Sample normalization with Bessel's correction: use divisor `n - 1`,
    /// equivalent to `Ddof(1)`.
    Sample,

    /// Divide by `n - ddof`, using a nonnegative integer adjustment.
    Ddof(usize),
}

impl Normalization {
    /// Returns a positive divisor, or `None` when there are too few
    /// observations.
    pub(crate) const fn divisor(self, n: usize) -> Option<usize> {
        let ddof = match self {
            Self::Population => 0,
            Self::Sample => 1,
            Self::Ddof(ddof) => ddof,
        };

        if n <= ddof { None } else { Some(n - ddof) }
    }
}
