/// Joanes-Gill convention for estimating distribution shape.
///
/// [`skewness`](super::skewness) and [`kurtosis`](super::kurtosis) use this
/// selection to choose their standardized moment definition and finite-sample
/// adjustment.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeEstimator {
    /// Ratio of empirical central moments, each using divisor `n`.
    ///
    /// Both statistics require at least two observations with nonzero spread.
    Type1,

    /// Finite-sample adjusted skewness and kurtosis.
    ///
    /// Requires at least three observations for skewness and four for kurtosis,
    /// with nonzero spread.
    Type2,

    /// Standardization using the sample standard deviation.
    ///
    /// Uses divisor `n - 1` and requires at least two observations with nonzero
    /// spread for either statistic.
    Type3,
}
