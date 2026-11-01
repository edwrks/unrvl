use crate::Normalization;
use crate::moments::mean::mean;
use crate::moments::skewness::skewness;
use crate::moments::var::var;
use crate::moments::{Kurtosis, ShapeEstimator, kurtosis};

/// Extension methods for moment statistics on `[f64]`.
pub trait MomentsExt {
    /// Returns the arithmetic mean of a slice using Neumaier compensated
    /// summation.
    ///
    /// Equivalent to [`mean`].
    fn mean(&self) -> f64;

    /// Returns the variance of a slice with the selected normalization.
    ///
    /// Equivalent to [`var`].
    fn var(&self, normalization: Normalization) -> f64;

    /// Returns skewness with the selected Joanes-Gill estimator.
    ///
    /// Equivalent to [`skewness`].
    fn skewness(&self, estimator: ShapeEstimator) -> f64;

    /// Returns kurtosis with the selected Joanes-Gill estimator and convention.
    ///
    /// Equivalent to [`kurtosis`].
    fn kurtosis(&self, estimator: ShapeEstimator, convention: Kurtosis) -> f64;
}

impl MomentsExt for [f64] {
    fn mean(&self) -> f64 {
        mean(self)
    }

    fn var(&self, normalization: Normalization) -> f64 {
        var(self, normalization)
    }

    fn skewness(&self, estimator: ShapeEstimator) -> f64 {
        skewness(self, estimator)
    }

    fn kurtosis(&self, estimator: ShapeEstimator, convention: Kurtosis) -> f64 {
        kurtosis(self, estimator, convention)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_extension_methods_match_free_functions() {
        let xs: &[f64] = &[1.0, 2.0, 4.0, 8.0, 16.0];

        assert_eq!(xs.mean(), mean(xs));

        for normalization in [
            Normalization::Population,
            Normalization::Sample,
            Normalization::Ddof(2),
        ] {
            assert_eq!(xs.var(normalization), var(xs, normalization));
        }

        for estimator in [
            ShapeEstimator::Type1,
            ShapeEstimator::Type2,
            ShapeEstimator::Type3,
        ] {
            assert_eq!(xs.skewness(estimator), skewness(xs, estimator));

            for convention in [Kurtosis::Pearson, Kurtosis::Excess] {
                assert_eq!(
                    xs.kurtosis(estimator, convention),
                    kurtosis(xs, estimator, convention)
                );
            }
        }
    }
}
