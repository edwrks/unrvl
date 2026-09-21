use crate::moments::excess_kurtosis;
use crate::moments::mean::mean;
use crate::moments::skewness::skewness;
use crate::moments::var::var;

/// Extension methods for moment statistics on `[f64]`.
pub trait MomentsExt {
    /// Returns the arithmetic mean of a slice using Neumaier compensated
    /// summation.
    ///
    /// Equivalent to [`mean`].
    fn mean(&self) -> f64;

    /// Returns the sample variance of a slice using Bessel's correction.
    ///
    /// Equivalent to [`var`].
    fn var(&self) -> f64;

    /// Returns the bias-corrected sample skewness G1 (Joanes & Gill 1998 type
    /// 2).
    ///
    /// Equivalent to [`skewness`].
    fn skewness(&self) -> f64;

    /// Returns the bias-corrected sample excess kurtosis G2 (Joanes & Gill 1998
    /// type 2).
    ///
    /// Equivalent to [`excess_kurtosis`].
    fn excess_kurtosis(&self) -> f64;
}

impl MomentsExt for [f64] {
    fn mean(&self) -> f64 {
        mean(self)
    }

    fn var(&self) -> f64 {
        var(self)
    }

    fn skewness(&self) -> f64 {
        skewness(self)
    }

    fn excess_kurtosis(&self) -> f64 {
        excess_kurtosis(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_extension_methods_match_free_functions() {
        let xs: &[f64] = &[1.0, 2.0, 4.0, 8.0, 16.0];

        assert_eq!(xs.mean(), mean(xs));
        assert_eq!(xs.var(), var(xs));
        assert_eq!(xs.skewness(), skewness(xs));
        assert_eq!(xs.excess_kurtosis(), excess_kurtosis(xs));
    }
}
