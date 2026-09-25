use crate::dispersion::{cv, iqr, mad, range, std_dev};

/// Extension methods for dispersion statistics on `[f64]`.
pub trait DispersionExt {
    /// Returns the maximum observation minus the minimum.
    ///
    /// Equivalent to [`range`].
    fn range(&self) -> f64;

    /// Returns the interquartile range using R's default Type 7 quartiles.
    ///
    /// Equivalent to [`iqr`].
    fn iqr(&self) -> f64;

    /// Returns the raw median absolute deviation from the sample median.
    ///
    /// Equivalent to [`mad`].
    fn mad(&self) -> f64;

    /// Returns the sample standard deviation using Bessel's correction.
    ///
    /// Equivalent to [`std_dev`].
    fn std_dev(&self) -> f64;

    /// Returns the signed sample coefficient of variation.
    ///
    /// Equivalent to [`cv`].
    fn cv(&self) -> f64;
}

impl DispersionExt for [f64] {
    fn range(&self) -> f64 {
        range(self)
    }

    fn iqr(&self) -> f64 {
        iqr(self)
    }

    fn mad(&self) -> f64 {
        mad(self)
    }

    fn std_dev(&self) -> f64 {
        std_dev(self)
    }

    fn cv(&self) -> f64 {
        cv(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_extension_methods_match_free_functions() {
        let xs: &[f64] = &[1.0, 2.0, 4.0, 8.0, 16.0];

        let expected = [range(xs), iqr(xs), mad(xs), std_dev(xs), cv(xs)];
        let result = [xs.range(), xs.iqr(), xs.mad(), xs.std_dev(), xs.cv()];

        assert_eq!(result, expected);
    }

    #[test]
    fn extension_methods_preserve_invalid_input_behavior() {
        let xs: &[f64] = &[1.0, f64::NAN, 3.0];
        let results = [xs.range(), xs.iqr(), xs.mad(), xs.std_dev(), xs.cv()];

        for result in results {
            assert!(result.is_nan());
        }
    }
}
