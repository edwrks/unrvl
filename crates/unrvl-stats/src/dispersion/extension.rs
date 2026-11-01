use crate::Normalization;
use crate::dispersion::{MadScale, cv, iqr, mad, range, std_dev};

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

    /// Returns the median absolute deviation with the selected scaling.
    ///
    /// Equivalent to [`mad`].
    fn mad(&self, scale: MadScale) -> f64;

    /// Returns the standard deviation with the selected normalization.
    ///
    /// Equivalent to [`std_dev`].
    fn std_dev(&self, normalization: Normalization) -> f64;

    /// Returns the signed coefficient of variation with the selected
    /// normalization.
    ///
    /// Equivalent to [`cv`].
    fn cv(&self, normalization: Normalization) -> f64;
}

impl DispersionExt for [f64] {
    fn range(&self) -> f64 {
        range(self)
    }

    fn iqr(&self) -> f64 {
        iqr(self)
    }

    fn mad(&self, scale: MadScale) -> f64 {
        mad(self, scale)
    }

    fn std_dev(&self, normalization: Normalization) -> f64 {
        std_dev(self, normalization)
    }

    fn cv(&self, normalization: Normalization) -> f64 {
        cv(self, normalization)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extension_methods_preserve_invalid_input_behavior() {
        let xs: &[f64] = &[1.0, f64::NAN, 3.0];
        let results = [xs.range(), xs.iqr()];

        for result in results {
            assert!(result.is_nan());
        }

        for scale in [MadScale::Raw, MadScale::Normal] {
            assert!(xs.mad(scale).is_nan());
        }

        for normalization in [
            Normalization::Population,
            Normalization::Sample,
            Normalization::Ddof(2),
        ] {
            assert!(xs.std_dev(normalization).is_nan());
            assert!(xs.cv(normalization).is_nan());
        }
    }

    #[test]
    fn public_extension_methods_match_free_functions() {
        let xs: &[f64] = &[1.0, 2.0, 4.0, 8.0, 16.0];

        assert_eq!(xs.range(), range(xs));
        assert_eq!(xs.iqr(), iqr(xs));

        for scale in [MadScale::Raw, MadScale::Normal] {
            assert_eq!(xs.mad(scale), mad(xs, scale));
        }

        for normalization in [
            Normalization::Population,
            Normalization::Sample,
            Normalization::Ddof(2),
        ] {
            assert_eq!(xs.std_dev(normalization), std_dev(xs, normalization));
            assert_eq!(xs.cv(normalization), cv(xs, normalization));
        }
    }
}
