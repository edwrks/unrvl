use super::{covariance, kendall, pearson, spearman};

/// Extension methods for paired association statistics on `[f64]`.
///
/// Each method pairs `self` with `ys` by observation position and preserves
/// the corresponding free function's validation and numerical behaviour.
///
/// ```
/// # use unrvl_stats::prelude::*;
/// let xs = [1.0, 2.0, 3.0];
/// let ys = [2.0, 4.0, 6.0];
///
/// assert_eq!(xs.covariance(&ys), 2.0);
/// assert!((xs.pearson(&ys) - 1.0).abs() < 1e-14);
/// ```
pub trait AssociationExt {
    /// Returns the sample covariance with `ys`.
    ///
    /// Equivalent to [`covariance`].
    fn covariance(&self, ys: &[f64]) -> f64;

    /// Returns the Pearson correlation with `ys`.
    ///
    /// Equivalent to [`pearson`].
    fn pearson(&self, ys: &[f64]) -> f64;

    /// Returns the Spearman correlation with `ys`, using average ranks for
    /// ties.
    ///
    /// Equivalent to [`spearman`].
    fn spearman(&self, ys: &[f64]) -> f64;

    /// Returns Kendall's tau-b correlation with `ys`, accounting for ties.
    ///
    /// Equivalent to [`kendall`].
    fn kendall(&self, ys: &[f64]) -> f64;
}

impl AssociationExt for [f64] {
    fn covariance(&self, ys: &[f64]) -> f64 {
        covariance(self, ys)
    }

    fn pearson(&self, ys: &[f64]) -> f64 {
        pearson(self, ys)
    }

    fn spearman(&self, ys: &[f64]) -> f64 {
        spearman(self, ys)
    }

    fn kendall(&self, ys: &[f64]) -> f64 {
        kendall(self, ys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_extension_methods_match_free_functions() {
        let xs: &[f64] = &[1.0, 2.0, 4.0, 8.0, 16.0];
        let ys: &[f64] = &[3.0, 1.0, 5.0, 5.0, 2.0];

        let expected = [
            covariance(xs, ys),
            pearson(xs, ys),
            spearman(xs, ys),
            kendall(xs, ys),
        ];

        let result = [
            xs.covariance(ys),
            xs.pearson(ys),
            xs.spearman(ys),
            xs.kendall(ys),
        ];

        assert_eq!(result, expected);
    }

    #[test]
    fn extension_methods_preserve_invalid_input_behavior() {
        let valid: &[f64] = &[1.0, 2.0, 3.0];
        let invalid: &[f64] = &[1.0, f64::NAN, 3.0];

        for (xs, ys) in [(invalid, valid), (valid, invalid)] {
            let results = [
                xs.covariance(ys),
                xs.pearson(ys),
                xs.spearman(ys),
                xs.kendall(ys),
            ];

            for result in results {
                assert!(result.is_nan());
            }
        }
    }
}
