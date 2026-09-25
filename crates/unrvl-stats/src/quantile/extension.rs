use crate::quantile::median::median;
use crate::quantile::quantiles::quantiles;
use crate::quantile::single::quantile;

/// Extension methods for quantile statistics on `[f64]`.
pub trait QuantileExt {
    /// Returns the `p`-quantile using R's default Type 7 interpolation.
    ///
    /// Equivalent to [`quantile`].
    fn quantile(&self, p: f64) -> f64;

    /// Returns multiple quantiles using R's default Type 7 interpolation.
    ///
    /// Equivalent to [`quantiles`].
    fn quantiles(&self, ps: &[f64]) -> Vec<f64>;

    /// Returns the median using R's default Type 7 interpolation.
    ///
    /// Equivalent to [`median`].
    fn median(&self) -> f64;
}

impl QuantileExt for [f64] {
    fn quantile(&self, p: f64) -> f64 {
        quantile(self, p)
    }

    fn quantiles(&self, ps: &[f64]) -> Vec<f64> {
        quantiles(self, ps)
    }

    fn median(&self) -> f64 {
        median(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_extension_methods_match_free_functions() {
        let xs: &[f64] = &[1.0, 2.0, 4.0, 8.0, 16.0];

        assert_eq!(xs.quantile(0.75), quantile(xs, 0.75));
        assert_eq!(xs.quantiles(&[0.25, 0.5]), quantiles(xs, &[0.25, 0.5]));
        assert_eq!(xs.median(), median(xs));
    }
}
