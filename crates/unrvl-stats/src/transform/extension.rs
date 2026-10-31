use crate::transform::{demean, ranks, z_score};

/// Extension methods for statistical transformations on `[f64]`.
///
/// ```
/// use unrvl_stats::prelude::*;
///
/// let xs = [1.0, 2.0, 3.0];
/// assert_eq!(xs.demean(), vec![-1.0, 0.0, 1.0]);
/// assert_eq!(xs.z_score(), vec![-1.0, 0.0, 1.0]);
/// assert_eq!(xs.ranks(), vec![1.0, 2.0, 3.0]);
/// ```
pub trait TransformExt {
    /// Returns deviations from the sample mean in original observation order.
    ///
    /// Equivalent to [`demean`].
    fn demean(&self) -> Vec<f64>;

    /// Returns z-scores using the sample standard deviation.
    ///
    /// Equivalent to [`z_score`].
    fn z_score(&self) -> Vec<f64>;

    /// Returns one-based average ranks in original observation order.
    ///
    /// Equivalent to [`ranks`].
    fn ranks(&self) -> Vec<f64>;
}

impl TransformExt for [f64] {
    fn demean(&self) -> Vec<f64> {
        demean(self)
    }

    fn z_score(&self) -> Vec<f64> {
        z_score(self)
    }

    fn ranks(&self) -> Vec<f64> {
        ranks(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extension_methods_match_free_functions() {
        let xs: &[f64] = &[1.0, 2.0, 4.0, 8.0, 16.0];

        assert_eq!(xs.demean(), demean(xs));
        assert_eq!(xs.z_score(), z_score(xs));
        assert_eq!(xs.ranks(), ranks(xs));
    }

    #[test]
    fn extension_methods_preserve_invalid_input_behavior() {
        let xs: &[f64] = &[1.0, f64::NAN, 3.0];

        let results = [xs.demean(), xs.z_score(), xs.ranks()];

        for result in results {
            assert_eq!(result.len(), xs.len());
            assert!(result.into_iter().all(f64::is_nan));
        }
    }
}
