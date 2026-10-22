//! Ordinary least squares with one predictor and an intercept.

use thiserror::Error;
use unrvl_numerics::scaling::{binary_exponent, scale_power_of_two};
use unrvl_numerics::summation::NeumaierSum;

use crate::internal::deviations::{CenteredSample, Centering};

/// Invalid paired observations or an unrepresentable simple linear fit.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum SimpleLinearError {
    /// Predictor and response slices have different lengths.
    #[error("inputs must have equal lengths, got x={x} and y={y}")]
    LengthMismatch {
        /// Number of predictor observations.
        x: usize,

        /// Number of response observations.
        y: usize,
    },

    /// Fewer than two observation pairs were provided.
    #[error("fit requires at least two observations, got {count}")]
    InsufficientObservations {
        /// Number of supplied pairs.
        count: usize,
    },

    /// A predictor is NaN or infinite.
    #[error("predictor at index {index} must be finite, got {value}")]
    NonFinitePredictor {
        /// Zero-based observation index.
        index: usize,

        /// The non-finite predictor.
        value: f64,
    },

    /// A response is NaN or infinite.
    #[error("response at index {index} must be finite, got {value}")]
    NonFiniteResponse {
        /// Zero-based observation index.
        index: usize,

        /// The non-finite response.
        value: f64,
    },

    /// A constant predictor cannot identify both slope and intercept.
    #[error("predictor must contain at least two distinct values")]
    ConstantPredictor,

    /// Coefficients or required intermediate statistics cannot be represented.
    #[error("fit cannot be represented with finite f64 coefficients and intermediates")]
    NumericalFailure,
}

/// A fitted line `y = slope * x + intercept` and its fit statistics.
///
/// Uses ordinary least squares with an intercept. Stores the fitted parameters
/// without retaining the input observations.
///
/// ```
/// # use unrvl_stats::regression::simple_linear::{SimpleLinearError, SimpleLinearRegression};
/// let xs = [1.0, 2.0, 3.0];
/// let ys = [3.0, 5.0, 7.0];
/// let model = SimpleLinearRegression::fit(&xs, &ys)?;
///
/// assert_eq!(model.slope(), 2.0);
/// assert_eq!(model.intercept(), 1.0);
/// assert_eq!(model.observation_count(), 3);
/// assert_eq!(model.predict(4.0), 9.0);
/// # Ok::<(), SimpleLinearError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SimpleLinearRegression {
    slope: f64,
    intercept: f64,
    r_squared: f64,
    observation_count: usize,
    x_center: Centering,
    y_center: Centering,
    scaled_slope: f64,
}

impl SimpleLinearRegression {
    /// Fits `y = slope * x + intercept` by minimizing the sum of squared
    /// residuals.
    ///
    /// `xs` contains predictors and `ys` contains the corresponding responses,
    /// paired by position. Requires at least two finite pairs and at least two
    /// distinct predictor values. No pairs are discarded.
    ///
    /// A constant response gives a horizontal line with zero slope and
    /// undefined (`NaN`) [`R²`](Self::r_squared).
    ///
    /// Scaling and compensated arithmetic reduce numerical error but do not
    /// eliminate rounding or underflow. Very small contributions relative to
    /// each input scale can be lost.
    ///
    /// Uses `O(n)` time over several passes and `O(1)` additional memory.
    ///
    /// # Errors
    ///
    /// Returns an input error for unequal lengths, fewer than two pairs,
    /// non-finite observations, or a constant predictor. Returns
    /// [`SimpleLinearError::NumericalFailure`] for non-finite coefficients
    /// or unusable intermediate results.
    ///
    /// # Panics
    ///
    /// Panics if a finite, nonconstant predictor contains more than `2^53`
    /// observations.
    pub fn fit(xs: &[f64], ys: &[f64]) -> Result<Self, SimpleLinearError> {
        validate_fit(xs, ys)?;

        let n = xs.len();

        // Center and scale both samples to limit overflow and precision loss.
        let x_sample = CenteredSample::new(xs).expect("validated nonempty finite predictors");
        let y_sample = CenteredSample::new(ys).expect("validated nonempty finite responses");
        let x_center = x_sample.centering();
        let y_center = y_sample.centering();

        // Accumulate the variance and covariance sums in scaled units.
        // Their common (n - 1) divisor cancels when calculating sxy / sxx.
        let mut sxx = NeumaierSum::new();
        let mut sxy = NeumaierSum::new();
        let mut syy = NeumaierSum::new();

        for (dx, dy) in x_sample.deviations().zip(y_sample.deviations()) {
            sxx.add_product(dx, dx);
            sxy.add_product(dx, dy);
            syy.add_product(dy, dy);
        }

        let sxx = sxx.total();
        let sxy = sxy.total();
        let syy = syy.total();

        if !sxx.is_finite() || sxx <= 0.0 || !syy.is_finite() {
            return Err(SimpleLinearError::NumericalFailure);
        }

        // Slope = Sxy / Sxx, initially in the scaled coordinates.
        let scaled_slope = sxy / sxx;

        if !scaled_slope.is_finite() {
            return Err(SimpleLinearError::NumericalFailure);
        }

        // Restore the slope's original units: scaled_slope * y_unit / x_unit.
        // Combining exponents avoids an overflowing or underflowing unit ratio.
        let x_exponent = binary_exponent(x_center.unit()).expect("positive finite scaling unit");
        let y_exponent = binary_exponent(y_center.unit()).expect("positive finite scaling unit");
        let scale_exponent = y_exponent - x_exponent;

        let slope = scale_power_of_two(scaled_slope, scale_exponent);

        // Intercept = mean_y - slope * mean_x, evaluated in scaled units.
        // Each mean has a main part and a small correction. Apply the formula
        // to both parts separately to avoid losing the corrections.
        let intercept_main =
            (-scaled_slope).mul_add(x_center.scaled_center(), y_center.scaled_center());

        let intercept_correction =
            (-scaled_slope).mul_add(x_center.scaled_correction(), y_center.scaled_correction());

        let intercept = (intercept_main + intercept_correction) * y_center.unit();

        if !slope.is_finite() || !intercept.is_finite() {
            return Err(SimpleLinearError::NumericalFailure);
        }

        // For OLS with an intercept, R² is the squared correlation. The input
        // scales cancel, so we can reuse the centered sums above.
        let r_squared = if syy == 0.0 {
            // A constant response has no variation, so R² is undefined.
            f64::NAN
        } else {
            let denominator = sxx.sqrt() * syy.sqrt();
            let correlation = sxy / denominator;
            if !correlation.is_finite() {
                return Err(SimpleLinearError::NumericalFailure);
            }

            let correlation = correlation.clamp(-1.0, 1.0);
            correlation * correlation
        };

        Ok(Self {
            slope,
            intercept,
            r_squared,
            observation_count: n,
            x_center,
            y_center,
            scaled_slope,
        })
    }

    /// Returns the fitted slope in response units per predictor unit.
    ///
    /// This coefficient may underflow to zero. [`Self::predict`] uses
    /// additional fit information and can preserve variation even when this
    /// happens.
    #[must_use]
    pub const fn slope(&self) -> f64 {
        self.slope
    }

    /// Returns the fitted intercept coefficient in response units.
    #[must_use]
    pub const fn intercept(&self) -> f64 {
        self.intercept
    }

    /// Returns R² for the observations used to fit this model.
    ///
    /// Calculated as the squared Pearson correlation, with correlation clamped
    /// to `[-1, 1]` for rounding. This is mathematically equivalent to
    /// `1 - SSE / SST` for OLS with an intercept, where SSE is the sum of
    /// squared residuals and SST is the sum of squared deviations from the
    /// response mean. Returns a value in `[0, 1]`, or `NaN` for a constant
    /// response. A perfect fit can round slightly below one.
    ///
    /// Calls to [`Self::predict`] and [`Self::residuals`] do not update this
    /// statistic.
    #[must_use]
    pub const fn r_squared(&self) -> f64 {
        self.r_squared
    }

    /// Returns the number of pairs used to fit this model.
    #[must_use]
    pub const fn observation_count(&self) -> usize {
        self.observation_count
    }

    /// Returns the fitted response at predictor `x`.
    ///
    /// Mathematically evaluates `slope * x + intercept`. Uses the retained
    /// centered fit to reduce rounding error, particularly for predictors with
    /// a large offset. The result may differ from evaluating
    /// `self.slope() * x + self.intercept()` directly.
    ///
    /// Returns `NaN` for a non-finite query. Results are subject to rounding
    /// and underflow, and may overflow to signed infinity.
    ///
    /// Uses `O(1)` time and additional memory.
    #[must_use]
    pub fn predict(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return f64::NAN;
        }

        if self.scaled_slope == 0.0 {
            return self.intercept;
        }

        // Substituting intercept = mean_y - slope * mean_x gives the same line:
        // mean_y + slope * (x - mean_x), evaluated here in scaled units.
        // This avoids cancellation between large slope * x and intercept terms.
        // Add the mean correction last to preserve small contributions.
        let dx = self.x_center.deviation(x);
        let scaled_mean_y = self.y_center.scaled_center();
        let scaled_mean_y_correction = self.y_center.scaled_correction();
        let scaled_prediction =
            self.scaled_slope.mul_add(dx, scaled_mean_y) + scaled_mean_y_correction;

        if scaled_prediction.is_finite() {
            return scaled_prediction * self.y_center.unit();
        }

        // If scaled arithmetic overflows, try slope * x + intercept directly.
        self.slope.mul_add(x, self.intercept)
    }

    /// Returns observed response minus fitted response for each supplied pair.
    ///
    /// `xs` and `ys` are paired by position, and results follow that order.
    /// These observations may differ from those used by [`Self::fit`]. The
    /// model is not refitted, the observations are not retained, and its
    /// fit statistics are not updated. Empty paired slices return an empty
    /// vector.
    ///
    /// Computes residuals directly to preserve small differences. Results may
    /// differ from `ys[i] - self.predict(xs[i])` because a separately returned
    /// prediction has already been rounded. Results remain subject to rounding
    /// and underflow, and may overflow to signed infinity.
    ///
    /// Uses `O(m)` time and `O(m)` memory for the returned vector, where `m`
    /// is the supplied pair count.
    ///
    /// # Errors
    ///
    /// Returns [`SimpleLinearError::LengthMismatch`] if the lengths differ,
    /// otherwise the first non-finite observation error in pair order
    /// (predictor before response). No pairs are discarded.
    ///
    /// # Examples
    ///
    /// ```
    /// # use unrvl_stats::regression::simple_linear::{SimpleLinearError, SimpleLinearRegression};
    /// let model = SimpleLinearRegression::fit(&[1.0, 2.0, 3.0], &[3.0, 5.0, 7.0])?;
    ///
    /// assert_eq!(model.predict(4.0), 9.0);
    /// assert_eq!(model.residuals(&[4.0], &[10.0])?, vec![1.0]);
    /// # Ok::<(), SimpleLinearError>(())
    /// ```
    pub fn residuals(&self, xs: &[f64], ys: &[f64]) -> Result<Vec<f64>, SimpleLinearError> {
        validate_residuals(xs, ys)?;

        Ok(xs
            .iter()
            .zip(ys)
            .map(|(&x, &y)| self.evaluate_residual(x, y))
            .collect())
    }

    fn evaluate_residual(&self, x: f64, y: f64) -> f64 {
        // In scaled units, residual = dy - scaled_slope * dx.
        // (-m).mul_add(dx, dy) computes dy - m * dx with one rounding.
        let dx = self.x_center.deviation(x);
        let dy = self.y_center.deviation(y);
        let scaled_residual = (-self.scaled_slope).mul_add(dx, dy);

        if scaled_residual.is_finite() {
            return scaled_residual * self.y_center.unit();
        }

        // Try y - slope * x - intercept in original units.
        // This can stay finite even when slope * x would overflow.
        let residual_without_intercept = (-self.slope).mul_add(x, y);
        residual_without_intercept - self.intercept
    }
}

#[expect(clippy::float_cmp, reason = "detect exactly constant predictor")]
fn validate_fit(xs: &[f64], ys: &[f64]) -> Result<(), SimpleLinearError> {
    if xs.len() != ys.len() {
        return Err(SimpleLinearError::LengthMismatch { x: xs.len(), y: ys.len() });
    }

    let count = xs.len();
    if count < 2 {
        return Err(SimpleLinearError::InsufficientObservations { count });
    }

    for (index, (&x, &y)) in xs.iter().zip(ys).enumerate() {
        if !x.is_finite() {
            return Err(SimpleLinearError::NonFinitePredictor { index, value: x });
        }

        if !y.is_finite() {
            return Err(SimpleLinearError::NonFiniteResponse { index, value: y });
        }
    }

    if xs.iter().all(|&x| x == xs[0]) {
        return Err(SimpleLinearError::ConstantPredictor);
    }

    Ok(())
}

fn validate_residuals(xs: &[f64], ys: &[f64]) -> Result<(), SimpleLinearError> {
    if xs.len() != ys.len() {
        return Err(SimpleLinearError::LengthMismatch { x: xs.len(), y: ys.len() });
    }

    for (index, (&x, &y)) in xs.iter().zip(ys).enumerate() {
        if !x.is_finite() {
            return Err(SimpleLinearError::NonFinitePredictor { index, value: x });
        }

        if !y.is_finite() {
            return Err(SimpleLinearError::NonFiniteResponse { index, value: y });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_utils::approx::{Tolerance, approx_eq, assert_all_approx_eq, assert_approx_eq};
    use test_utils::strategies::finite_paired_samples;

    use super::{SimpleLinearError, SimpleLinearRegression};

    #[expect(clippy::float_cmp, reason = "detect exact sample constancy")]
    fn nonconstant(xs: &[f64]) -> bool {
        xs.windows(2).any(|pair| pair[0] != pair[1])
    }

    mod fit {
        use test_utils::strategies::exact_translation_case;

        use super::*;

        #[test]
        fn fits_positive_and_negative_slopes() {
            let xs = [-2.0, 0.0, 1.0, 3.0];
            let intercept = 3.0;

            for slope in [-2.0_f64, 2.0] {
                let ys = xs.map(|x| slope.mul_add(x, intercept));

                let model = SimpleLinearRegression::fit(&xs, &ys).unwrap();

                assert_approx_eq(model.slope(), slope, Tolerance::VERY_STRICT);
                assert_approx_eq(model.intercept(), intercept, Tolerance::VERY_STRICT);
                assert_eq!(model.observation_count(), xs.len());
                assert_approx_eq(model.r_squared(), 1.0, Tolerance::VERY_STRICT);
            }
        }

        #[test]
        fn fits_two_distinct_pairs() {
            let xs = [1.0, 3.0];
            let ys = [8.0, 2.0];

            let model = SimpleLinearRegression::fit(&xs, &ys).unwrap();

            assert_eq!(model.slope(), -3.0);
            assert_eq!(model.intercept(), 11.0);
            assert_approx_eq(model.r_squared(), 1.0, Tolerance::VERY_STRICT);
        }

        #[test]
        fn accepts_repeated_predictors_when_the_sample_varies() {
            let xs = [0.0, 0.0, 2.0, 2.0];
            let ys = [0.0, 2.0, 2.0, 4.0];

            let model = SimpleLinearRegression::fit(&xs, &ys).unwrap();

            assert_eq!(model.slope(), 1.0);
            assert_eq!(model.intercept(), 1.0);
            assert_approx_eq(model.r_squared(), 0.5, Tolerance::VERY_STRICT);
        }

        #[test]
        fn preserves_constant_responses_at_extreme_scales() {
            let xs = [-f64::MAX, 0.0, f64::MAX];
            let values = [0.0, f64::from_bits(1), f64::MAX, -f64::MAX];

            for value in values {
                let ys = [value; 3];

                let model = SimpleLinearRegression::fit(&xs, &ys).unwrap();

                assert_eq!(model.slope(), 0.0);
                assert_eq!(model.intercept(), value);
                assert_eq!(model.predict(f64::MAX), value);
                assert!(model.r_squared().is_nan());
            }
        }

        #[test]
        fn rejects_fewer_than_two_pairs() {
            for count in [0, 1] {
                let xs = vec![1.0; count];
                let ys = vec![2.0; count];

                let error = SimpleLinearRegression::fit(&xs, &ys).unwrap_err();

                assert_eq!(error, SimpleLinearError::InsufficientObservations { count });
            }
        }

        #[test]
        fn checks_lengths_before_count_or_values() {
            let xs = [];
            let ys = [f64::NAN];

            let error = SimpleLinearRegression::fit(&xs, &ys).unwrap_err();

            assert_eq!(error, SimpleLinearError::LengthMismatch { x: 0, y: 1 });
        }

        #[test]
        fn checks_pair_count_before_values() {
            let xs = [f64::NAN];
            let ys = [f64::NAN];

            let error = SimpleLinearRegression::fit(&xs, &ys).unwrap_err();

            assert_eq!(
                error,
                SimpleLinearError::InsufficientObservations { count: 1 }
            );
        }

        #[test]
        fn rejects_nonfinite_predictors() {
            for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let xs = [0.0, value];
                let ys = [0.0, 1.0];

                let error = SimpleLinearRegression::fit(&xs, &ys).unwrap_err();

                let SimpleLinearError::NonFinitePredictor { index, value: actual } = error else {
                    panic!("expected predictor error");
                };

                assert_eq!(index, 1);
                assert_eq!(actual.to_bits(), value.to_bits());
            }
        }

        #[test]
        fn rejects_nonfinite_responses() {
            for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let xs = [0.0, 1.0];
                let ys = [0.0, value];

                let error = SimpleLinearRegression::fit(&xs, &ys).unwrap_err();

                let SimpleLinearError::NonFiniteResponse { index, value: actual } = error else {
                    panic!("expected response error");
                };

                assert_eq!(index, 1);
                assert_eq!(actual.to_bits(), value.to_bits());
            }
        }

        #[test]
        fn checks_predictor_before_response_in_each_pair() {
            let xs = [0.0, f64::NAN];
            let ys = [0.0, f64::INFINITY];

            let error = SimpleLinearRegression::fit(&xs, &ys).unwrap_err();

            assert!(matches!(
                error,
                SimpleLinearError::NonFinitePredictor { index: 1, .. }
            ));
        }

        #[test]
        fn checks_nonfinite_pairs_in_index_order() {
            let xs = [0.0, f64::NAN];
            let ys = [f64::NAN, 1.0];

            let error = SimpleLinearRegression::fit(&xs, &ys).unwrap_err();

            assert!(matches!(
                error,
                SimpleLinearError::NonFiniteResponse { index: 0, .. }
            ));
        }

        #[test]
        fn checks_values_before_constant_predictor() {
            let xs = [0.0, 0.0];
            let ys = [f64::NAN, 1.0];

            let error = SimpleLinearRegression::fit(&xs, &ys).unwrap_err();

            assert!(matches!(
                error,
                SimpleLinearError::NonFiniteResponse { index: 0, .. }
            ));
        }

        #[test]
        fn rejects_constant_predictors_including_signed_zero_and_extremes() {
            let values = [0.0, f64::from_bits(1), f64::MAX, -f64::MAX];
            let ys = [0.0, 1.0, 2.0];

            for value in values {
                let xs = [value; 3];

                let error = SimpleLinearRegression::fit(&xs, &ys).unwrap_err();

                assert_eq!(error, SimpleLinearError::ConstantPredictor);
            }

            let xs = [-0.0, 0.0];
            let ys = [0.0, 1.0];

            let error = SimpleLinearRegression::fit(&xs, &ys).unwrap_err();

            assert_eq!(error, SimpleLinearError::ConstantPredictor);
        }

        #[test]
        fn fits_opposite_extremes_without_forming_unscaled_variances() {
            let xs = [-f64::MAX, 0.0, f64::MAX];
            let ys = [f64::MAX, 0.0, -f64::MAX];

            let model = SimpleLinearRegression::fit(&xs, &ys).unwrap();

            let predictions = xs.map(|x| model.predict(x));
            let residuals = model.residuals(&xs, &ys).unwrap();

            assert_eq!(model.slope(), -1.0);
            assert_eq!(model.intercept(), 0.0);
            assert_approx_eq(model.r_squared(), 1.0, Tolerance::VERY_STRICT);
            assert_eq!(predictions, ys);
            assert_eq!(residuals, vec![0.0; 3]);
        }

        #[test]
        fn recovers_finite_slope_when_the_unit_ratio_overflows() {
            let x_unit = 2.0_f64.powi(-600);
            let y_unit = 2.0_f64.powi(470);

            let xs = [x_unit, -x_unit, x_unit, -x_unit];
            let ys = [y_unit, y_unit, -y_unit, -y_unit + 2.0_f64.powi(421)];

            let model = SimpleLinearRegression::fit(&xs, &ys).unwrap();

            assert_eq!(model.slope(), -2.0_f64.powi(1019));
            assert_eq!(model.intercept(), 2.0_f64.powi(419));
        }

        #[test]
        fn recovers_subnormal_slope_when_the_unit_ratio_underflows() {
            let base = 2.0_f64.powi(600);
            let step = 2.0_f64.powi(548);
            let small = 2.0_f64.powi(-500);

            let xs = [base, base + step];
            let ys = [0.0, small];

            let model = SimpleLinearRegression::fit(&xs, &ys).unwrap();

            let predictions = xs.map(|x| model.predict(x));
            let expected_slope = f64::MIN_POSITIVE / 2.0_f64.powi(26);

            assert_eq!(model.slope(), expected_slope);
            assert_eq!(model.intercept(), -2.0_f64.powi(-448));
            assert_eq!(predictions, ys);
        }

        #[test]
        fn rejects_overflowing_slope_or_intercept() {
            let tiny = f64::from_bits(1);
            let cases = [
                ([0.0, tiny], [0.0, 1.0]),
                ([2.0, 3.0], [f64::MAX / 2.0, 0.0]),
            ];

            for (xs, ys) in cases {
                let error = SimpleLinearRegression::fit(&xs, &ys).unwrap_err();

                assert_eq!(error, SimpleLinearError::NumericalFailure);
            }
        }

        proptest! {
            #[test]
            fn pair_reordering_preserves_fit((mut xs, mut ys) in finite_paired_samples(3)) {
                prop_assume!(nonconstant(&xs));

                let original = SimpleLinearRegression::fit(&xs, &ys).unwrap();

                xs.reverse();
                ys.reverse();

                let reordered = SimpleLinearRegression::fit(&xs, &ys).unwrap();

                prop_assert!(approx_eq(
                    original.slope(),
                    reordered.slope(),
                    Tolerance::STRICT,
                ));

                prop_assert!(approx_eq(
                    original.intercept(),
                    reordered.intercept(),
                    Tolerance::STRICT,
                ));

                if original.r_squared().is_nan() {
                    prop_assert!(reordered.r_squared().is_nan());
                } else {
                    prop_assert!(approx_eq(
                        original.r_squared(),
                        reordered.r_squared(),
                        Tolerance::STRICT,
                    ));
                }
            }

            #[test]
            fn affine_transformations_change_coefficients_as_expected(
                (xs, offset) in exact_translation_case(3),
            ) {
                let ys: Vec<_> = xs.iter().rev().copied().collect();

                let model = SimpleLinearRegression::fit(&xs, &ys).unwrap();

                let shifted_xs: Vec<_> = xs.iter()
                    .map(|&x| (-2.0_f64).mul_add(x, offset))
                    .collect();

                let shifted_ys: Vec<_> = ys.iter()
                    .map(|&y| 4.0_f64.mul_add(y, -offset))
                    .collect();

                let shifted = SimpleLinearRegression::fit(&shifted_xs, &shifted_ys).unwrap();

                // Transform the response, then account for the predictor's offset.
                let expected_slope = -2.0 * model.slope();
                let response_intercept = 4.0_f64.mul_add(model.intercept(), -offset);
                let expected_intercept = (-expected_slope).mul_add(offset, response_intercept);

                prop_assert!(approx_eq(shifted.slope(), expected_slope, Tolerance::STRICT));

                prop_assert!(approx_eq(
                    shifted.intercept(),
                    expected_intercept,
                    Tolerance::new(1e-10, 1e-14),
                ));

                prop_assert!(approx_eq(shifted.r_squared(), model.r_squared(), Tolerance::STRICT));
            }
        }
    }

    mod r_squared {
        use super::*;
        use crate::association::pearson;

        #[test]
        fn is_undefined_for_a_constant_response() {
            let xs = [-2.0, 0.0, 1.0, 3.0];
            let ys = [3.0; 4];

            let model = SimpleLinearRegression::fit(&xs, &ys).unwrap();

            assert_eq!(model.slope(), 0.0);
            assert_eq!(model.intercept(), 3.0);
            assert!(model.r_squared().is_nan());
        }

        #[test]
        fn is_zero_for_zero_covariance() {
            let xs = [-1.0, -1.0, 1.0, 1.0];
            let ys = [0.0, 2.0, 0.0, 2.0];

            let model = SimpleLinearRegression::fit(&xs, &ys).unwrap();

            assert_eq!(model.slope(), 0.0);
            assert_eq!(model.intercept(), 1.0);
            assert_eq!(model.r_squared(), 0.0);
        }

        #[test]
        fn preserves_a_weak_relationship() {
            let delta = 2.0_f64.powi(-26);
            let xs = [-1.0, 1.0, -1.0, 1.0];
            let ys = [-1.0, -1.0, 1.0, 1.0 + delta];

            let model = SimpleLinearRegression::fit(&xs, &ys).unwrap();

            // Sxx = 4, Sxy = delta, Syy = 4 + 2*delta + 3*delta²/4.
            // This small positive R² is lost by subtracting SSE / SST from one.
            let delta_squared = delta * delta;
            let denominator = 8.0_f64.mul_add(delta, 16.0);
            let denominator = 3.0_f64.mul_add(delta_squared, denominator);
            let expected = delta_squared / denominator;

            assert_approx_eq(model.r_squared(), expected, Tolerance::new(0.0, 1e-15));
        }

        #[test]
        fn remains_accurate_near_a_perfect_fit() {
            let xs = [-1.0, 0.0, 1.0];

            for delta in [0.0, 2.0_f64.powi(-20)] {
                let ys = [-1.0, delta, 1.0];

                let model = SimpleLinearRegression::fit(&xs, &ys).unwrap();

                // The slope is one and SSE = 2*delta²/3, giving this exact
                // formula.
                let expected = 1.0 / (1.0 + delta * delta / 3.0);

                assert_approx_eq(model.r_squared(), expected, Tolerance::VERY_STRICT);
            }
        }

        proptest! {
            #[test]
            fn agrees_with_squared_pearson((xs, ys) in finite_paired_samples(3)) {
                prop_assume!(nonconstant(&xs));
                prop_assume!(nonconstant(&ys));

                let model = SimpleLinearRegression::fit(&xs, &ys).unwrap();

                let correlation = pearson(&xs, &ys);
                let expected = correlation * correlation;
                let actual = model.r_squared();

                prop_assert!(approx_eq(actual, expected, Tolerance::STRICT));
                prop_assert!((0.0..=1.0).contains(&actual));
            }
        }
    }

    mod predict {
        use super::*;

        #[test]
        fn evaluates_the_fitted_line() {
            let xs = [-2.0, 0.0, 1.0, 3.0];
            let intercept = 3.0;
            let query = 4.0;

            for slope in [-2.0_f64, 0.0, 2.0] {
                let ys = xs.map(|x| slope.mul_add(x, intercept));

                let model = SimpleLinearRegression::fit(&xs, &ys).unwrap();

                let actual = model.predict(query);
                let expected = slope.mul_add(query, intercept);

                assert_approx_eq(actual, expected, Tolerance::VERY_STRICT);
            }
        }

        #[test]
        fn returns_nan_for_nonfinite_queries() {
            let xs = [1.0, 2.0, 3.0];
            let ys = [3.0, 5.0, 7.0];

            let model = SimpleLinearRegression::fit(&xs, &ys).unwrap();
            let constant_model = SimpleLinearRegression::fit(&xs, &[3.0; 3]).unwrap();

            for query in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let prediction = model.predict(query);
                let constant_prediction = constant_model.predict(query);

                assert!(prediction.is_nan());
                assert!(constant_prediction.is_nan());
            }
        }

        #[test]
        fn overflows_to_signed_infinity() {
            let xs = [1.0, 2.0, 3.0];
            let ys = [3.0, 5.0, 7.0];

            let model = SimpleLinearRegression::fit(&xs, &ys).unwrap();

            assert_eq!(model.predict(f64::MAX), f64::INFINITY);
            assert_eq!(model.predict(-f64::MAX), f64::NEG_INFINITY);
        }

        #[test]
        fn centered_prediction_retains_small_changes_around_large_offsets() {
            let xs = [1e16, 1e16, 1e16 + 2.0];
            let ys = [1.0, 1.0, 4.0];

            let model = SimpleLinearRegression::fit(&xs, &ys).unwrap();

            let predictions = xs.map(|x| model.predict(x));
            let residuals = model.residuals(&xs, &ys).unwrap();
            let coefficient_prediction = model.slope().mul_add(xs[0], model.intercept());

            assert_approx_eq(model.slope(), 1.5, Tolerance::VERY_STRICT);
            assert_all_approx_eq(&predictions, &ys, Tolerance::VERY_STRICT);
            assert_all_approx_eq(&residuals, &[0.0; 3], Tolerance::VERY_STRICT);
            assert!((coefficient_prediction - ys[0]).abs() > 0.5);
        }

        #[test]
        fn fits_and_extrapolates_from_subnormal_inputs() {
            let tiny = f64::from_bits(1);
            let xs = [-tiny, 0.0, tiny];

            let model = SimpleLinearRegression::fit(&xs, &xs).unwrap();

            let tiny_prediction = model.predict(tiny);
            let extrapolated_prediction = model.predict(f64::MAX);
            let residuals = model.residuals(&[f64::MAX], &[f64::MAX]).unwrap();

            assert_eq!(model.slope(), 1.0);
            assert_eq!(model.intercept(), 0.0);
            assert_eq!(tiny_prediction, tiny);
            assert_eq!(extrapolated_prediction, f64::MAX);
            assert_eq!(residuals, vec![0.0]);
        }

        #[test]
        fn retains_scaled_prediction_when_the_original_slope_underflows() {
            let large = 2.0_f64.powi(1000);
            let small = 2.0_f64.powi(-1000);

            let xs = [-large, 0.0, large];
            let ys = [-small, 0.0, small];

            let model = SimpleLinearRegression::fit(&xs, &ys).unwrap();

            let prediction = model.predict(large);
            let residuals = model.residuals(&xs, &ys).unwrap();

            assert_eq!(model.slope(), 0.0);
            assert_eq!(prediction, small);
            assert_eq!(residuals, vec![0.0; 3]);
        }
    }

    mod residuals {
        use unrvl_numerics::convert::usize_to_f64;
        use unrvl_numerics::summation::NeumaierSumExt;

        use super::*;
        use crate::moments::mean;

        #[test]
        fn are_zero_for_exact_lines() {
            let xs = [-2.0, 0.0, 1.0, 3.0];
            let intercept = 3.0;

            for slope in [-2.0_f64, 0.0, 2.0] {
                let ys = xs.map(|x| slope.mul_add(x, intercept));

                let model = SimpleLinearRegression::fit(&xs, &ys).unwrap();

                let residuals = model.residuals(&xs, &ys).unwrap();
                let expected = [0.0; 4];

                assert_all_approx_eq(&residuals, &expected, Tolerance::VERY_STRICT);
            }
        }

        #[test]
        fn evaluates_new_pairs_in_input_order_without_refitting() {
            let xs = [1.0, 2.0, 3.0];
            let ys = [3.0, 5.0, 7.0];

            let model = SimpleLinearRegression::fit(&xs, &ys).unwrap();

            let new_xs = [0.0, 4.0];
            let new_ys = [2.0, 8.0];

            let residuals = model.residuals(&new_xs, &new_ys).unwrap();
            let expected = [1.0, -1.0];

            assert_all_approx_eq(&residuals, &expected, Tolerance::VERY_STRICT);
            assert_eq!(model.observation_count(), xs.len());
            assert_eq!(model.slope(), 2.0);
            assert_eq!(model.intercept(), 1.0);
            assert_approx_eq(model.r_squared(), 1.0, Tolerance::VERY_STRICT);
        }

        #[test]
        fn accepts_a_single_pair() {
            let xs = [1.0, 2.0, 3.0];
            let ys = [3.0, 5.0, 7.0];

            let model = SimpleLinearRegression::fit(&xs, &ys).unwrap();

            let residuals = model.residuals(&[4.0], &[10.0]).unwrap();

            assert_eq!(residuals, vec![1.0]);
        }

        #[test]
        fn returns_empty_for_empty_pairs() {
            let xs = [1.0, 2.0, 3.0];
            let ys = [3.0, 5.0, 7.0];

            let model = SimpleLinearRegression::fit(&xs, &ys).unwrap();

            let residuals = model.residuals(&[], &[]).unwrap();

            assert_eq!(residuals, [] as [f64; 0]);
        }

        #[test]
        fn checks_lengths_before_values() {
            let xs = [1.0, 2.0, 3.0];
            let ys = [3.0, 5.0, 7.0];

            let model = SimpleLinearRegression::fit(&xs, &ys).unwrap();

            let error = model.residuals(&[], &[f64::NAN]).unwrap_err();

            assert_eq!(error, SimpleLinearError::LengthMismatch { x: 0, y: 1 });
        }

        #[test]
        fn rejects_nonfinite_predictors() {
            let xs = [1.0, 2.0, 3.0];
            let ys = [3.0, 5.0, 7.0];

            let model = SimpleLinearRegression::fit(&xs, &ys).unwrap();

            for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let query_xs = [4.0, value];
                let query_ys = [10.0, 0.0];

                let error = model.residuals(&query_xs, &query_ys).unwrap_err();

                let SimpleLinearError::NonFinitePredictor { index, value: actual } = error else {
                    panic!("expected predictor error");
                };

                assert_eq!(index, 1);
                assert_eq!(actual.to_bits(), value.to_bits());
            }
        }

        #[test]
        fn rejects_nonfinite_responses() {
            let xs = [1.0, 2.0, 3.0];
            let ys = [3.0, 5.0, 7.0];

            let model = SimpleLinearRegression::fit(&xs, &ys).unwrap();

            for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let query_xs = [4.0, 0.0];
                let query_ys = [10.0, value];

                let error = model.residuals(&query_xs, &query_ys).unwrap_err();

                let SimpleLinearError::NonFiniteResponse { index, value: actual } = error else {
                    panic!("expected response error");
                };

                assert_eq!(index, 1);
                assert_eq!(actual.to_bits(), value.to_bits());
            }
        }

        #[test]
        fn centered_residuals_preserve_changes_below_the_response_offset_ulp() {
            let base = 1e16;
            let xs = [1.0, 2.0, 3.0];
            let ys = [base, base, base + 2.0];

            let model = SimpleLinearRegression::fit(&xs, &ys).unwrap();

            // Rounding a prediction at this scale can hide a small residual.
            // Evaluate residuals before restoring response units.
            let residuals = model.residuals(&xs, &ys).unwrap();
            let expected = [1.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0];

            let rounded_residual = ys[0] - model.predict(xs[0]);

            assert_all_approx_eq(&residuals, &expected, Tolerance::VERY_STRICT);
            assert_approx_eq(model.slope(), 1.0, Tolerance::VERY_STRICT);
            assert_approx_eq(model.r_squared(), 0.75, Tolerance::VERY_STRICT);
            assert!((rounded_residual - expected[0]).abs() > 0.25);
        }

        #[test]
        fn extrapolation_avoids_subtracting_an_overflowed_prediction() {
            let tiny = f64::from_bits(1);
            let xs = [0.0, tiny];
            let ys = [0.0, 2.0 * tiny];

            let model = SimpleLinearRegression::fit(&xs, &ys).unwrap();

            let query_xs = [f64::MAX, -f64::MAX];
            let query_ys = [f64::MAX, -f64::MAX];

            let residuals = model.residuals(&query_xs, &query_ys).unwrap();
            let expected = [-f64::MAX, f64::MAX];

            assert_eq!(model.predict(f64::MAX), f64::INFINITY);
            assert_eq!(residuals, expected);
        }

        #[test]
        fn overflow_preserves_its_sign() {
            let tiny = f64::from_bits(1);
            let xs = [0.0, tiny];
            let ys = [0.0, 2.0 * tiny];

            let model = SimpleLinearRegression::fit(&xs, &ys).unwrap();

            let query_xs = [-f64::MAX, f64::MAX];
            let query_ys = [f64::MAX, -f64::MAX];

            let residuals = model.residuals(&query_xs, &query_ys).unwrap();
            let expected = [f64::INFINITY, f64::NEG_INFINITY];

            assert_eq!(residuals, expected);
        }

        #[test]
        fn constant_response_residuals_can_overflow() {
            let tiny = f64::from_bits(1);
            let xs = [0.0, tiny];

            for value in [f64::MAX, -f64::MAX] {
                let ys = [value; 2];

                let model = SimpleLinearRegression::fit(&xs, &ys).unwrap();

                let residual = model.residuals(&[f64::MAX], &[-value]).unwrap()[0];
                let expected = f64::INFINITY.copysign(-value);

                assert_eq!(residual, expected);
            }
        }

        proptest! {
            #[test]
            fn fit_residuals_satisfy_normal_equations((xs, ys) in finite_paired_samples(3)) {
                prop_assume!(nonconstant(&xs));

                let model = SimpleLinearRegression::fit(&xs, &ys).unwrap();

                let residuals = model.residuals(&xs, &ys).unwrap();

                let n = usize_to_f64(xs.len());
                let x_mean = mean(&xs);

                let x_scale = xs.iter().map(|x| x.abs()).fold(1.0, f64::max);
                let y_scale = ys.iter().map(|y| y.abs()).fold(1.0, f64::max);

                // The normal equations give sum(r) = 0 and sum((x - mean_x) * r) = 0.
                let residual_sum = residuals.iter().neumaier_sum();
                let centered_x_residual_sum = xs.iter()
                    .zip(&residuals)
                    .map(|(&x, &residual)| (x - x_mean) * residual)
                    .neumaier_sum();

                // Scale the rounding allowance with sample count and input magnitudes.
                // The second sum also includes a centered predictor factor.
                let rounding_tolerance = 2e-13;
                let residual_sum_bound = rounding_tolerance * n * y_scale;
                let centered_x_residual_sum_bound = rounding_tolerance * n * x_scale * y_scale;

                prop_assert!(residual_sum.abs() <= residual_sum_bound);
                prop_assert!(centered_x_residual_sum.abs() <= centered_x_residual_sum_bound);
            }
        }
    }
}
