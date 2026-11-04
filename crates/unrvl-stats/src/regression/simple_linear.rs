//! Ordinary least squares with one predictor and an intercept.

use thiserror::Error;
use unrvl_numerics::convert::usize_to_f64;
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
    /// [`SimpleLinearError::NumericalFailure`] for non-finite coefficients or
    /// unusable intermediate results.
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

    /// Returns adjusted R² for the observations used to fit this model.
    ///
    /// For one predictor and an intercept, the adjustment is
    /// `1 - (1 - R²) * (n - 1) / (n - 2)`, where `n` is the fitted observation
    /// count. Values may be negative and are not clamped to `[0, 1]`.
    ///
    /// Returns `NaN` for at most two observations or undefined
    /// [`R²`](Self::r_squared), including a constant response.
    ///
    /// Uses the stored R² and observation count. Evaluates the equivalent
    /// `((n - 1) * R² - 1) / (n - 2)` with a fused multiply-add in the
    /// numerator to reduce intermediate rounding near zero. Rounding in the
    /// stored R² still carries through; a perfect fit can round slightly below
    /// one.
    ///
    /// Uses `O(1)` time and additional memory.
    ///
    /// # Examples
    ///
    /// ```
    /// # use unrvl_stats::regression::simple_linear::SimpleLinearRegression;
    /// let xs = [-1.0, 0.0, 1.0];
    /// let ys = [1.0, -2.0, 1.0];
    /// let model = SimpleLinearRegression::fit(&xs, &ys).unwrap();
    /// assert_eq!(model.adjusted_r_squared(), -1.0);
    /// ```
    #[must_use]
    pub fn adjusted_r_squared(&self) -> f64 {
        let n = self.observation_count;
        if n <= 2 {
            return f64::NAN;
        }

        let n_minus_1 = usize_to_f64(n - 1);
        let n_minus_2 = usize_to_f64(n - 2);

        n_minus_1.mul_add(self.r_squared, -1.0) / n_minus_2
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
mod tests;
