use unrvl_numerics::dot::compensated_dot;

use super::{MatrixError, prepared_series};

/// Returns the Pearson correlation matrix of aligned variable series.
///
/// Each inner slice contains all observations for one variable. For `p` series
/// of `n` observations, returns a `p` by `p` matrix. Off-diagonal entry
/// `[i][j]` is [`pearson`](crate::association::pearson) applied to `series[i]`
/// and `series[j]`. Both output axes follow input series order. Nonconstant
/// series have diagonal entries of exactly one. Both triangles contain exactly
/// matching values.
///
/// Series must have equal lengths and matching observation order; timestamp
/// alignment is the caller's responsibility. All observations must be finite.
/// Constant series are valid inputs, but their correlations are undefined:
/// every entry in their row and column is `NaN`, including the diagonal.
/// All other entries lie in `[-1, 1]`. No observations are discarded or filled
/// in.
///
/// Calculates correlations directly with the numerical safeguards of
/// [`pearson`](crate::association::pearson), preserving results when covariance
/// would overflow or underflow. Prepares each series's centered deviations and
/// their norm once. Uses `O(n · p²)` time and `O(n · p + p²)` additional
/// memory, including the returned matrix.
///
/// # Errors
///
/// Returns [`MatrixError`] for no series, fewer than two observations per
/// series, unequal lengths, or non-finite observations. Validation follows the
/// same order as [`covariance_matrix`](super::covariance_matrix).
///
/// # Panics
///
/// Panics if otherwise valid series contain more than `2^53` observations each.
///
/// # Examples
///
/// ```
/// # use unrvl_stats::association::{correlation_matrix, MatrixError};
/// let series = vec![vec![1.0, 2.0, 3.0], vec![1.0, 3.0, 2.0]];
/// let matrix = correlation_matrix(&series)?;
///
/// // Entry [0][1] relates the first series to the second series.
/// assert_eq!(matrix[0][0], 1.0);
/// assert_eq!(matrix[1][1], 1.0);
/// assert!((matrix[0][1] - 0.5).abs() < 1e-14);
/// assert_eq!(matrix[0][1], matrix[1][0]);
/// # Ok::<(), MatrixError>(())
/// ```
pub fn correlation_matrix<S: AsRef<[f64]>>(series: &[S]) -> Result<Vec<Vec<f64>>, MatrixError> {
    let prepared = prepared_series(series)?;
    let dimension = prepared.len();

    // Pearson's denominator is the product of the Euclidean norms of the
    // centered deviation vectors.
    //
    // Each norm is sqrt(sum of squared deviations). Precompute it once per
    // series so it can be reused across variable pairs. The original positive
    // power-of-two scales cancel.
    let norms: Vec<_> = prepared
        .iter()
        .map(|series| compensated_dot(&series.deviations, &series.deviations).sqrt())
        .collect();

    let mut matrix = vec![vec![0.0; dimension]; dimension];

    for i in 0..dimension {
        for j in i..dimension {
            let value = if norms[i] == 0.0 || norms[j] == 0.0 {
                f64::NAN
            } else if i == j {
                1.0
            } else {
                let product_sum = compensated_dot(&prepared[i].deviations, &prepared[j].deviations);
                (product_sum / (norms[i] * norms[j])).clamp(-1.0, 1.0)
            };

            matrix[i][j] = value;
            matrix[j][i] = value;
        }
    }

    Ok(matrix)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_utils::approx::{Tolerance, approx_eq, assert_all_approx_eq, assert_approx_eq};
    use test_utils::strategies::finite_series;

    use super::{MatrixError, correlation_matrix};
    use crate::association::pearson;
    use crate::moments::var;

    #[test]
    fn rejects_fewer_than_two_observations_per_series() {
        assert_eq!(
            correlation_matrix(&[Vec::new()]),
            Err(MatrixError::InsufficientObservations { count: 0 })
        );

        assert_eq!(
            correlation_matrix(&[vec![1.0], vec![2.0]]),
            Err(MatrixError::InsufficientObservations { count: 1 })
        );
    }

    #[test]
    fn rejects_inputs_without_variables() {
        assert_eq!(
            correlation_matrix::<Vec<f64>>(&[]),
            Err(MatrixError::NoVariables)
        );
    }

    #[test]
    fn reports_shorter_and_longer_series() {
        for last_series in [Vec::new(), vec![3.0], vec![3.0, 4.0, 5.0]] {
            let expected = MatrixError::UnequalSeriesLength {
                series: 2,
                expected: 2,
                actual: last_series.len(),
            };

            let series = [vec![1.0, 2.0], vec![2.0, 3.0], last_series];
            assert_eq!(correlation_matrix(&series), Err(expected));
        }
    }

    #[test]
    fn reports_the_position_and_value_of_nonfinite_observations() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for (series_index, observation) in [(0, 0), (1, 1), (0, 2)] {
                let mut series = vec![vec![1.0, 2.0, 3.0]; 2];
                series[series_index][observation] = invalid;

                let error = correlation_matrix(&series).unwrap_err();

                let MatrixError::NonFiniteObservation {
                    series: actual_series,
                    observation: actual_observation,
                    value,
                } = error
                else {
                    panic!("expected a non-finite observation error, got {error:?}");
                };

                assert_eq!(actual_series, series_index);
                assert_eq!(actual_observation, observation);
                assert_eq!(value.to_bits(), invalid.to_bits());
            }
        }
    }

    #[test]
    fn validates_series_in_input_order_and_lengths_before_values() {
        let series = [vec![1.0, 2.0], vec![f64::INFINITY], vec![3.0, 4.0]];
        let actual = correlation_matrix(&series);
        let expected = MatrixError::UnequalSeriesLength { series: 1, expected: 2, actual: 1 };

        assert_eq!(actual, Err(expected));

        let series = [vec![1.0, f64::INFINITY], vec![3.0]];
        let actual = correlation_matrix(&series);
        let expected =
            MatrixError::NonFiniteObservation { series: 0, observation: 1, value: f64::INFINITY };

        assert_eq!(actual, Err(expected));
    }

    #[test]
    fn returns_one_row_and_column_per_variable() {
        // Three observations, two variables. Pearson correlation is 1/2.
        let series = [vec![1.0, 2.0, 3.0], vec![1.0, 3.0, 2.0]];
        let actual = correlation_matrix(&series).unwrap();

        assert_eq!(actual.len(), 2);
        assert_all_approx_eq(&actual[0], &[1.0, 0.5], Tolerance::VERY_STRICT);
        assert_all_approx_eq(&actual[1], &[0.5, 1.0], Tolerance::VERY_STRICT);
    }

    #[test]
    fn single_variable_correlation_is_one_unless_constant() {
        let varying = [vec![1.0, 2.0, 3.0]];
        let actual = correlation_matrix(&varying).unwrap();
        assert_eq!(actual, vec![vec![1.0]]);

        let constant = [vec![2.0; 3]];
        let actual = correlation_matrix(&constant).unwrap();
        assert_eq!(actual.len(), 1);
        assert_eq!(actual[0].len(), 1);
        assert!(actual[0][0].is_nan());
    }

    #[test]
    fn constant_series_have_nan_correlations_without_affecting_other_variables() {
        let series = [vec![f64::MAX; 3], vec![1.0, 2.0, 3.0], vec![6.0, 4.0, 2.0]];
        let actual = correlation_matrix(&series).unwrap();

        // Only the constant variable's row and column are undefined.
        for i in 0..3 {
            assert!(actual[0][i].is_nan());
            assert!(actual[i][0].is_nan());
        }

        assert_eq!(actual[1][1], 1.0);
        assert_eq!(actual[2][2], 1.0);
        assert_approx_eq(actual[1][2], -1.0, Tolerance::VERY_STRICT);
        assert_approx_eq(actual[2][1], -1.0, Tolerance::VERY_STRICT);
    }

    #[test]
    fn supports_more_variables_than_observations() {
        let series = [vec![1.0, 3.0], vec![3.0, 1.0], vec![5.0, 7.0]];
        let actual = correlation_matrix(&series).unwrap();
        let expected = [[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, 1.0]];

        assert_eq!(actual.len(), 3);

        for (actual, expected) in actual.iter().zip(expected) {
            assert_all_approx_eq(actual, &expected, Tolerance::VERY_STRICT);
        }
    }

    #[test]
    fn correlations_survive_when_covariances_overflow_or_underflow() {
        // The series move in opposite directions despite vastly different
        // units. Normalizing a covariance matrix would encounter infinity and
        // zero variances; using Pearson directly preserves -1.
        let smallest_positive = f64::from_bits(1);
        let series = [
            vec![-f64::MAX, 0.0, f64::MAX],
            vec![smallest_positive, 0.0, -smallest_positive],
        ];

        let actual = correlation_matrix(&series).unwrap();

        assert_eq!(actual[0][0], 1.0);
        assert_eq!(actual[1][1], 1.0);
        assert_approx_eq(actual[0][1], -1.0, Tolerance::VERY_STRICT);
        assert_approx_eq(actual[1][0], -1.0, Tolerance::VERY_STRICT);
    }

    #[test]
    fn supports_borrowed_slices_and_arrays() {
        let x = [1.0, 2.0, 3.0];
        let y = [2.0, 4.0, 6.0];

        let borrowed = [x.as_slice(), y.as_slice()];
        let arrays = [x, y];

        let expected = [[1.0, 1.0], [1.0, 1.0]];
        let a = correlation_matrix(&borrowed).unwrap();
        let b = correlation_matrix(&arrays).unwrap();

        for actual in [a, b] {
            for (actual, expected) in actual.iter().zip(expected) {
                assert_all_approx_eq(actual, &expected, Tolerance::VERY_STRICT);
            }
        }
    }

    proptest! {
        #[test]
        fn prepared_matrix_matches_scalar_calculations(series in finite_series(2)) {
            let matrix = correlation_matrix(&series).unwrap();

            for (i, xs) in series.iter().enumerate() {
                for (j, ys) in series.iter().enumerate().skip(i) {
                    let expected = pearson(xs, ys);

                    if expected.is_nan() {
                        prop_assert!(matrix[i][j].is_nan());
                    } else if i == j {
                        prop_assert_eq!(matrix[i][j].to_bits(), 1.0_f64.to_bits());
                    } else {
                        prop_assert_eq!(matrix[i][j].to_bits(), expected.to_bits());
                    }
                }
            }
        }

        #[test]
        fn is_symmetric_bounded_and_has_expected_diagonals(series in finite_series(2)) {
            let matrix = correlation_matrix(&series).unwrap();
            let dimension = series.len();
            prop_assert_eq!(matrix.len(), dimension);

            // Bounded generated inputs let variance identify constant series.
            let constant: Vec<_> = series.iter().map(|values| var(values) == 0.0).collect();

            for (i, row) in matrix.iter().enumerate() {
                prop_assert_eq!(row.len(), dimension);

                for (j, &value) in row.iter().enumerate() {
                    prop_assert_eq!(value.to_bits(), matrix[j][i].to_bits());

                    if constant[i] || constant[j] {
                        prop_assert!(value.is_nan());
                    } else {
                        prop_assert!((-1.0..=1.0).contains(&value));
                        if i == j {
                            prop_assert_eq!(value.to_bits(), 1.0_f64.to_bits());
                        }
                    }
                }
            }
        }

        #[test]
        fn reordering_variables_reorders_both_result_axes(mut series in finite_series(2)) {
            let expected = correlation_matrix(&series).unwrap();
            series.reverse();

            let actual = correlation_matrix(&series).unwrap();
            let dimension = expected.len();

            for (i, row) in actual.iter().enumerate() {
                for (j, &value) in row.iter().enumerate() {
                    let expected_value = expected[dimension - 1 - i][dimension - 1 - j];

                    if expected_value.is_nan() {
                        prop_assert!(value.is_nan());
                    } else {
                        prop_assert!(approx_eq(value, expected_value, Tolerance::STRICT));
                    }
                }
            }
        }
    }
}
