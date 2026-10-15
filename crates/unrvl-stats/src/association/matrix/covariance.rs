use unrvl_numerics::convert::usize_to_f64;
use unrvl_numerics::dot::compensated_dot;

use super::{MatrixError, prepared_series};
use crate::association::covariance::restore_units;

/// Returns the sample covariance matrix of aligned variable series.
///
/// Each inner slice contains all observations for one variable. For `p` series
/// of `n` observations, returns a `p` by `p` matrix. Entry `[i][j]` is the
/// sample [`covariance`](crate::association::covariance) of `series[i]` and
/// `series[j]`, using denominator `n - 1`. Both output axes follow input series
/// order. Diagonal entries are sample variances, and both triangles contain
/// exactly matching values.
///
/// Series must have equal lengths and matching observation order; timestamp
/// alignment is the caller's responsibility. All observations must be finite.
/// No observations are discarded or values filled in. Constant series are
/// valid and have zero covariance with every series.
///
/// The numerical safeguards and rounding limitations of
/// [`covariance`](crate::association::covariance) apply to each entry; a result
/// may overflow to signed infinity or underflow to zero.
///
/// Prepares each series's scale and centered deviations once, then calculates
/// each distinct variable pair and mirrors its result. Uses `O(n · p²)` time
/// and `O(n · p + p²)` additional memory, including the returned matrix.
///
/// # Errors
///
/// Returns [`MatrixError`] for no series, fewer than two observations per
/// series, unequal lengths, or non-finite observations. Checks for no series
/// and the first series's length first, then visits series in input order,
/// checking each series's length before its values.
///
/// # Panics
///
/// Panics if otherwise valid series contain more than `2^53` observations each.
///
/// # Examples
///
/// ```
/// # use unrvl_stats::association::{covariance_matrix, MatrixError};
/// // Three observations of two variables, x and y.
/// let x = [1.0, 2.0, 3.0];
/// let y = [2.0, 4.0, 6.0];
/// let matrix = covariance_matrix(&[x.as_slice(), y.as_slice()])?;
///
/// // Row/column 0 identifies x; row/column 1 identifies y.
/// assert_eq!(matrix, vec![vec![1.0, 2.0], vec![2.0, 4.0]]);
/// # Ok::<(), MatrixError>(())
/// ```
pub fn covariance_matrix<S: AsRef<[f64]>>(series: &[S]) -> Result<Vec<Vec<f64>>, MatrixError> {
    let prepared = prepared_series(series)?;
    let dimension = prepared.len();
    let divisor = usize_to_f64(series[0].as_ref().len() - 1);
    let mut matrix = vec![vec![0.0; dimension]; dimension];

    for i in 0..dimension {
        for j in i..dimension {
            let x = &prepared[i];
            let y = &prepared[j];
            let product_sum = compensated_dot(&x.deviations, &y.deviations);
            let value = restore_units(product_sum / divisor, x.unit, y.unit);
            matrix[i][j] = value;
            matrix[j][i] = value;
        }
    }

    Ok(matrix)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_utils::approx::{Tolerance, approx_eq};
    use test_utils::strategies::finite_series;

    use super::{MatrixError, covariance_matrix};
    use crate::association::covariance;
    use crate::moments::var;

    #[test]
    fn rejects_fewer_than_two_observations_per_series() {
        assert_eq!(
            covariance_matrix(&[Vec::new()]),
            Err(MatrixError::InsufficientObservations { count: 0 })
        );

        assert_eq!(
            covariance_matrix(&[vec![1.0], vec![2.0]]),
            Err(MatrixError::InsufficientObservations { count: 1 })
        );
    }

    #[test]
    fn rejects_inputs_without_variables() {
        assert_eq!(
            covariance_matrix::<Vec<f64>>(&[]),
            Err(MatrixError::NoVariables)
        );
    }

    #[test]
    fn reports_shorter_and_longer_series() {
        for last_series in [Vec::<f64>::new(), vec![3.0], vec![3.0, 4.0, 5.0]] {
            let expected = MatrixError::UnequalSeriesLength {
                series: 2,
                expected: 2,
                actual: last_series.len(),
            };

            let series = [vec![1.0, 2.0], vec![2.0, 3.0], last_series];
            assert_eq!(covariance_matrix(&series), Err(expected));
        }
    }

    #[test]
    fn reports_the_position_and_value_of_nonfinite_observations() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for (series_index, observation) in [(0, 0), (1, 1), (0, 2)] {
                let mut series = vec![vec![1.0, 2.0, 3.0]; 2];
                series[series_index][observation] = invalid;

                let error = covariance_matrix(&series).unwrap_err();

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

        let actual = covariance_matrix(&series);
        let expected = MatrixError::UnequalSeriesLength { series: 1, expected: 2, actual: 1 };

        assert_eq!(actual, Err(expected));

        let series = [vec![1.0, f64::INFINITY], vec![3.0]];
        let actual = covariance_matrix(&series);
        let expected =
            MatrixError::NonFiniteObservation { series: 0, observation: 1, value: f64::INFINITY };

        assert_eq!(actual, Err(expected));
    }

    #[test]
    fn returns_one_row_and_column_per_variable() {
        // Three observations, two variables. Their sample variances are 1 and
        // 4, and their covariance is 2. The result is 2x2, not 3x3.
        let series = [vec![1.0, 2.0, 3.0], vec![2.0, 4.0, 6.0]];
        let expected = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let actual = covariance_matrix(&series).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn supports_a_single_variable() {
        let series = [vec![1.0, 2.0, 3.0]];
        let actual = covariance_matrix(&series).unwrap();
        let expected = vec![vec![1.0]];

        assert_eq!(actual, expected);
    }

    #[test]
    fn supports_more_variables_than_observations() {
        // With two observations, each covariance is half the product of the
        // changes. Changes are [2, -2, 0, 2], including a constant third
        // variable.
        let series = [
            vec![1.0, 3.0],
            vec![3.0, 1.0],
            vec![5.0, 5.0],
            vec![7.0, 9.0],
        ];

        let expected = vec![
            vec![2.0, -2.0, 0.0, 2.0],
            vec![-2.0, 2.0, 0.0, -2.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![2.0, -2.0, 0.0, 2.0],
        ];

        let actual = covariance_matrix(&series).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn constant_series_have_zero_covariances_even_at_extreme_values() {
        let series = [vec![f64::MAX; 3], vec![1.0, 2.0, 3.0]];
        let expected = vec![vec![0.0, 0.0], vec![0.0, 1.0]];
        let actual = covariance_matrix(&series).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn preserves_finite_covariance_when_variances_overflow_or_underflow() {
        // Each series has pattern [-scale, 0, scale], so its variance is
        // scale². One variance overflows and the other rounds to zero. Their
        // covariance is large*small = 1, which must survive in both
        // off-diagonal entries.
        let large = 2.0_f64.powi(600);
        let small = 2.0_f64.powi(-600);
        let series = [vec![-large, 0.0, large], vec![-small, 0.0, small]];
        let expected = vec![vec![f64::INFINITY, 1.0], vec![1.0, 0.0]];
        let actual = covariance_matrix(&series).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn supports_borrowed_slices_and_arrays() {
        let x = [1.0, 2.0, 3.0];
        let y = [2.0, 4.0, 6.0];

        let borrowed = [x.as_slice(), y.as_slice()];
        let arrays = [x, y];

        let expected = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let a = covariance_matrix(&borrowed).unwrap();
        let b = covariance_matrix(&arrays).unwrap();

        assert_eq!(a, expected);
        assert_eq!(b, expected);
    }

    proptest! {
        #[test]
        fn prepared_matrix_matches_scalar_calculations(series in finite_series(2)) {
            let matrix = covariance_matrix(&series).unwrap();

            for (i, xs) in series.iter().enumerate() {
                for (j, ys) in series.iter().enumerate().skip(i) {
                    let actual = matrix[i][j].to_bits();
                    let expected = covariance(xs, ys).to_bits();

                    prop_assert_eq!(actual, expected);
                }
            }
        }

        #[test]
        fn is_symmetric_with_sample_variances_on_the_diagonal(series in finite_series(2)) {
            let matrix = covariance_matrix(&series).unwrap();
            let dimension = series.len();
            prop_assert_eq!(matrix.len(), dimension);

            for (i, row) in matrix.iter().enumerate() {
                prop_assert_eq!(row.len(), dimension);
                prop_assert!(approx_eq(row[i], var(&series[i]), Tolerance::STRICT));

                for (j, &value) in row.iter().enumerate() {
                    prop_assert_eq!(value.to_bits(), matrix[j][i].to_bits());
                }
            }
        }

        #[test]
        fn reordering_variables_reorders_both_result_axes(mut series in finite_series(2)) {
            let expected = covariance_matrix(&series).unwrap();
            series.reverse();

            let actual = covariance_matrix(&series).unwrap();
            let dimension = expected.len();

            for (i, row) in actual.iter().enumerate() {
                for (j, &value) in row.iter().enumerate() {
                    let expected_value = expected[dimension - 1 - i][dimension - 1 - j];
                    prop_assert!(approx_eq(value, expected_value, Tolerance::STRICT));
                }
            }
        }
    }
}
