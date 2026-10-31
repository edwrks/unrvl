use unrvl_numerics::convert::usize_to_f64;
use unrvl_numerics::dot::compensated_dot;

use super::{MatrixError, prepared_series, validate_series};
use crate::Normalization;
use crate::association::covariance::restore_units;

/// Returns the covariance matrix of aligned variable series with the selected
/// normalization.
///
/// Each inner slice contains all observations for one variable. For `p` series
/// of `n` observations, returns a `p` by `p` matrix. Entry `[i][j]` is the
/// [`covariance`](crate::association::covariance) of `series[i]` and
/// `series[j]` using the selected normalization. Both output axes follow input
/// series order. Diagonal entries are variances with that normalization, and
/// both triangles contain exactly matching values.
///
/// The divisor is `n - ddof`: [`Normalization::Population`] uses `ddof = 0`,
/// [`Normalization::Sample`] uses `ddof = 1` (Bessel's correction), and
/// [`Normalization::Ddof`] uses the supplied nonnegative integer.
///
/// Series must have equal lengths and matching observation order; timestamp
/// alignment is the caller's responsibility. All observations must be finite.
/// No observations are discarded or values filled in. Constant series are
/// valid and have zero covariance with every series. With population
/// normalization, a single finite observation per series gives an all-zero
/// matrix. All series are validated before calculating entries.
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
/// Returns [`MatrixError`] for no series, an invalid divisor (`n <= ddof`),
/// unequal lengths, or non-finite observations. Checks for no series and the
/// first series's divisor first, then visits series in input order, checking
/// each series's length before its values.
///
/// # Panics
///
/// Panics if otherwise valid series contain more than `2^53` observations each.
///
/// # Examples
///
/// ```
/// use unrvl_stats::Normalization;
/// use unrvl_stats::association::{MatrixError, covariance_matrix};
///
/// // Three observations of two variables, x and y.
/// let x = [1.0, 2.0, 3.0];
/// let y = [2.0, 4.0, 6.0];
/// let matrix = covariance_matrix(&[x.as_slice(), y.as_slice()], Normalization::Sample)?;
///
/// // Row/column 0 identifies x; row/column 1 identifies y.
/// assert_eq!(matrix, vec![vec![1.0, 2.0], vec![2.0, 4.0]]);
/// # Ok::<(), MatrixError>(())
/// ```
pub fn covariance_matrix<S: AsRef<[f64]>>(
    series: &[S],
    normalization: Normalization,
) -> Result<Vec<Vec<f64>>, MatrixError> {
    let divisor = usize_to_f64(validate_series(series, normalization)?);
    let prepared = prepared_series(series);
    let dimension = prepared.len();
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
    use test_utils::approx::{Tolerance, approx_eq, assert_approx_eq};
    use test_utils::strategies::{MAX_SAMPLE_LEN, finite_series, nonfinite_value};

    use super::{MatrixError, covariance_matrix};
    use crate::Normalization;
    use crate::association::covariance;
    use crate::moments::var;

    const NORMALIZATIONS: [Normalization; 3] = [
        Normalization::Population,
        Normalization::Sample,
        Normalization::Ddof(2),
    ];

    #[test]
    fn reports_the_position_and_value_of_nonfinite_observations() {
        for normalization in NORMALIZATIONS {
            for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                for (series_index, observation) in [(0, 0), (1, 1), (2, 2)] {
                    let mut series = vec![vec![1.0, 2.0, 3.0], vec![f64::MAX; 3], vec![0.0; 3]];
                    series[series_index][observation] = invalid;

                    let error = covariance_matrix(&series, normalization).unwrap_err();

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
    }

    #[test]
    fn population_singletons_reject_nonfinite_observations() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for series_index in 0..2 {
                let mut series = [vec![f64::MAX], vec![0.0]];
                series[series_index][0] = invalid;

                let error = covariance_matrix(&series, Normalization::Population).unwrap_err();
                let MatrixError::NonFiniteObservation { series, observation, value } = error else {
                    panic!("expected a non-finite observation error, got {error:?}");
                };

                assert_eq!(series, series_index);
                assert_eq!(observation, 0);
                assert_eq!(value.to_bits(), invalid.to_bits());
            }
        }
    }

    #[test]
    fn rejects_fewer_than_two_observations_per_series() {
        assert_eq!(
            covariance_matrix(&[Vec::new()], Normalization::Sample),
            Err(MatrixError::InsufficientObservations { count: 0 })
        );

        assert_eq!(
            covariance_matrix(&[vec![1.0], vec![2.0]], Normalization::Sample),
            Err(MatrixError::InsufficientObservations { count: 1 })
        );
    }

    #[test]
    fn rejects_inputs_without_variables() {
        let normalizations = [
            Normalization::Population,
            Normalization::Sample,
            Normalization::Ddof(2),
            Normalization::Ddof(usize::MAX),
        ];

        for normalization in normalizations {
            assert_eq!(
                covariance_matrix::<Vec<f64>>(&[], normalization),
                Err(MatrixError::NoVariables)
            );
        }
    }

    #[test]
    fn checks_observation_boundaries_without_count_overflow() {
        for n in 0..=4 {
            let series = [vec![7.0; n], vec![-3.0; n]];
            let adjustments = [0, 1, 2, n, n + 1, usize::MAX];

            for ddof in adjustments {
                let result = covariance_matrix(&series, Normalization::Ddof(ddof));

                if n > ddof {
                    assert_eq!(result.unwrap(), vec![vec![0.0; 2]; 2]);
                } else {
                    assert_eq!(
                        result,
                        Err(MatrixError::InsufficientObservations { count: n })
                    );
                }
            }
        }
    }

    #[test]
    fn checks_the_first_series_divisor_before_lengths_and_values() {
        let cases = [
            (Normalization::Population, 0),
            (Normalization::Sample, 1),
            (Normalization::Ddof(2), 2),
            (Normalization::Ddof(usize::MAX), 3),
        ];

        for (normalization, n) in cases {
            let series = [vec![f64::NAN; n], vec![f64::INFINITY; n + 1]];

            assert_eq!(
                covariance_matrix(&series, normalization),
                Err(MatrixError::InsufficientObservations { count: n })
            );
        }
    }

    #[test]
    fn reports_shorter_and_longer_series() {
        for normalization in NORMALIZATIONS {
            for last_series in [Vec::<f64>::new(), vec![3.0], vec![3.0; 2], vec![3.0; 4]] {
                let expected = MatrixError::UnequalSeriesLength {
                    series: 2,
                    expected: 3,
                    actual: last_series.len(),
                };

                let series = [vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0], last_series];
                assert_eq!(covariance_matrix(&series, normalization), Err(expected));
            }
        }
    }

    #[test]
    fn validates_series_in_input_order_and_lengths_before_values() {
        for normalization in NORMALIZATIONS {
            let series = [
                vec![1.0, 2.0, 3.0],
                vec![f64::INFINITY],
                vec![3.0, 4.0, 5.0],
            ];
            let expected = MatrixError::UnequalSeriesLength { series: 1, expected: 3, actual: 1 };

            assert_eq!(covariance_matrix(&series, normalization), Err(expected));

            let series = [vec![1.0, f64::INFINITY, 3.0], vec![3.0]];
            let expected = MatrixError::NonFiniteObservation {
                series: 0,
                observation: 1,
                value: f64::INFINITY,
            };

            assert_eq!(covariance_matrix(&series, normalization), Err(expected));
        }
    }

    #[test]
    fn accepts_finite_population_singletons() {
        let tiny = f64::from_bits(1);
        let values = [0.0, -0.0, tiny, -tiny, f64::MAX, -f64::MAX];
        let series: Vec<_> = values.into_iter().map(|value| vec![value]).collect();
        let expected = vec![vec![0.0; values.len()]; values.len()];

        for normalization in [Normalization::Population, Normalization::Ddof(0)] {
            assert_eq!(covariance_matrix(&series, normalization).unwrap(), expected);
        }
    }

    #[test]
    fn calculates_each_normalization_for_a_known_matrix() {
        let series = [
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 4.0, 6.0, 8.0],
            vec![7.0; 4],
        ];

        // The first series has squared-deviation sum 5. The second is twice
        // the first, and the third is constant.
        let cases = [
            (
                Normalization::Population,
                [[1.25, 2.5, 0.0], [2.5, 5.0, 0.0], [0.0; 3]],
            ),
            (
                Normalization::Sample,
                [
                    [5.0 / 3.0, 10.0 / 3.0, 0.0],
                    [10.0 / 3.0, 20.0 / 3.0, 0.0],
                    [0.0; 3],
                ],
            ),
            (
                Normalization::Ddof(2),
                [[2.5, 5.0, 0.0], [5.0, 10.0, 0.0], [0.0; 3]],
            ),
            (
                Normalization::Ddof(3),
                [[5.0, 10.0, 0.0], [10.0, 20.0, 0.0], [0.0; 3]],
            ),
        ];

        for (normalization, expected) in cases {
            assert_eq!(covariance_matrix(&series, normalization).unwrap(), expected);
        }
    }

    #[test]
    fn returns_one_row_and_column_per_variable() {
        // Three observations, two variables. Their sample variances are 1 and
        // 4, and their covariance is 2. The result is 2x2, not 3x3.
        let series = [vec![1.0, 2.0, 3.0], vec![2.0, 4.0, 6.0]];
        let expected = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let actual = covariance_matrix(&series, Normalization::Sample).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn supports_a_single_variable() {
        let series = [vec![1.0, 2.0, 3.0]];
        let cases = [
            (Normalization::Population, 2.0 / 3.0),
            (Normalization::Sample, 1.0),
            (Normalization::Ddof(2), 2.0),
        ];

        for (normalization, expected) in cases {
            assert_eq!(
                covariance_matrix(&series, normalization).unwrap(),
                vec![vec![expected]]
            );
        }
    }

    #[test]
    fn supports_more_variables_than_observations() {
        // With two observations, each sample covariance is half the product
        // of the changes: [2, -2, 0, 2], including a constant third variable.
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

        assert_eq!(
            covariance_matrix(&series, Normalization::Sample).unwrap(),
            expected
        );
    }

    #[test]
    fn constant_series_have_zero_covariances_even_at_extreme_values() {
        let series = [vec![f64::MAX; 3], vec![1.0, 2.0, 3.0]];
        let cases = [
            (Normalization::Population, 2.0 / 3.0),
            (Normalization::Sample, 1.0),
            (Normalization::Ddof(2), 2.0),
        ];

        for (normalization, variance) in cases {
            let expected = vec![vec![0.0, 0.0], vec![0.0, variance]];

            assert_eq!(covariance_matrix(&series, normalization).unwrap(), expected);
        }
    }

    #[test]
    fn preserves_finite_covariance_when_variances_overflow_or_underflow() {
        // One variance overflows and the other rounds to zero. Their sample
        // covariance is large*small = 1, which must survive in both triangles.
        let large = 2.0_f64.powi(600);
        let small = 2.0_f64.powi(-600);
        let series = [vec![-large, 0.0, large], vec![-small, 0.0, small]];
        let cases = [
            (Normalization::Population, 2.0 / 3.0),
            (Normalization::Sample, 1.0),
            (Normalization::Ddof(2), 2.0),
        ];

        for (normalization, covariance) in cases {
            let expected = vec![vec![f64::INFINITY, covariance], vec![covariance, 0.0]];

            assert_eq!(covariance_matrix(&series, normalization).unwrap(), expected);
        }
    }

    #[test]
    fn population_covariance_is_finite_when_sample_covariance_overflows() {
        let series = [vec![-f64::MAX, f64::MAX], vec![-1.0, 1.0], vec![1.0, -1.0]];
        let population = [
            [f64::INFINITY, f64::MAX, -f64::MAX],
            [f64::MAX, 1.0, -1.0],
            [-f64::MAX, -1.0, 1.0],
        ];
        let sample = [
            [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY],
            [f64::INFINITY, 2.0, -2.0],
            [f64::NEG_INFINITY, -2.0, 2.0],
        ];
        let cases = [
            (Normalization::Population, population),
            (Normalization::Sample, sample),
        ];

        for (normalization, expected) in cases {
            assert_eq!(covariance_matrix(&series, normalization).unwrap(), expected);
        }
    }

    #[test]
    fn compensates_product_rounding_before_cancellation() {
        let delta = 2.0_f64.powi(-27);
        let series = [
            vec![1.0 + delta, -1.0, -(1.0 + delta), 1.0],
            vec![1.0 - delta, 1.0, -(1.0 - delta), -1.0],
        ];

        // Both means are zero. The cross-product sum is -2 * delta²;
        // rounding each product before summing would give zero.
        let cases = [
            (Normalization::Population, -2.0_f64.powi(-55)),
            (Normalization::Sample, -2.0_f64.powi(-53) / 3.0),
            (Normalization::Ddof(2), -2.0_f64.powi(-54)),
        ];

        for (normalization, expected) in cases {
            let matrix = covariance_matrix(&series, normalization).unwrap();

            assert_approx_eq(matrix[0][1], expected, Tolerance::new(0.0, 1e-15));
            assert_eq!(matrix[0][1].to_bits(), matrix[1][0].to_bits());
        }
    }

    #[test]
    fn preserves_subnormal_covariance_until_final_rounding() {
        let x = 2.0_f64.powi(-537);
        let y = 2.0_f64.powi(-538);
        let tiny = f64::from_bits(1);
        let series = [vec![-x, x], vec![-y, y]];

        // The cross-product sum is 2^-1074. Sample covariance is exactly tiny;
        // population covariance is half that and rounds to zero.
        let cases = [
            (Normalization::Population, [[tiny, 0.0], [0.0, 0.0]]),
            (Normalization::Sample, [[2.0 * tiny, tiny], [tiny, 0.0]]),
        ];

        for (normalization, expected) in cases {
            assert_eq!(covariance_matrix(&series, normalization).unwrap(), expected);
        }
    }

    #[test]
    fn supports_borrowed_slices_and_arrays() {
        let x = [1.0, 2.0, 3.0];
        let y = [2.0, 4.0, 6.0];
        let borrowed = [x.as_slice(), y.as_slice()];
        let arrays = [x, y];
        let cases = [
            (
                Normalization::Population,
                [[2.0 / 3.0, 4.0 / 3.0], [4.0 / 3.0, 8.0 / 3.0]],
            ),
            (Normalization::Sample, [[1.0, 2.0], [2.0, 4.0]]),
            (Normalization::Ddof(2), [[2.0, 4.0], [4.0, 8.0]]),
        ];

        for (normalization, expected) in cases {
            assert_eq!(
                covariance_matrix(&borrowed, normalization).unwrap(),
                expected
            );
            assert_eq!(covariance_matrix(&arrays, normalization).unwrap(), expected);
        }
    }

    proptest! {
        #[test]
        fn prepared_matrix_matches_scalar_calculations(
            series in finite_series(2),
            adjustment in 0..MAX_SAMPLE_LEN,
        ) {
            let normalizations = [
                Normalization::Population,
                Normalization::Sample,
                Normalization::Ddof(adjustment % series[0].len()),
            ];

            for normalization in normalizations {
                let matrix = covariance_matrix(&series, normalization).unwrap();

                for (i, xs) in series.iter().enumerate() {
                    for (j, ys) in series.iter().enumerate().skip(i) {
                        let actual = matrix[i][j].to_bits();
                        let expected = covariance(xs, ys, normalization).to_bits();

                        prop_assert_eq!(actual, expected);
                    }
                }
            }
        }

        #[test]
        fn is_symmetric_with_selected_variances_on_the_diagonal(
            series in finite_series(2),
            adjustment in 0..MAX_SAMPLE_LEN,
        ) {
            let normalizations = [
                Normalization::Population,
                Normalization::Sample,
                Normalization::Ddof(adjustment % series[0].len()),
            ];

            for normalization in normalizations {
                let matrix = covariance_matrix(&series, normalization).unwrap();
                let dimension = series.len();
                prop_assert_eq!(matrix.len(), dimension);

                for (i, row) in matrix.iter().enumerate() {
                    prop_assert_eq!(row.len(), dimension);
                    prop_assert!(approx_eq(row[i], var(&series[i], normalization), Tolerance::STRICT));

                    for (j, &value) in row.iter().enumerate() {
                        prop_assert_eq!(value.to_bits(), matrix[j][i].to_bits());
                    }
                }
            }
        }

        #[test]
        fn reordering_variables_reorders_both_result_axes(
            series in finite_series(2),
            adjustment in 0..MAX_SAMPLE_LEN,
        ) {
            let mut reordered = series.clone();
            reordered.reverse();
            let normalizations = [
                Normalization::Population,
                Normalization::Sample,
                Normalization::Ddof(adjustment % series[0].len()),
            ];

            for normalization in normalizations {
                let expected = covariance_matrix(&series, normalization).unwrap();
                let actual = covariance_matrix(&reordered, normalization).unwrap();
                let dimension = expected.len();

                for (i, row) in actual.iter().enumerate() {
                    for (j, &value) in row.iter().enumerate() {
                        let expected_value = expected[dimension - 1 - i][dimension - 1 - j];
                        prop_assert!(approx_eq(value, expected_value, Tolerance::STRICT));
                    }
                }
            }
        }

        #[test]
        fn presets_match_explicit_ddof(series in finite_series(2)) {
            let aliases = [
                (Normalization::Population, Normalization::Ddof(0)),
                (Normalization::Sample, Normalization::Ddof(1)),
            ];

            for (preset, explicit) in aliases {
                let a = covariance_matrix(&series, preset).unwrap();
                let b = covariance_matrix(&series, explicit).unwrap();

                prop_assert_eq!(a.len(), b.len());
                for (a_row, b_row) in a.iter().zip(&b) {
                    prop_assert_eq!(a_row.len(), b_row.len());
                    for (a, b) in a_row.iter().zip(b_row) {
                        prop_assert_eq!(a.to_bits(), b.to_bits());
                    }
                }
            }
        }

        #[test]
        fn nonfinite_observation_invalidates_the_matrix(
            mut series in finite_series(3),
            value in nonfinite_value(),
            series_position in any::<usize>(),
            observation_position in any::<usize>(),
        ) {
            let series_index = series_position % series.len();
            let observation = observation_position % series[0].len();
            series[series_index][observation] = value;

            for normalization in NORMALIZATIONS {
                let error = covariance_matrix(&series, normalization).unwrap_err();

                match error {
                    MatrixError::NonFiniteObservation {
                        series,
                        observation: actual_observation,
                        value: invalid,
                    } => {
                        let actual = (series, actual_observation, invalid.to_bits());
                        let expected = (series_index, observation, value.to_bits());

                        prop_assert_eq!(actual, expected);
                    }
                    other => {
                        prop_assert!(false, "expected a non-finite observation error, got {other:?}");
                    }
                }
            }
        }
    }
}
