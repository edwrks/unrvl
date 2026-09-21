//! Extremum operations for slices of floating-point values.
//!
//! NaN values are ignored when a non-NaN value is present. Index-returning
//! operations choose the first occurrence when values tie.

/// Returns the maximum absolute value from the given slice.
///
/// Inherits [`f64::max`]'s NaN policy: NaN values are ignored when a numeric
/// value is present, while an all-NaN slice returns NaN.
///
/// # Panics
///
/// Panics if the input slice is empty.
#[must_use]
pub fn max_abs(xs: &[f64]) -> f64 {
    assert_ne!(xs, &[], "input must not be empty");

    let first = xs[0].abs();
    xs.iter().skip(1).fold(first, |acc, &x| acc.max(x.abs()))
}

/// Returns the minimum absolute value from the given slice.
///
/// Inherits [`f64::min`]'s NaN policy: NaN values are ignored when a numeric
/// value is present, while an all-NaN slice returns NaN.
///
/// # Panics
///
/// Panics if the input slice is empty.
#[must_use]
pub fn min_abs(xs: &[f64]) -> f64 {
    assert_ne!(xs, &[], "input must not be empty");

    let first = xs[0].abs();
    xs.iter().skip(1).fold(first, |acc, &x| acc.min(x.abs()))
}

/// Returns the index of the maximum value in the given slice.
///
/// NaN values are ignored when a numeric value is present. If all values are
/// NaN, returns the first index. Ties are resolved in favor of the first
/// occurrence.
///
/// # Panics
///
/// Panics if the input slice is empty.
#[must_use]
pub fn argmax(xs: &[f64]) -> usize {
    assert_ne!(xs, &[], "input must not be empty");

    let mut index = 0;
    let mut maximum = xs[0];

    for (i, &x) in xs.iter().enumerate().skip(1) {
        if (maximum.is_nan() && !x.is_nan()) || x > maximum {
            maximum = x;
            index = i;
        }
    }

    index
}

/// Returns the index of the minimum value in the given slice.
///
/// NaN values are ignored when a numeric value is present. If all values are
/// NaN, returns the first index. Ties are resolved in favor of the first
/// occurrence.
///
/// # Panics
///
/// Panics if the input slice is empty.
#[must_use]
pub fn argmin(xs: &[f64]) -> usize {
    assert_ne!(xs, &[], "input must not be empty");

    let mut index = 0;
    let mut minimum = xs[0];

    for (i, &x) in xs.iter().enumerate().skip(1) {
        if (minimum.is_nan() && !x.is_nan()) || x < minimum {
            minimum = x;
            index = i;
        }
    }

    index
}

/// Returns the index of the maximum absolute value in the given slice.
///
/// NaN values are ignored when a numeric value is present. If all values are
/// NaN, returns the first index. Ties are resolved in favor of the first
/// occurrence.
///
/// # Panics
///
/// Panics if the input slice is empty.
#[must_use]
pub fn argmax_abs(xs: &[f64]) -> usize {
    assert_ne!(xs, &[], "input must not be empty");

    let mut index = 0;
    let mut maximum = xs[0].abs();

    for (i, &x) in xs.iter().enumerate().skip(1) {
        let magnitude = x.abs();

        if (maximum.is_nan() && !magnitude.is_nan()) || magnitude > maximum {
            maximum = magnitude;
            index = i;
        }
    }

    index
}

/// Returns the index of the minimum absolute value in the given slice.
///
/// NaN values are ignored when a numeric value is present. If all values are
/// NaN, returns the first index. Ties are resolved in favor of the first
/// occurrence.
///
/// # Panics
///
/// Panics if the input slice is empty.
#[must_use]
pub fn argmin_abs(xs: &[f64]) -> usize {
    assert_ne!(xs, &[], "input must not be empty");

    let mut index = 0;
    let mut minimum = xs[0].abs();

    for (i, &x) in xs.iter().enumerate().skip(1) {
        let magnitude = x.abs();

        if (minimum.is_nan() && !magnitude.is_nan()) || magnitude < minimum {
            minimum = magnitude;
            index = i;
        }
    }

    index
}

/// Returns the minimum and maximum values in the given slice.
///
/// Inherits [`f64::min`] and [`f64::max`]'s NaN policy: NaN values are
/// ignored when a numeric value is present, while an all-NaN slice returns
/// `(NaN, NaN)`.
///
/// # Panics
///
/// Panics if the input slice is empty.
#[must_use]
pub fn min_max(xs: &[f64]) -> (f64, f64) {
    let (&first, rest) = xs.split_first().expect("input must not be empty");

    rest.iter().fold((first, first), |(lower, upper), &x| {
        (lower.min(x), upper.max(x))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    mod max_abs {
        use super::max_abs;

        #[test]
        #[should_panic(expected = "input must not be empty")]
        fn panics_on_empty_input() {
            let _ = max_abs(&[]);
        }

        #[test]
        fn ignores_nan_when_numeric_values_are_present() {
            let xs = &[f64::NAN, 1.0, -2.0, f64::NAN];
            assert_eq!(max_abs(xs), 2.0);
        }

        #[test]
        fn returns_nan_when_all_values_are_nan() {
            let xs = &[f64::NAN, f64::NAN];
            let result = max_abs(xs);
            assert!(result.is_nan());
        }

        #[test]
        fn handles_infinity() {
            let xs = &[1.0, f64::NEG_INFINITY, 2.0];
            assert_eq!(max_abs(xs), f64::INFINITY);
        }

        #[test]
        fn returns_max_abs() {
            let xs = &[1.0, 100.0, -101.5, 13.25];
            assert_eq!(max_abs(xs), 101.5);
        }
    }

    mod min_abs {
        use super::min_abs;

        #[test]
        #[should_panic(expected = "input must not be empty")]
        fn panics_on_empty_input() {
            let _ = min_abs(&[]);
        }

        #[test]
        fn ignores_nan_when_numeric_values_are_present() {
            let xs = &[f64::NAN, 1.0, -2.0, f64::NAN];
            assert_eq!(min_abs(xs), 1.0);
        }

        #[test]
        fn returns_nan_when_all_values_are_nan() {
            let xs = &[f64::NAN, f64::NAN];
            let result = min_abs(xs);
            assert!(result.is_nan());
        }

        #[test]
        fn handles_infinity() {
            let xs = &[1.0, f64::NEG_INFINITY, 2.0];
            assert_eq!(min_abs(xs), 1.0);
        }

        #[test]
        fn handles_negative_zero() {
            let xs = &[-0.0, 1.0];
            let result = min_abs(xs);

            assert_eq!(result, 0.0);
            assert!(result.is_sign_positive());
        }

        #[test]
        fn returns_min_abs() {
            let xs = &[1.0, 100.0, -101.5, 13.25];
            assert_eq!(min_abs(xs), 1.0);
        }
    }

    mod argmax {
        use super::argmax;

        #[test]
        #[should_panic(expected = "input must not be empty")]
        fn panics_on_empty_input() {
            let _ = argmax(&[]);
        }

        #[test]
        fn ignores_nan_when_numeric_values_are_present() {
            let xs = &[f64::NAN, 1.0, -2.0, f64::NAN];
            assert_eq!(argmax(xs), 1);
        }

        #[test]
        fn returns_first_index_when_all_values_are_nan() {
            let xs = &[f64::NAN, f64::NAN];
            assert_eq!(argmax(xs), 0);
        }

        #[test]
        fn handles_infinity() {
            let xs = &[1.0, f64::INFINITY, 2.0];
            assert_eq!(argmax(xs), 1);
        }

        #[test]
        fn returns_first_occurrence() {
            let xs = &[1.0, 2.0, 2.0];
            assert_eq!(argmax(xs), 1);
        }

        #[test]
        fn returns_arg_max() {
            let xs = &[1.0, 100.0, -101.5, 13.25];
            assert_eq!(argmax(xs), 1);
        }
    }

    mod argmin {
        use super::argmin;

        #[test]
        #[should_panic(expected = "input must not be empty")]
        fn panics_on_empty_input() {
            let _ = argmin(&[]);
        }

        #[test]
        fn ignores_nan_when_numeric_values_are_present() {
            let xs = &[f64::NAN, 1.0, -2.0, f64::NAN];
            assert_eq!(argmin(xs), 2);
        }

        #[test]
        fn returns_first_index_when_all_values_are_nan() {
            let xs = &[f64::NAN, f64::NAN];
            assert_eq!(argmin(xs), 0);
        }

        #[test]
        fn handles_infinity() {
            let xs = &[1.0, f64::NEG_INFINITY, 2.0];
            assert_eq!(argmin(xs), 1);
        }

        #[test]
        fn returns_first_occurrence() {
            let xs = &[1.0, -2.0, -2.0];
            assert_eq!(argmin(xs), 1);
        }

        #[test]
        fn returns_arg_min() {
            let xs = &[1.0, 100.0, -101.5, 13.25];
            assert_eq!(argmin(xs), 2);
        }
    }

    mod argmax_abs {
        use super::argmax_abs;

        #[test]
        #[should_panic(expected = "input must not be empty")]
        fn panics_on_empty_input() {
            let _ = argmax_abs(&[]);
        }

        #[test]
        fn ignores_nan_when_numeric_values_are_present() {
            let xs = &[f64::NAN, 1.0, -2.0, f64::NAN];
            assert_eq!(argmax_abs(xs), 2);
        }

        #[test]
        fn returns_first_index_when_all_values_are_nan() {
            let xs = &[f64::NAN, f64::NAN];
            assert_eq!(argmax_abs(xs), 0);
        }

        #[test]
        fn handles_infinity() {
            let xs = &[1.0, f64::NEG_INFINITY, 2.0];
            assert_eq!(argmax_abs(xs), 1);
        }

        #[test]
        fn returns_first_occurrence() {
            let xs = &[1.0, -2.0, -2.0];
            assert_eq!(argmax_abs(xs), 1);
        }

        #[test]
        fn returns_arg_max_abs() {
            let xs = &[1.0, 100.0, -101.5, 13.25];
            assert_eq!(argmax_abs(xs), 2);
        }
    }

    mod argmin_abs {
        use super::argmin_abs;

        #[test]
        #[should_panic(expected = "input must not be empty")]
        fn panics_on_empty_input() {
            let _ = argmin_abs(&[]);
        }

        #[test]
        fn ignores_nan_when_numeric_values_are_present() {
            let xs = &[f64::NAN, 1.0, -2.0, f64::NAN];
            assert_eq!(argmin_abs(xs), 1);
        }

        #[test]
        fn returns_first_index_when_all_values_are_nan() {
            let xs = &[f64::NAN, f64::NAN];
            assert_eq!(argmin_abs(xs), 0);
        }

        #[test]
        fn handles_infinity() {
            let xs = &[1.0, f64::NEG_INFINITY, 2.0];
            assert_eq!(argmin_abs(xs), 0);
        }

        #[test]
        fn returns_first_occurrence() {
            let xs = &[1.0, -0.5, -0.5];
            assert_eq!(argmin_abs(xs), 1);
        }

        #[test]
        fn returns_arg_min_abs() {
            let xs = &[1.0, 100.0, -101.5, 13.25];
            assert_eq!(argmin_abs(xs), 0);
        }
    }

    mod min_max {
        use super::min_max;

        #[test]
        #[should_panic(expected = "input must not be empty")]
        fn panics_on_empty_input() {
            let _ = min_max(&[]);
        }

        #[test]
        fn singleton_returns_value_as_both_bounds() {
            let xs = &[42.0];

            assert_eq!(min_max(xs), (42.0, 42.0));
        }

        #[test]
        fn finds_minimum_and_maximum() {
            let xs = &[3.0, -2.0, 7.0, 1.0];
            assert_eq!(min_max(xs), (-2.0, 7.0));
        }

        #[test]
        fn handles_repeated_extrema() {
            let xs = &[-2.0, 7.0, -2.0, 3.0, 7.0];
            assert_eq!(min_max(xs), (-2.0, 7.0));
        }

        #[test]
        fn handles_negative_infinity() {
            let xs = &[1.0, f64::NEG_INFINITY, 3.0];
            assert_eq!(min_max(xs), (f64::NEG_INFINITY, 3.0));
        }

        #[test]
        fn handles_positive_infinity() {
            let xs = &[1.0, f64::INFINITY, 3.0];
            assert_eq!(min_max(xs), (1.0, f64::INFINITY));
        }

        #[test]
        fn handles_both_infinities() {
            let xs = &[0.0, f64::INFINITY, f64::NEG_INFINITY];
            assert_eq!(min_max(xs), (f64::NEG_INFINITY, f64::INFINITY));
        }

        #[test]
        fn ignores_nan_when_numeric_values_are_present() {
            let xs = &[1.0, f64::NAN, 3.0];
            assert_eq!(min_max(xs), (1.0, 3.0));
        }

        #[test]
        fn recovers_when_first_value_is_nan() {
            let xs = &[f64::NAN, 1.0, 3.0];
            assert_eq!(min_max(xs), (1.0, 3.0));
        }

        #[test]
        fn all_nan_values_return_nan_bounds() {
            let (min, max) = min_max(&[f64::NAN, f64::NAN]);

            assert!(min.is_nan());
            assert!(max.is_nan());
        }
    }
}
