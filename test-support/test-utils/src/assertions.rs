//! Assertions for statistical input contracts.

/// Checks that `f` returns `NaN` for a non-finite value at every input
/// position.
///
/// Starts from a nonempty, finite control sample whose result must be finite.
/// Replaces each observation in turn with `NaN`, positive infinity, and
/// negative infinity. The caller selects a control sample appropriate for the
/// function and its estimator. Minimum sample sizes and other domain rules
/// remain separate tests.
///
/// # Panics
///
/// Panics if the control sample is empty, contains a non-finite value, or
/// produces a non-finite result. Also panics if any modified sample produces
/// a result other than `NaN`, reporting the position, input, and result.
#[track_caller]
pub fn assert_nan_for_nonfinite_inputs(f: impl Fn(&[f64]) -> f64, valid_sample: &[f64]) {
    assert!(!valid_sample.is_empty(), "control sample must be nonempty");
    assert!(
        valid_sample.iter().all(|value| value.is_finite()),
        "control sample must contain only finite values",
    );
    assert!(
        f(valid_sample).is_finite(),
        "control sample {valid_sample:?} must produce a finite result",
    );

    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for index in 0..valid_sample.len() {
            let mut sample = valid_sample.to_vec();
            sample[index] = invalid;

            let result = f(&sample);
            assert!(
                result.is_nan(),
                "index={index}, invalid={invalid:?}, sample={sample:?}, result={result:?}",
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_a_function_that_rejects_nonfinite_inputs() {
        let f = |xs: &[f64]| {
            if xs.iter().all(|value| value.is_finite()) {
                1.0
            } else {
                f64::NAN
            }
        };

        assert_nan_for_nonfinite_inputs(f, &[1.0, 2.0, 4.0]);
    }

    #[test]
    #[should_panic(expected = "control sample must be nonempty")]
    fn rejects_an_empty_control_sample() {
        assert_nan_for_nonfinite_inputs(|_| 0.0, &[]);
    }

    #[test]
    #[should_panic(expected = "control sample must contain only finite values")]
    fn rejects_a_nonfinite_control_sample() {
        assert_nan_for_nonfinite_inputs(|_| 0.0, &[f64::NAN]);
    }

    #[test]
    #[should_panic(expected = "must produce a finite result")]
    fn rejects_a_control_sample_with_an_undefined_result() {
        assert_nan_for_nonfinite_inputs(|_| f64::NAN, &[1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "invalid=inf")]
    fn detects_a_function_that_accepts_infinity() {
        assert_nan_for_nonfinite_inputs(
            |xs| {
                if xs.iter().any(|value| value.is_nan()) {
                    f64::NAN
                } else {
                    0.0
                }
            },
            &[1.0, 2.0],
        );
    }

    #[test]
    #[should_panic(expected = "index=1")]
    fn detects_a_function_that_only_checks_the_first_observation() {
        assert_nan_for_nonfinite_inputs(
            |xs| if xs[0].is_finite() { 0.0 } else { f64::NAN },
            &[1.0, 2.0],
        );
    }
}
