//! Empirical distribution functions for repeated sample queries.

use thiserror::Error;
use unrvl_numerics::convert::usize_to_f64;

/// An empirical distribution backed by a sorted copy of a finite sample.
///
/// Construction validates and sorts the observations once. CDF and survival
/// queries use binary search; inverse CDF queries select an observation without
/// interpolation. Queries do not allocate or modify the stored sample.
///
/// ```
/// use unrvl_stats::distribution::empirical::Ecdf;
///
/// let distribution = Ecdf::new(&[10.0, 0.0, 10.0, 20.0])?;
///
/// assert_eq!(distribution.cdf(10.0), 0.75);
/// assert_eq!(distribution.sf(10.0), 0.25);
/// assert_eq!(distribution.inverse_cdf(0.5), 10.0);
/// # Ok::<(), unrvl_stats::distribution::empirical::EcdfError>(())
/// ```
#[derive(Debug, Clone)]
pub struct Ecdf {
    sorted: Vec<f64>,
}

/// An invalid sample supplied to [`Ecdf::new`].
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum EcdfError {
    /// No observations were provided.
    #[error("sample must contain at least one observation")]
    EmptySample,

    /// An observation is NaN or infinite.
    #[error("observation at index {index} must be finite, got {value}")]
    NonFiniteObservation {
        /// Zero-based index in the original sample.
        index: usize,

        /// The non-finite observation.
        value: f64,
    },
}

impl Ecdf {
    /// Creates an empirical distribution from a nonempty, finite sample.
    ///
    /// Copies and sorts the observations.
    ///
    /// # Errors
    ///
    /// Returns [`EcdfError::EmptySample`] if `xs` is empty, or
    /// [`EcdfError::NonFiniteObservation`] for the first non-finite observation
    /// in the original sample.
    ///
    /// # Panics
    ///
    /// Panics if a finite sample contains more than `2^53` observations.
    pub fn new(xs: &[f64]) -> Result<Self, EcdfError> {
        if xs.is_empty() {
            return Err(EcdfError::EmptySample);
        }

        for (index, &value) in xs.iter().enumerate() {
            if !value.is_finite() {
                return Err(EcdfError::NonFiniteObservation { index, value });
            }
        }

        // Establish exact representability of all counts before copying.
        let _ = usize_to_f64(xs.len());

        let mut sorted = xs.to_vec();
        sorted.sort_unstable_by(f64::total_cmp);

        Ok(Self { sorted })
    }

    /// Returns the fraction of observations less than or equal to `query`.
    ///
    /// Includes all observations tied at the query. Negative infinity returns
    /// zero, positive infinity returns one, and NaN returns `NaN`.
    #[must_use]
    pub fn cdf(&self, query: f64) -> f64 {
        if query.is_nan() {
            return f64::NAN;
        }

        let count = self.sorted.partition_point(|&x| x <= query);

        usize_to_f64(count) / usize_to_f64(self.sorted.len())
    }

    /// Returns the fraction of observations strictly greater than `query`.
    ///
    /// Excludes observations tied at the query. Negative infinity returns one,
    /// positive infinity returns zero, and NaN returns `NaN`.
    ///
    /// Computes the exceedance count before division, avoiding subtraction
    /// from a rounded cumulative probability.
    #[must_use]
    pub fn sf(&self, query: f64) -> f64 {
        if query.is_nan() {
            return f64::NAN;
        }

        let count = self.sorted.partition_point(|&x| x <= query);
        let exceedances = self.sorted.len() - count;

        usize_to_f64(exceedances) / usize_to_f64(self.sorted.len())
    }

    /// Returns the inverse empirical CDF at `p` using Hyndman–Fan Type 1
    /// quantiles.
    ///
    /// For `0 < p <= 1`, selects the observation at one-based rank
    /// `ceil(n * p)`, where `n` is the sample length. Values are selected
    /// without interpolation.
    ///
    /// Probabilities zero and one return the minimum and maximum, respectively.
    /// A single observation is returned for every valid probability. Returns
    /// `NaN` if `p` is non-finite or outside `[0, 1]`.
    ///
    /// The rank is computed in floating-point arithmetic. Rounding near a
    /// cumulative jump can affect which observation is selected.
    #[must_use]
    pub fn inverse_cdf(&self, p: f64) -> f64 {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return f64::NAN;
        }

        if p == 0.0 {
            return self.sorted[0];
        }

        let n = usize_to_f64(self.sorted.len());
        let h = (n * p).ceil();

        #[expect(
            clippy::cast_sign_loss,
            reason = "position is positive for a valid nonzero probability"
        )]
        #[expect(
            clippy::cast_possible_truncation,
            reason = "position is an integer within the sample length"
        )]
        let rank = h as usize;

        self.sorted[rank - 1]
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_utils::strategies::{
        constant_sample, finite_sample, finite_value, positive_power_of_two_scale, probability,
    };

    use super::*;

    mod new {
        use super::*;

        #[test]
        fn rejects_empty_sample() {
            let xs = [];
            let error = Ecdf::new(&xs).unwrap_err();

            assert_eq!(error, EcdfError::EmptySample);
        }

        #[test]
        fn reports_first_nonfinite_observation_in_input_order() {
            for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let xs = [3.0, invalid, f64::NAN, -1.0];
                let error = Ecdf::new(&xs).unwrap_err();
                let EcdfError::NonFiniteObservation { index, value } = error else {
                    panic!("expected a non-finite observation error");
                };

                assert_eq!(index, 1);
                assert_eq!(value.to_bits(), invalid.to_bits());
            }
        }

        #[test]
        fn rejects_nonfinite_singletons() {
            for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let xs = [value];
                let result = Ecdf::new(&xs);

                assert!(matches!(
                    result,
                    Err(EcdfError::NonFiniteObservation { index: 0, .. })
                ));
            }
        }

        proptest! {
            #[test]
            fn rejects_nonfinite_observation_at_any_position(
                mut xs in finite_sample(1),
                invalid in prop::sample::select(vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY]),
                position in any::<usize>(),
            ) {
                let index = position % xs.len();
                xs[index] = invalid;
                let error = Ecdf::new(&xs).unwrap_err();

                let EcdfError::NonFiniteObservation { index: actual_index, value } = error else {
                    return Err(TestCaseError::fail("expected a non-finite observation error"));
                };

                prop_assert_eq!(actual_index, index);
                prop_assert_eq!(value.to_bits(), invalid.to_bits());
            }
        }
    }

    mod cdf {
        use super::*;

        #[test]
        fn includes_all_ties() {
            let xs = [2.0, -1.0, 2.0, 0.0, 2.0, 4.0];
            let distribution = Ecdf::new(&xs).unwrap();
            let query = 2.0;
            let expected = 5.0 / 6.0;

            assert_eq!(distribution.cdf(query), expected);
        }

        #[test]
        fn stays_constant_between_observations() {
            let xs = [10.0, 0.0];
            let distribution = Ecdf::new(&xs).unwrap();

            for query in [0.0, 2.5, 5.0, 7.5, 10.0_f64.next_down()] {
                assert_eq!(distribution.cdf(query), 0.5);
            }
        }

        #[test]
        fn handles_queries_below_at_and_above_sample_bounds() {
            let xs = [-2.0, 4.0];
            let distribution = Ecdf::new(&xs).unwrap();

            assert_eq!(distribution.cdf(-3.0), 0.0);
            assert_eq!(distribution.cdf(-2.0), 0.5);
            assert_eq!(distribution.cdf(4.0), 1.0);
            assert_eq!(distribution.cdf(5.0), 1.0);
        }

        #[test]
        fn infinite_queries_return_probability_endpoints() {
            let xs = [-f64::MAX, 0.0, f64::MAX];
            let distribution = Ecdf::new(&xs).unwrap();

            assert_eq!(distribution.cdf(f64::NEG_INFINITY), 0.0);
            assert_eq!(distribution.cdf(f64::INFINITY), 1.0);
        }

        #[test]
        fn nan_query_returns_nan() {
            let xs = [1.0];
            let distribution = Ecdf::new(&xs).unwrap();

            assert!(distribution.cdf(f64::NAN).is_nan());
        }

        #[test]
        fn treats_signed_zeros_as_ties() {
            let xs = [0.0, -1.0, -0.0, 1.0];
            let distribution = Ecdf::new(&xs).unwrap();

            for query in [-0.0, 0.0] {
                assert_eq!(distribution.cdf(query), 0.75);
            }
        }

        #[test]
        fn zero_probabilities_are_positive_zero() {
            let xs = [-0.0, 0.0];
            let distribution = Ecdf::new(&xs).unwrap();
            let expected = 0.0_f64;

            assert_eq!(distribution.cdf(-1.0).to_bits(), expected.to_bits());
        }

        #[test]
        fn distinguishes_adjacent_values_at_extreme_scales() {
            let tiny = f64::from_bits(1);
            let samples = [
                [f64::MAX.next_down(), f64::MAX],
                [-f64::MAX, (-f64::MAX).next_up()],
                [tiny, tiny.next_up()],
            ];

            for xs in samples {
                let distribution = Ecdf::new(&xs).unwrap();
                let query = xs[0];

                assert_eq!(distribution.cdf(query), 0.5);
            }
        }

        proptest! {
            #[test]
            fn matches_direct_counts(
                xs in finite_sample(1),
                query in finite_value(),
            ) {
                let distribution = Ecdf::new(&xs).unwrap();
                let count = xs.iter().filter(|&&x| x <= query).count();
                let expected = usize_to_f64(count) / usize_to_f64(xs.len());
                let result = distribution.cdf(query);

                prop_assert_eq!(result.to_bits(), expected.to_bits());
            }

            #[test]
            fn probability_is_bounded(
                xs in finite_sample(1),
                query in finite_value(),
            ) {
                let distribution = Ecdf::new(&xs).unwrap();
                let result = distribution.cdf(query);

                prop_assert!((0.0..=1.0).contains(&result));
            }

            #[test]
            fn is_nondecreasing(
                xs in finite_sample(1),
                q1 in finite_value(),
                q2 in finite_value(),
            ) {
                let distribution = Ecdf::new(&xs).unwrap();
                let lower = q1.min(q2);
                let upper = q1.max(q2);
                let lower_probability = distribution.cdf(lower);
                let upper_probability = distribution.cdf(upper);

                prop_assert!(lower_probability <= upper_probability);
            }

            #[test]
            fn is_invariant_to_input_order(
                xs in finite_sample(1),
                query in finite_value(),
            ) {
                let original = Ecdf::new(&xs).unwrap();
                let mut reversed = xs;
                reversed.reverse();
                let reordered = Ecdf::new(&reversed).unwrap();

                let expected = original.cdf(query);
                let result = reordered.cdf(query);

                prop_assert_eq!(result.to_bits(), expected.to_bits());
            }

            #[test]
            fn positive_power_of_two_scaling_preserves_probability(
                xs in finite_sample(1),
                query in finite_value(),
                scale in positive_power_of_two_scale(),
            ) {
                let original = Ecdf::new(&xs).unwrap();

                let scaled_sample: Vec<_> = xs.iter().map(|&x| x * scale).collect();
                let scaled = Ecdf::new(&scaled_sample).unwrap();

                let scaled_query = query * scale;
                let expected = original.cdf(query);
                let result = scaled.cdf(scaled_query);

                prop_assert_eq!(result.to_bits(), expected.to_bits());
            }

            #[test]
            fn repeated_observations_preserve_probability(
                xs in finite_sample(1),
                query in finite_value(),
            ) {
                let original = Ecdf::new(&xs).unwrap();
                let repeated_sample = xs.repeat(2);
                let repeated = Ecdf::new(&repeated_sample).unwrap();
                let expected = original.cdf(query);
                let result = repeated.cdf(query);

                prop_assert_eq!(result.to_bits(), expected.to_bits());
            }
        }
    }

    mod sf {
        use super::*;

        #[test]
        fn excludes_all_ties() {
            let xs = [2.0, -1.0, 2.0, 0.0, 2.0, 4.0];
            let distribution = Ecdf::new(&xs).unwrap();
            let query = 2.0;
            let expected = 1.0 / 6.0;

            assert_eq!(distribution.sf(query), expected);
        }

        #[test]
        fn stays_constant_between_observations() {
            let xs = [10.0, 0.0];
            let distribution = Ecdf::new(&xs).unwrap();

            for query in [0.0, 2.5, 5.0, 7.5, 10.0_f64.next_down()] {
                assert_eq!(distribution.sf(query), 0.5);
            }
        }

        #[test]
        fn handles_queries_below_at_and_above_sample_bounds() {
            let xs = [-2.0, 4.0];
            let distribution = Ecdf::new(&xs).unwrap();

            assert_eq!(distribution.sf(-3.0), 1.0);
            assert_eq!(distribution.sf(-2.0), 0.5);
            assert_eq!(distribution.sf(4.0), 0.0);
            assert_eq!(distribution.sf(5.0), 0.0);
        }

        #[test]
        fn infinite_queries_return_probability_endpoints() {
            let xs = [-f64::MAX, 0.0, f64::MAX];
            let distribution = Ecdf::new(&xs).unwrap();

            assert_eq!(distribution.sf(f64::NEG_INFINITY), 1.0);
            assert_eq!(distribution.sf(f64::INFINITY), 0.0);
        }

        #[test]
        fn nan_query_returns_nan() {
            let xs = [1.0];
            let distribution = Ecdf::new(&xs).unwrap();

            assert!(distribution.sf(f64::NAN).is_nan());
        }

        #[test]
        fn singleton_and_constant_samples_have_one_jump() {
            let samples: &[&[f64]] = &[&[-2.0], &[-2.0, -2.0, -2.0]];

            for xs in samples {
                let distribution = Ecdf::new(xs).unwrap();

                assert_eq!(distribution.sf((-2.0_f64).next_down()), 1.0);
                assert_eq!(distribution.sf(-2.0), 0.0);
            }
        }

        #[test]
        fn treats_signed_zeros_as_ties() {
            let xs = [0.0, -1.0, -0.0, 1.0];
            let distribution = Ecdf::new(&xs).unwrap();

            for query in [-0.0, 0.0] {
                assert_eq!(distribution.sf(query), 0.25);
            }
        }

        #[test]
        fn zero_probabilities_are_positive_zero() {
            let xs = [-0.0, 0.0];
            let distribution = Ecdf::new(&xs).unwrap();
            let expected = 0.0_f64;

            assert_eq!(distribution.sf(0.0).to_bits(), expected.to_bits());
        }

        #[test]
        fn uses_the_exceedance_count_directly() {
            let mut xs = [0.0; 10];
            xs[9] = 1.0;
            let distribution = Ecdf::new(&xs).unwrap();
            let expected = 1.0 / 10.0;

            assert_eq!(distribution.sf(0.0), expected);
        }

        #[test]
        fn distinguishes_adjacent_values_at_extreme_scales() {
            let tiny = f64::from_bits(1);
            let samples = [
                [f64::MAX.next_down(), f64::MAX],
                [-f64::MAX, (-f64::MAX).next_up()],
                [tiny, tiny.next_up()],
            ];

            for xs in samples {
                let distribution = Ecdf::new(&xs).unwrap();
                let query = xs[0];

                assert_eq!(distribution.sf(query), 0.5);
            }
        }

        proptest! {
            #[test]
            fn matches_direct_counts(
                xs in finite_sample(1),
                query in finite_value(),
            ) {
                let distribution = Ecdf::new(&xs).unwrap();
                let count = xs.iter().filter(|&&x| x > query).count();
                let expected = usize_to_f64(count) / usize_to_f64(xs.len());
                let result = distribution.sf(query);

                prop_assert_eq!(result.to_bits(), expected.to_bits());
            }

            #[test]
            fn probability_is_bounded(
                xs in finite_sample(1),
                query in finite_value(),
            ) {
                let distribution = Ecdf::new(&xs).unwrap();
                let result = distribution.sf(query);

                prop_assert!((0.0..=1.0).contains(&result));
            }

            #[test]
            fn is_nonincreasing(
                xs in finite_sample(1),
                q1 in finite_value(),
                q2 in finite_value(),
            ) {
                let distribution = Ecdf::new(&xs).unwrap();
                let lower = q1.min(q2);
                let upper = q1.max(q2);
                let lower_probability = distribution.sf(lower);
                let upper_probability = distribution.sf(upper);

                prop_assert!(lower_probability >= upper_probability);
            }

            #[test]
            fn is_invariant_to_input_order(
                xs in finite_sample(1),
                query in finite_value(),
            ) {
                let original = Ecdf::new(&xs).unwrap();
                let mut reversed = xs;
                reversed.reverse();

                let reordered = Ecdf::new(&reversed).unwrap();

                let expected = original.sf(query);
                let result = reordered.sf(query);

                prop_assert_eq!(result.to_bits(), expected.to_bits());
            }

            #[test]
            fn positive_power_of_two_scaling_preserves_probability(
                xs in finite_sample(1),
                query in finite_value(),
                scale in positive_power_of_two_scale(),
            ) {
                let original = Ecdf::new(&xs).unwrap();
                let scaled_sample: Vec<_> = xs.iter().map(|&x| x * scale).collect();
                let scaled = Ecdf::new(&scaled_sample).unwrap();
                let scaled_query = query * scale;
                let expected = original.sf(query);
                let result = scaled.sf(scaled_query);

                prop_assert_eq!(result.to_bits(), expected.to_bits());
            }

            #[test]
            fn repeated_observations_preserve_probability(
                xs in finite_sample(1),
                query in finite_value(),
            ) {
                let original = Ecdf::new(&xs).unwrap();
                let repeated_sample = xs.repeat(2);
                let repeated = Ecdf::new(&repeated_sample).unwrap();
                let expected = original.sf(query);
                let result = repeated.sf(query);

                prop_assert_eq!(result.to_bits(), expected.to_bits());
            }

            #[test]
            fn complements_cdf_within_rounding_error(
                xs in finite_sample(1),
                query in finite_value(),
            ) {
                let distribution = Ecdf::new(&xs).unwrap();
                let cumulative = distribution.cdf(query);
                let survival = distribution.sf(query);
                let total = cumulative + survival;

                prop_assert!((total - 1.0).abs() <= f64::EPSILON);
            }
        }
    }

    mod inverse_cdf {
        use super::*;

        #[test]
        fn selects_observations_without_interpolation() {
            let xs = [10.0, 0.0];
            let distribution = Ecdf::new(&xs).unwrap();

            assert_eq!(distribution.inverse_cdf(0.25), 0.0);
            assert_eq!(distribution.inverse_cdf(0.5), 0.0);
            assert_eq!(distribution.inverse_cdf(0.75), 10.0);
        }

        #[test]
        fn selects_observations_across_half_probability() {
            let xs = [10.0, 0.0];
            let distribution = Ecdf::new(&xs).unwrap();
            let boundary = 0.5_f64;

            assert_eq!(distribution.inverse_cdf(boundary.next_down()), 0.0);
            assert_eq!(distribution.inverse_cdf(boundary), 0.0);
            assert_eq!(distribution.inverse_cdf(boundary.next_up()), 10.0);
        }

        #[test]
        fn endpoints_return_extreme_observations_without_arithmetic() {
            let xs = [f64::MAX, -f64::MAX];
            let distribution = Ecdf::new(&xs).unwrap();

            assert_eq!(distribution.inverse_cdf(0.0), -f64::MAX);
            assert_eq!(distribution.inverse_cdf(-0.0), -f64::MAX);
            assert_eq!(distribution.inverse_cdf(1.0), f64::MAX);
        }

        #[test]
        fn smallest_positive_probability_selects_minimum() {
            let xs = [3.0, -4.0, 1.0];
            let distribution = Ecdf::new(&xs).unwrap();
            let probability = f64::from_bits(1);

            assert_eq!(distribution.inverse_cdf(probability), -4.0);
        }

        #[test]
        fn singleton_and_constant_samples_return_their_value() {
            let samples: &[&[f64]] = &[&[2.5], &[2.5, 2.5, 2.5]];

            for xs in samples {
                let distribution = Ecdf::new(xs).unwrap();

                for p in [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0] {
                    assert_eq!(distribution.inverse_cdf(p), 2.5);
                }
            }
        }

        #[test]
        fn handles_unequal_tie_frequencies() {
            let xs = [2.0, -1.0, 2.0, -1.0, 4.0, 2.0, 4.0, 2.0];
            let distribution = Ecdf::new(&xs).unwrap();
            let lower_jump = 0.25_f64;
            let upper_jump = 0.75_f64;

            assert_eq!(distribution.inverse_cdf(lower_jump), -1.0);
            assert_eq!(distribution.inverse_cdf(lower_jump.next_up()), 2.0);
            assert_eq!(distribution.inverse_cdf(upper_jump), 2.0);
            assert_eq!(distribution.inverse_cdf(upper_jump.next_up()), 4.0);
        }

        #[test]
        fn preserves_subnormal_observations() {
            let tiny = f64::from_bits(1);
            let xs = [tiny, -tiny];
            let distribution = Ecdf::new(&xs).unwrap();

            assert_eq!(distribution.inverse_cdf(0.5), -tiny);
            assert_eq!(distribution.inverse_cdf(0.75), tiny);
        }

        #[test]
        fn signed_zero_selection_is_independent_of_input_order() {
            let samples = [[-0.0, 0.0], [0.0, -0.0]];

            for xs in samples {
                let distribution = Ecdf::new(&xs).unwrap();

                assert_eq!(
                    distribution.inverse_cdf(0.0).to_bits(),
                    (-0.0_f64).to_bits()
                );
                assert_eq!(distribution.inverse_cdf(1.0).to_bits(), 0.0_f64.to_bits());
            }
        }

        #[test]
        fn invalid_probabilities_return_nan() {
            let xs = [1.0, 2.0, 3.0];
            let distribution = Ecdf::new(&xs).unwrap();
            let probabilities = [
                0.0_f64.next_down(),
                1.0_f64.next_up(),
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
            ];

            for p in probabilities {
                assert!(distribution.inverse_cdf(p).is_nan());
            }
        }

        proptest! {
            #[test]
            fn selects_a_sample_observation(
                xs in finite_sample(1),
                p in probability(),
            ) {
                let distribution = Ecdf::new(&xs).unwrap();
                let result = distribution.inverse_cdf(p);

                prop_assert!(result.is_finite());
                prop_assert!(xs.contains(&result));
            }

            #[test]
            fn is_monotonic(
                xs in finite_sample(1),
                p1 in probability(),
                p2 in probability(),
            ) {
                let distribution = Ecdf::new(&xs).unwrap();
                let lower = p1.min(p2);
                let upper = p1.max(p2);

                prop_assert!(distribution.inverse_cdf(lower) <= distribution.inverse_cdf(upper));
            }

            #[test]
            fn constant_sample_returns_the_constant(
                xs in constant_sample(1),
                p in probability(),
            ) {
                let distribution = Ecdf::new(&xs).unwrap();
                let expected = xs[0];

                prop_assert_eq!(distribution.inverse_cdf(p).to_bits(), expected.to_bits());
            }

            #[test]
            fn is_invariant_to_input_order(
                xs in finite_sample(1),
                p in probability(),
            ) {
                let original = Ecdf::new(&xs).unwrap();
                let mut reversed = xs;
                reversed.reverse();

                let reordered = Ecdf::new(&reversed).unwrap();

                let expected = original.inverse_cdf(p);
                let result = reordered.inverse_cdf(p);

                prop_assert_eq!(result.to_bits(), expected.to_bits());
            }

            #[test]
            fn positive_power_of_two_scaling_scales_quantile(
                xs in finite_sample(1),
                p in probability(),
                scale in positive_power_of_two_scale(),
            ) {
                let original = Ecdf::new(&xs).unwrap();
                let scaled_sample: Vec<_> = xs.iter().map(|&x| x * scale).collect();
                let scaled = Ecdf::new(&scaled_sample).unwrap();
                let expected = original.inverse_cdf(p) * scale;
                let result = scaled.inverse_cdf(p);

                prop_assert_eq!(result.to_bits(), expected.to_bits());
            }
        }
    }

    mod error {
        use super::*;

        #[test]
        fn describes_empty_sample() {
            let error = EcdfError::EmptySample;
            let expected = "sample must contain at least one observation";
            let result = error.to_string();

            assert_eq!(result, expected);
        }

        #[test]
        fn describes_nonfinite_observation() {
            let error = EcdfError::NonFiniteObservation { index: 2, value: f64::INFINITY };
            let expected = "observation at index 2 must be finite, got inf";
            let result = error.to_string();

            assert_eq!(result, expected);
        }
    }
}
