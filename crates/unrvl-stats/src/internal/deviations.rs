use unrvl_numerics::convert::usize_to_f64;
use unrvl_numerics::scaling::power_of_two_unit;
use unrvl_numerics::summation::NeumaierSumExt;

/// A validated sample prepared for corrected centering in scaled units.
/// Borrows the original observations without copying or allocating.
#[derive(Debug, Clone, Copy)]
pub struct CenteredSample<'a> {
    values: &'a [f64],
    centering: Centering,
}

/// Copyable centering parameters that can outlive the original observations.
/// The rounded center and correction remain separate to preserve small
/// differences.
#[derive(Debug, Clone, Copy)]
pub struct Centering {
    unit: f64,
    center: f64,
    correction: f64,
}

impl<'a> CenteredSample<'a> {
    /// Validates and prepares a nonempty finite sample in `O(n)` time and
    /// `O(1)` additional memory. Returns `None` for empty or non-finite input.
    ///
    /// Uses a positive power-of-two unit and Neumaier summation for the mean
    /// and its correction. Constant samples produce zero deviations; all-zero
    /// samples use a unit of one. Scaling can still lose values that are very
    /// small relative to the sample's largest magnitude.
    ///
    /// # Panics
    ///
    /// Panics if a finite sample contains more than `2^53` observations.
    #[expect(clippy::float_cmp, reason = "detect exactly constant")]
    pub fn new(values: &'a [f64]) -> Option<Self> {
        let &first = values.first()?;

        let mut maximum = 0.0_f64;
        let mut is_constant = true;

        for &x in values {
            if !x.is_finite() {
                return None;
            }

            maximum = maximum.max(x.abs());
            is_constant &= x == first;
        }

        let unit = power_of_two_unit(maximum);
        let n = usize_to_f64(values.len());
        let scaled = values.iter().map(move |&x| x / unit);

        let center = if is_constant {
            // Preserve a constant sample's scaled center exactly.
            first / unit
        } else {
            scaled.clone().neumaier_sum() / n
        };

        let correction = scaled.clone().map(|x| x - center).neumaier_sum() / n;

        Some(Self {
            values,
            centering: Centering { unit, center, correction },
        })
    }

    /// Copies the parameters without retaining a reference to the observations.
    pub const fn centering(&self) -> Centering {
        self.centering
    }

    /// Iterates over this sample's deviations without copying observations.
    pub fn deviations(&self) -> impl Iterator<Item = f64> + Clone + '_ {
        let centering = self.centering;
        self.values.iter().map(move |&x| centering.deviation(x))
    }
}

impl Centering {
    /// Positive, finite power-of-two scaling unit.
    pub const fn unit(&self) -> f64 {
        self.unit
    }

    /// Rounded center expressed in scaled units.
    pub const fn scaled_center(&self) -> f64 {
        self.center
    }

    /// Separately retained mean correction, expressed in scaled units.
    pub const fn scaled_correction(&self) -> f64 {
        self.correction
    }

    /// Applies the stored centering to one value, including a new observation.
    /// Returns a scaled deviation; does not validate or recompute the center.
    pub fn deviation(&self, x: f64) -> f64 {
        (x / self.unit - self.center) - self.correction
    }
}

#[cfg(test)]
mod tests {
    use test_utils::approx::{Tolerance, assert_all_approx_eq, assert_approx_eq};

    use super::*;

    #[test]
    fn rejects_empty_or_nonfinite_samples() {
        assert!(CenteredSample::new(&[]).is_none());

        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(CenteredSample::new(&[1.0, value]).is_none());
        }
    }

    #[test]
    fn centers_in_input_units() {
        let xs = [1000.0, 1001.0, 1002.0];
        let sample = CenteredSample::new(&xs).unwrap();
        let expected = [-1.0 / 512.0, 0.0, 1.0 / 512.0];

        assert_eq!(sample.centering().unit(), 512.0);
        assert_eq!(sample.deviations().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn centers_subnormal_observations_before_information_is_lost() {
        let d = f64::from_bits(1);
        let xs = [0.0, 0.0, d];
        let sample = CenteredSample::new(&xs).unwrap();
        let expected = [-1.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0];

        assert_eq!(sample.centering().unit(), d);
        assert_all_approx_eq(
            &sample.deviations().collect::<Vec<_>>(),
            &expected,
            Tolerance::VERY_STRICT,
        );
    }

    #[test]
    fn retains_rounding_correction_around_a_large_offset() {
        let xs = [1e16, 1e16, 1e16 + 2.0];
        let sample = CenteredSample::new(&xs).unwrap();
        let unit = sample.centering().unit();
        let reconstructed: Vec<_> = sample.deviations().map(|d| d * unit).collect();
        let expected = [-2.0 / 3.0, -2.0 / 3.0, 4.0 / 3.0];

        assert_all_approx_eq(&reconstructed, &expected, Tolerance::VERY_STRICT);
    }

    #[test]
    fn centers_opposite_finite_extremes_without_overflow() {
        let xs = [-f64::MAX, 0.0, f64::MAX];
        let sample = CenteredSample::new(&xs).unwrap();
        let unit = sample.centering().unit();

        assert!(unit.is_finite());
        assert!(unit > 0.0);

        let sample = sample
            .deviations()
            .zip([-f64::MAX / unit, 0.0, f64::MAX / unit]);

        for (actual, expected) in sample {
            assert_approx_eq(actual, expected, Tolerance::VERY_STRICT);
        }
    }

    #[test]
    fn constants_have_positive_units_and_zero_deviations() {
        let values = [
            0.0,
            42.0,
            f64::from_bits(1),
            -f64::from_bits(1),
            f64::MAX,
            -f64::MAX,
        ];

        for value in values {
            let xs = [value; 4];
            let sample = CenteredSample::new(&xs).unwrap();
            let unit = sample.centering().unit();

            assert!(unit.is_finite());
            assert!(unit > 0.0);

            for deviation in sample.deviations() {
                assert_eq!(deviation, 0.0);
            }
        }
    }

    #[test]
    fn copied_centering_outlives_observations_and_transforms_new_values() {
        let centering = {
            let values = [2.0, 4.0, 6.0];
            let sample = CenteredSample::new(&values).unwrap();
            assert_eq!(sample.deviations().collect::<Vec<_>>(), [-0.5, 0.0, 0.5]);
            sample.centering()
        };

        assert_eq!(centering.unit(), 4.0);
        assert_eq!(centering.scaled_center(), 1.0);
        assert_eq!(centering.scaled_correction(), 0.0);
        assert_eq!(centering.deviation(8.0), 1.0);
    }
}
