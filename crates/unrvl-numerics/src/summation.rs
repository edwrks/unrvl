//! Compensated summation utilities for improved floating-point accuracy.

/// Accumulates values using Neumaier compensated summation.
///
/// This accumulator is intended for finite operands. It does not provide
/// extended-real arithmetic: `NaN`, infinities, or intermediate overflow can
/// produce a non-finite total.
///
/// A new empty accumulator is created with [`NeumaierSum::new`].
///
/// Values can be added individually using [`NeumaierSum::add`], `+=`, or `+`.
///
/// An existing accumulator can be extended by an iterator using
/// [`Extend::extend`].
///
/// A new accumulator can also be created from an iterator using
/// [`Iterator::sum`] or [`Iterator::collect`]. Both owned `f64` values and
/// borrowed `&f64` values are supported.
///
/// Use [`NeumaierSum::total`] to retrieve the accumulated result.
///
/// # Examples
///
/// Adding values individually:
///
/// ```
/// # use unrvl_numerics::summation::NeumaierSum;
/// let mut acc = NeumaierSum::new();
/// acc += 1.0;
/// acc.add(0.5);
/// acc += 0.5;
///
/// assert_eq!(acc.total(), 2.0);
/// ```
///
/// Extending an existing accumulator:
///
/// ```
/// # use unrvl_numerics::summation::NeumaierSum;
/// let mut acc = NeumaierSum::new();
/// acc.extend([2.0, 3.0]);
///
/// assert_eq!(acc.total(), 5.0);
/// ```
///
/// Creating an accumulator from an iterator:
///
/// ```
/// # use unrvl_numerics::summation::NeumaierSum;
/// let xs = [1.0, 2.0, 3.0];
///
/// let summed = xs.iter().sum::<NeumaierSum>();
/// let collected = xs.iter().collect::<NeumaierSum>();
///
/// assert_eq!(summed.total(), 6.0);
/// assert_eq!(collected.total(), 6.0);
/// ```
#[must_use]
#[derive(Debug, Clone, Copy)]
pub struct NeumaierSum {
    /// Accumulated sum.
    sum: f64,

    /// Compensation value.
    c: f64,
}

impl NeumaierSum {
    /// Creates an empty accumulator.
    ///
    /// # Examples
    ///
    /// ```
    /// # use unrvl_numerics::summation::NeumaierSum;
    /// let mut acc = NeumaierSum::new();
    ///
    /// assert_eq!(acc.total(), 0.0);
    /// ```
    pub const fn new() -> Self {
        Self { sum: 0.0, c: 0.0 }
    }

    /// Creates an accumulator with a starting sum of `x` and compensation of
    /// zero.
    ///
    /// # Examples
    ///
    /// ```
    /// # use unrvl_numerics::summation::NeumaierSum;
    /// let mut acc = NeumaierSum::with_value(1.5);
    ///
    /// assert_eq!(acc.total(), 1.5);
    /// ```
    pub const fn with_value(x: f64) -> Self {
        Self { sum: x, c: 0.0 }
    }

    /// Adds a value to the accumulator.
    ///
    /// # Examples
    ///
    /// ```
    /// # use unrvl_numerics::summation::NeumaierSum;
    /// let mut acc = NeumaierSum::new();
    /// acc.add(42.1);
    /// assert_eq!(acc.total(), 42.1);
    /// ```
    pub fn add(&mut self, x: f64) {
        let t = self.sum + x;

        if self.sum.abs() >= x.abs() {
            self.c += (self.sum - t) + x;
        } else {
            self.c += (x - t) + self.sum;
        }

        self.sum = t;
    }

    /// The current total summation.
    #[must_use]
    pub const fn total(&self) -> f64 {
        self.sum + self.c
    }
}

impl Default for NeumaierSum {
    fn default() -> Self {
        Self::new()
    }
}

impl From<f64> for NeumaierSum {
    fn from(value: f64) -> Self {
        Self::with_value(value)
    }
}

impl std::ops::AddAssign<f64> for NeumaierSum {
    fn add_assign(&mut self, rhs: f64) {
        self.add(rhs);
    }
}

impl std::ops::AddAssign<&f64> for NeumaierSum {
    fn add_assign(&mut self, rhs: &f64) {
        self.add(*rhs);
    }
}

impl std::ops::Add<f64> for NeumaierSum {
    type Output = Self;

    fn add(mut self, rhs: f64) -> Self::Output {
        self += rhs;
        self
    }
}

impl std::ops::Add<&f64> for NeumaierSum {
    type Output = Self;

    fn add(mut self, rhs: &f64) -> Self::Output {
        self += *rhs;
        self
    }
}

impl Extend<f64> for NeumaierSum {
    fn extend<T: IntoIterator<Item = f64>>(&mut self, iter: T) {
        for x in iter {
            *self += x;
        }
    }
}

impl<'a> Extend<&'a f64> for NeumaierSum {
    fn extend<T: IntoIterator<Item = &'a f64>>(&mut self, iter: T) {
        for &x in iter {
            *self += x;
        }
    }
}

impl FromIterator<f64> for NeumaierSum {
    fn from_iter<T: IntoIterator<Item = f64>>(iter: T) -> Self {
        let mut acc = Self::new();
        acc.extend(iter);
        acc
    }
}

impl<'a> FromIterator<&'a f64> for NeumaierSum {
    fn from_iter<T: IntoIterator<Item = &'a f64>>(iter: T) -> Self {
        let mut acc = Self::new();
        acc.extend(iter);
        acc
    }
}

impl std::iter::Sum<f64> for NeumaierSum {
    fn sum<I: Iterator<Item = f64>>(iter: I) -> Self {
        let mut acc = Self::new();
        acc.extend(iter);
        acc
    }
}

impl<'a> std::iter::Sum<&'a f64> for NeumaierSum {
    fn sum<I: Iterator<Item = &'a f64>>(iter: I) -> Self {
        let mut acc = Self::new();
        acc.extend(iter);
        acc
    }
}

/// Extension methods for summing iterator values with Neumaier summation.
pub trait NeumaierSumExt: Iterator + Sized {
    /// Consumes the iterator and returns its Neumaier-compensated sum.
    ///
    /// # Examples
    ///
    /// ```
    /// # use unrvl_numerics::summation::NeumaierSumExt;
    /// let total = [1.0, 2.0, 3.0].iter().neumaier_sum();
    /// assert_eq!(total, 6.0);
    /// ```
    fn neumaier_sum(self) -> f64
    where
        NeumaierSum: std::iter::Sum<Self::Item>,
    {
        self.sum::<NeumaierSum>().total()
    }
}

impl<I: Iterator> NeumaierSumExt for I {}

#[cfg(test)]
mod tests {
    use test_utils::approx::{Tolerance, assert_approx_eq};

    use super::*;

    #[test]
    fn starts_at_zero() {
        let acc = NeumaierSum::new();
        assert_eq!(acc.total(), 0.0);
    }

    #[test]
    fn handles_empty_iterable() {
        let xs: Vec<f64> = Vec::new();
        let mut acc = NeumaierSum::new();
        acc.extend(xs);

        assert_eq!(acc.total(), 0.0);
    }

    #[test]
    fn add() {
        let x = 1.5;
        let acc = NeumaierSum::new();
        let res = acc + x;

        assert_eq!(res.total(), x);
    }

    #[test]
    fn add_borrow() {
        let x = 1.5;
        let acc = NeumaierSum::new();
        let res = <NeumaierSum as std::ops::Add<&f64>>::add(acc, &x);

        assert_eq!(res.total(), x);
    }

    #[test]
    fn add_assign() {
        let x = 1.5;
        let mut acc = NeumaierSum::new();
        acc += x;

        assert_eq!(acc.total(), x);
    }

    #[test]
    fn add_assign_borrow() {
        let x = 1.5;
        let mut acc = NeumaierSum::new();
        acc += &x;

        assert_eq!(acc.total(), x);
    }

    #[test]
    fn extend() {
        let xs = [1.0, 2.0, 3.0];
        let mut acc = NeumaierSum::new();
        acc.extend(xs);

        assert_eq!(acc.total(), 6.0);
    }

    #[test]
    fn extend_borrow() {
        let xs = [1.0, 2.0, 3.0];
        let mut acc = NeumaierSum::new();
        acc.extend(&xs);

        assert_eq!(acc.total(), 6.0);
    }

    #[test]
    fn collect() {
        let acc: NeumaierSum = [1.0, 2.0, 3.0].into_iter().collect();
        assert_eq!(acc.total(), 6.0);
    }

    #[test]
    fn collect_borrow() {
        let acc: NeumaierSum = [1.0, 2.0, 3.0].iter().collect();
        assert_eq!(acc.total(), 6.0);
    }

    #[test]
    fn sum() {
        let acc: NeumaierSum = [1.0, 2.0, 3.0].into_iter().sum();
        assert_eq!(acc.total(), 6.0);
    }

    #[test]
    fn sum_borrow() {
        let acc: NeumaierSum = [1.0, 2.0, 3.0].iter().sum();
        assert_eq!(acc.total(), 6.0);
    }

    #[test]
    fn propagates_nan() {
        let xs = [1.0, f64::NAN, 2.0];
        assert!(xs.iter().neumaier_sum().is_nan());
    }

    #[test]
    fn positive_infinity_produces_nan() {
        let xs = [1.0, f64::INFINITY, 2.0];
        assert!(xs.iter().neumaier_sum().is_nan());
    }

    #[test]
    fn negative_infinity_produce_nan() {
        let xs = [1.0, f64::NEG_INFINITY, 2.0];
        assert!(xs.iter().neumaier_sum().is_nan());
    }

    #[test]
    fn opposite_infinities_produce_nan() {
        let xs = [f64::INFINITY, f64::NEG_INFINITY];
        assert!(xs.iter().neumaier_sum().is_nan());
    }

    #[test]
    fn neumaier_sum_ext() {
        let total = [1.0, 2.0, 3.0].iter().neumaier_sum();
        assert_eq!(total, 6.0);
    }

    #[test]
    fn recovers_increments_lost_by_naive_sum() {
        let n = 1_000_000;
        let nf = 1_000_000.0;
        let large = 2.0_f64.powi(53);

        // One large value followed by many tiny additions is the canonical
        // pathology for naive accumulation. At 2^53, the ULP is ~2.0.
        // Increments of 1.0 are below representable precision and are
        // ignored by naive summation.
        let xs = || std::iter::once(large).chain(std::iter::repeat_n(1.0, n));

        let naive: f64 = xs().sum();
        let compensated: f64 = xs().neumaier_sum();
        let expected = large + nf;

        assert_eq!(large + 1.0, large);
        assert_eq!(naive, large);
        assert_eq!(compensated, expected);
    }

    #[test]
    fn sums_ordinary_decimal_values_within_tolerance() {
        let xs = [0.1; 10];

        let tol = Tolerance::new(1e-16, 0.0);
        assert_approx_eq(xs.iter().neumaier_sum(), 1.0, tol);
    }

    #[test]
    fn preserves_small_sum_before_large_cancellation() {
        let xs = [1.0, 1e100, -1e100];

        assert_eq!(xs.iter().neumaier_sum(), 1.0);
    }
}
