//! Paired fixtures and default R `lm(y ~ x)` references.
//!
//! Fit statistics are embedded in one table; predictions and residuals at the
//! fixture observations share a per-observation table. Indices are zero-based
//! and preserve fixture order.

use super::parsing::{parse_finite, parse_probability, parse_usize};

/// A bundled simple linear regression fixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dataset {
    /// Asymmetric observations with a tail.
    AsymmetricTail,

    /// Ordinary decimal observations.
    Baseline,

    /// A small noisy fit with four observations.
    MinimumShapeSample,

    /// An exact line with negative slope.
    PerfectNegative,

    /// An exact line with positive slope.
    PerfectPositive,
}

impl Dataset {
    const REFERENCES: &'static str = include_str!("../../data/r/regression.csv");
    const OBSERVATION_REFERENCES: &'static str =
        include_str!("../../data/r/regression-observations.csv");

    /// Loads paired observations and all R regression references.
    ///
    /// # Panics
    ///
    /// Panics if bundled tables are malformed or their counts/indices do not
    /// agree with the fixture.
    #[must_use]
    pub fn load(self) -> Reference {
        Parser::new(self).parse()
    }

    const fn case_name(self) -> &'static str {
        match self {
            Self::AsymmetricTail => "asymmetric-tail",
            Self::Baseline => "baseline",
            Self::MinimumShapeSample => "minimum-shape-sample",
            Self::PerfectNegative => "perfect-negative",
            Self::PerfectPositive => "perfect-positive",
        }
    }

    const fn fixture(self) -> &'static str {
        match self {
            Self::AsymmetricTail => {
                include_str!("../../data/fixtures/r/regression/asymmetric-tail.csv")
            }
            Self::Baseline => {
                include_str!("../../data/fixtures/r/regression/baseline.csv")
            }
            Self::MinimumShapeSample => {
                include_str!("../../data/fixtures/r/regression/minimum-shape-sample.csv")
            }
            Self::PerfectNegative => {
                include_str!("../../data/fixtures/r/regression/perfect-negative.csv")
            }
            Self::PerfectPositive => {
                include_str!("../../data/fixtures/r/regression/perfect-positive.csv")
            }
        }
    }
}

/// One paired fixture, fitted statistics, predictions, and residuals from R.
#[derive(Debug)]
pub struct Reference {
    dataset: Dataset,
    x: Vec<f64>,
    y: Vec<f64>,
    statistics: Statistics,
    predictions: Vec<f64>,
    residuals: Vec<f64>,
}

impl Reference {
    /// Dataset identity.
    #[must_use]
    pub const fn dataset(&self) -> Dataset {
        self.dataset
    }

    /// Predictor observations in fixture order.
    #[must_use]
    pub fn x(&self) -> &[f64] {
        &self.x
    }

    /// Response observations aligned with [`Self::x`].
    #[must_use]
    pub fn y(&self) -> &[f64] {
        &self.y
    }

    /// Fitted statistics from R.
    #[must_use]
    pub const fn statistics(&self) -> &Statistics {
        &self.statistics
    }

    /// R predictions at each fixture predictor in observation order.
    #[must_use]
    pub fn predictions(&self) -> &[f64] {
        &self.predictions
    }

    /// R residuals in observation order.
    #[must_use]
    pub fn residuals(&self) -> &[f64] {
        &self.residuals
    }
}

/// Scalar statistics from default R `lm(y ~ x)`.
#[derive(Debug, Clone, Copy)]
pub struct Statistics {
    n: usize,
    intercept: f64,
    slope: f64,
    r_squared: f64,
}

impl Statistics {
    /// Number of fitted pairs.
    #[must_use]
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Fitted response at predictor zero.
    #[must_use]
    pub const fn intercept(&self) -> f64 {
        self.intercept
    }

    /// Fitted slope.
    #[must_use]
    pub const fn slope(&self) -> f64 {
        self.slope
    }

    /// Coefficient of determination for the fixture observations.
    #[must_use]
    pub const fn r_squared(&self) -> f64 {
        self.r_squared
    }
}

struct Parser {
    dataset: Dataset,
}

impl Parser {
    const FIXTURE_HEADER: &'static str = "x,y";
    const REFERENCE_HEADER: &'static str = "case,n,intercept,slope,r_squared";
    const OBSERVATION_REFERENCE_HEADER: &'static str = "case,observation,prediction,residual";

    const fn new(dataset: Dataset) -> Self {
        Self { dataset }
    }

    fn parse(self) -> Reference {
        let (x, y) = self.parse_fixture();
        let statistics = self.parse_statistics();
        let (predictions, residuals) = self.parse_observation_references();

        assert_eq!(
            statistics.n,
            x.len(),
            "fixture observation count should match the R reference count"
        );

        assert_eq!(
            predictions.len(),
            x.len(),
            "regression per-observation count should match the fixture"
        );

        Reference {
            dataset: self.dataset,
            x,
            y,
            statistics,
            predictions,
            residuals,
        }
    }

    fn parse_fixture(&self) -> (Vec<f64>, Vec<f64>) {
        let mut lines = self.dataset.fixture().lines();

        let header = lines
            .next()
            .expect("regression fixture should contain a header");

        assert_eq!(
            header,
            Self::FIXTURE_HEADER,
            "regression fixture CSV header should match the expected schema"
        );

        let mut x = Vec::new();
        let mut y = Vec::new();

        for line in lines {
            let fields: Vec<_> = line.split(',').collect();

            assert_eq!(
                fields.len(),
                2,
                "regression fixture rows should contain exactly two fields"
            );

            x.push(parse_finite(fields[0], "fixture x"));
            y.push(parse_finite(fields[1], "fixture y"));
        }

        assert!(
            x.len() >= 2,
            "regression fixture should contain at least two observations"
        );

        (x, y)
    }

    fn parse_statistics(&self) -> Statistics {
        let case = self.dataset.case_name();
        let mut reference = Dataset::REFERENCES.lines();

        let header = reference
            .next()
            .expect("regression reference CSV should contain a header");

        assert_eq!(
            header,
            Self::REFERENCE_HEADER,
            "regression reference CSV header should match the expected schema"
        );

        let mut statistics = None;

        for line in reference {
            let fields: Vec<_> = line.split(',').collect();

            assert_eq!(
                fields.len(),
                5,
                "regression reference rows should contain exactly five fields"
            );

            if fields[0] != case {
                continue;
            }

            assert!(
                statistics.is_none(),
                "regression reference data should contain only one row per case"
            );

            statistics = Some(Statistics {
                n: parse_usize(fields[1], "reference n"),
                intercept: parse_finite(fields[2], "reference intercept"),
                slope: parse_finite(fields[3], "reference slope"),
                r_squared: parse_probability(fields[4], "reference R²"),
            });
        }

        statistics.expect("regression reference data should contain the requested case")
    }

    fn parse_observation_references(&self) -> (Vec<f64>, Vec<f64>) {
        let case = self.dataset.case_name();
        let mut reference = Dataset::OBSERVATION_REFERENCES.lines();

        let header = reference
            .next()
            .expect("regression observation reference CSV should contain a header");

        assert_eq!(
            header,
            Self::OBSERVATION_REFERENCE_HEADER,
            "regression observation reference CSV header should match the expected schema"
        );

        let mut predictions = Vec::new();
        let mut residuals = Vec::new();

        for line in reference {
            let fields: Vec<_> = line.split(',').collect();

            assert_eq!(
                fields.len(),
                4,
                "regression observation reference rows should contain exactly four fields"
            );

            if fields[0] != case {
                continue;
            }

            let observation = parse_usize(fields[1], "reference observation");

            assert_eq!(
                observation,
                predictions.len(),
                "regression reference observations should be in fixture order"
            );

            predictions.push(parse_finite(fields[2], "reference prediction"));
            residuals.push(parse_finite(fields[3], "reference residual"));
        }

        (predictions, residuals)
    }
}

#[cfg(test)]
mod tests {
    use super::Dataset;

    #[test]
    fn all_datasets_load_with_aligned_observations() {
        for (dataset, n) in [
            (Dataset::AsymmetricTail, 7),
            (Dataset::Baseline, 8),
            (Dataset::MinimumShapeSample, 4),
            (Dataset::PerfectNegative, 5),
            (Dataset::PerfectPositive, 5),
        ] {
            let reference = dataset.load();
            assert_eq!(reference.dataset(), dataset);
            assert_eq!(reference.x().len(), n);
            assert_eq!(reference.y().len(), n);
            assert_eq!(reference.statistics().n(), n);
            assert_eq!(reference.predictions().len(), n);
            assert_eq!(reference.residuals().len(), n);
        }
    }
}
