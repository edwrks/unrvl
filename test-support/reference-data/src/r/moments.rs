//! R conformance fixtures and reference values for moment statistics.
//!
//! Select a [`Dataset`] and call [`Dataset::load`] to obtain its observations
//! and corresponding [`Statistics`].
//!
//! References cover the arithmetic mean, sample variance with denominator
//! `n - 1`, and `e1071` Type-2 skewness and excess kurtosis.
//!
//! The fixture corpus is designed to produce finite values for every referenced
//! statistic. Fixtures and generated results are embedded in the crate; loading
//! them does not run R or require network access.

use super::parsing::{parse_finite, parse_usize, parse_value_fixture};

/// A bundled R moments dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dataset {
    /// Uneven tails.
    AsymmetricTail,

    /// Ordinary decimal observations.
    Baseline,

    /// Four observations and small-sample corrections.
    MinimumShapeSample,

    /// Zero skewness.
    Symmetric,

    /// Repeated integer observations and quantile boundaries.
    Ties,
}

impl Dataset {
    const REFERENCES: &'static str = include_str!("../../data/r/moments.csv");

    /// Loads the fixture and its corresponding R reference statistics.
    ///
    /// # Panics
    ///
    /// Panics if the bundled fixture or reference data is malformed.
    #[must_use]
    pub fn load(self) -> Reference {
        Parser::new(self).parse()
    }

    const fn case_name(self) -> &'static str {
        match self {
            Self::AsymmetricTail => "asymmetric-tail",
            Self::Baseline => "baseline",
            Self::MinimumShapeSample => "minimum-shape-sample",
            Self::Symmetric => "symmetric",
            Self::Ties => "ties",
        }
    }

    const fn fixture(self) -> &'static str {
        match self {
            Self::AsymmetricTail => {
                include_str!("../../data/fixtures/r/moments/asymmetric-tail.csv")
            }
            Self::Baseline => {
                include_str!("../../data/fixtures/r/moments/baseline.csv")
            }
            Self::MinimumShapeSample => {
                include_str!("../../data/fixtures/r/moments/minimum-shape-sample.csv")
            }
            Self::Symmetric => {
                include_str!("../../data/fixtures/r/moments/symmetric.csv")
            }
            Self::Ties => {
                include_str!("../../data/fixtures/r/moments/ties.csv")
            }
        }
    }
}

/// One moments fixture and its R reference statistics.
#[derive(Debug)]
pub struct Reference {
    dataset: Dataset,
    observations: Vec<f64>,
    statistics: Statistics,
}

impl Reference {
    /// The bundled dataset this reference was loaded from.
    #[must_use]
    pub const fn dataset(&self) -> Dataset {
        self.dataset
    }

    /// Observed values in the dataset.
    #[must_use]
    pub fn observations(&self) -> &[f64] {
        &self.observations
    }

    /// Reference statistics computed by R.
    #[must_use]
    pub const fn statistics(&self) -> &Statistics {
        &self.statistics
    }
}

/// Reference moment statistics computed by R.
#[derive(Debug, Clone, Copy)]
pub struct Statistics {
    n: usize,
    mean: f64,
    variance: f64,
    skewness: f64,
    excess_kurtosis: f64,
}

impl Statistics {
    /// Number of observations in the dataset.
    #[must_use]
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Arithmetic mean.
    #[must_use]
    pub const fn mean(&self) -> f64 {
        self.mean
    }

    /// Sample variance with denominator `n - 1`.
    #[must_use]
    pub const fn variance(&self) -> f64 {
        self.variance
    }

    /// Adjusted sample skewness (e1071 Type 2).
    #[must_use]
    pub const fn skewness(&self) -> f64 {
        self.skewness
    }

    /// Adjusted excess kurtosis (e1071 Type 2).
    #[must_use]
    pub const fn excess_kurtosis(&self) -> f64 {
        self.excess_kurtosis
    }
}

struct Parser {
    dataset: Dataset,
}

impl Parser {
    const REFERENCE_HEADER: &'static str = "case,n,mean,variance,skewness,excess_kurtosis";

    const fn new(dataset: Dataset) -> Self {
        Self { dataset }
    }

    fn parse(self) -> Reference {
        let observations = parse_value_fixture(self.dataset.fixture(), "moments fixture");
        let statistics = self.parse_statistics();

        assert_eq!(
            observations.len(),
            statistics.n,
            "fixture observation count should match the R reference count"
        );

        Reference { dataset: self.dataset, observations, statistics }
    }

    fn parse_statistics(&self) -> Statistics {
        let case = self.dataset.case_name();

        let mut reference = Dataset::REFERENCES.lines();

        let header = reference
            .next()
            .expect("moments reference CSV should contain a header");

        assert_eq!(
            header,
            Self::REFERENCE_HEADER,
            "moments reference CSV header should match the expected schema"
        );

        let row = reference
            .find(|line| {
                line.split_once(',')
                    .is_some_and(|(row_case, _)| row_case == case)
            })
            .expect("moments reference data should contain the requested case");

        let fields: Vec<_> = row.split(',').collect();

        assert_eq!(
            fields.len(),
            6,
            "moments reference row should contain exactly six fields"
        );

        let n = parse_usize(fields[1], "reference n");
        let mean = parse_finite(fields[2], "reference mean");
        let variance = parse_finite(fields[3], "reference variance");
        let skewness = parse_finite(fields[4], "reference skewness");
        let excess_kurtosis = parse_finite(fields[5], "reference excess_kurtosis");

        Statistics { n, mean, variance, skewness, excess_kurtosis }
    }
}

#[cfg(test)]
mod tests {
    use super::Dataset;

    #[test]
    fn all_datasets_load() {
        let datasets = [
            Dataset::AsymmetricTail,
            Dataset::Baseline,
            Dataset::MinimumShapeSample,
            Dataset::Symmetric,
            Dataset::Ties,
        ];

        for dataset in datasets {
            let reference = dataset.load();
            let statistics = reference.statistics();

            assert_eq!(reference.dataset(), dataset);
            assert_ne!(reference.observations(), []);
            assert_eq!(statistics.n(), reference.observations().len());
        }
    }
}
