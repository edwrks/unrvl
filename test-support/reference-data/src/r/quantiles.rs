//! R conformance fixtures and reference values for quantile statistics.
//!
//! Select a [`Dataset`] and call [`Dataset::load`] to obtain its observations
//! and corresponding [`Statistics`] entries.
//!
//! Each entry contains a probability and its Type 7 quantile computed by R.
//! Entries are returned in ascending probability order.
//!
//! Fixtures and generated results are embedded in the crate. Loading them
//! does not run R or require network access. Observations and reference
//! values must be finite, and reference probabilities must be unique and
//! lie in `[0, 1]`.

use super::parsing::{parse_finite, parse_probability, parse_value_fixture};

/// A bundled R quantile dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dataset {
    /// Uneven tails.
    AsymmetricTail,

    /// Ordinary decimal observations.
    Baseline,

    /// Four observations with interpolation between order statistics.
    MinimumShapeSample,

    /// Repeated integer observations and quantile boundaries.
    Ties,
}

impl Dataset {
    const REFERENCES: &'static str = include_str!("../../data/r/quantiles.csv");

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
            Self::Ties => "ties",
        }
    }

    const fn fixture(self) -> &'static str {
        match self {
            Self::AsymmetricTail => {
                include_str!("../../data/fixtures/r/quantiles/asymmetric-tail.csv")
            }
            Self::Baseline => {
                include_str!("../../data/fixtures/r/quantiles/baseline.csv")
            }
            Self::MinimumShapeSample => {
                include_str!("../../data/fixtures/r/quantiles/minimum-shape-sample.csv")
            }
            Self::Ties => {
                include_str!("../../data/fixtures/r/quantiles/ties.csv")
            }
        }
    }
}

/// One quantile fixture and its R reference statistics.
#[derive(Debug)]
pub struct Reference {
    dataset: Dataset,
    observations: Vec<f64>,
    statistics: Vec<Statistics>,
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

    /// Reference quantiles in ascending probability order.
    #[must_use]
    pub fn statistics(&self) -> &[Statistics] {
        &self.statistics
    }
}

/// One probability and its Type 7 quantile computed by R.
#[derive(Debug, Clone, Copy)]
pub struct Statistics {
    probability: f64,
    value: f64,
}

impl Statistics {
    /// Probability in the closed interval `[0, 1]`.
    #[must_use]
    pub const fn probability(&self) -> f64 {
        self.probability
    }

    /// Reference quantile at this probability.
    #[must_use]
    pub const fn value(&self) -> f64 {
        self.value
    }
}

struct Parser {
    dataset: Dataset,
}

impl Parser {
    const REFERENCE_HEADER: &'static str = "case,probability,value";

    const fn new(dataset: Dataset) -> Self {
        Self { dataset }
    }

    fn parse(self) -> Reference {
        let observations = parse_value_fixture(self.dataset.fixture(), "quantile fixture");
        let statistics = self.parse_statistics();

        Reference { dataset: self.dataset, observations, statistics }
    }

    fn parse_statistics(&self) -> Vec<Statistics> {
        let case = self.dataset.case_name();
        let mut reference = Dataset::REFERENCES.lines();

        let header = reference
            .next()
            .expect("quantile reference CSV should contain a header");

        assert_eq!(
            header,
            Self::REFERENCE_HEADER,
            "quantile reference CSV header should match the expected schema"
        );

        let mut statistics = Vec::new();

        for line in reference {
            let fields: Vec<_> = line.split(',').collect();
            if fields[0] != case {
                continue;
            }

            assert_eq!(
                fields.len(),
                3,
                "quantile reference row should contain exactly three fields"
            );

            let probability = parse_probability(fields[1], "reference probability");
            let value = parse_finite(fields[2], "reference quantile");

            statistics.push(Statistics { probability, value });
        }

        assert!(
            !statistics.is_empty(),
            "quantile reference data should contain the requested case"
        );

        assert!(
            statistics
                .windows(2)
                .all(|pair| { pair[0].probability < pair[1].probability }),
            "reference probabilities should be unique and in ascending order"
        );

        statistics
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
            Dataset::Ties,
        ];

        for dataset in datasets {
            let reference = dataset.load();
            let statistics = reference.statistics();

            assert_eq!(reference.dataset(), dataset);
            assert_ne!(reference.observations(), []);
            assert!(!statistics.is_empty());
        }
    }
}
