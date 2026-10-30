//! R conformance fixtures and reference values for statistical transformations.
//!
//! Select a [`Dataset`] and call [`Dataset::load`] to obtain its observations
//! and aligned [`Statistics`] entries. Each entry contains a demeaned value,
//! sample z-score, and one-based average rank, in original observation order.
//!
//! Fixtures and reference values are embedded; loading does not run R or
//! require network access. Samples are finite, nonconstant, and contain at
//! least two observations so every reference value is finite.

use super::parsing::{parse_finite, parse_usize, parse_value_fixture};

/// A bundled R transform dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dataset {
    /// Uneven tails with a negative mean.
    AsymmetricTail,

    /// Ordinary decimal observations in unsorted order.
    Baseline,

    /// Positive and negative zero tied between distinct observations.
    SignedZeros,

    /// Several tied groups in unsorted observation order.
    Ties,

    /// Two observations distinguishing sample from population standardization.
    TwoPoint,
}

impl Dataset {
    const REFERENCES: &'static str = include_str!("../../data/r/transform.csv");

    /// Loads the fixture and its corresponding R transform references.
    ///
    /// # Panics
    ///
    /// Panics if the bundled fixture or reference data is malformed, or its
    /// observation count or indices do not match the fixture.
    #[must_use]
    pub fn load(self) -> Reference {
        Parser::new(self).parse()
    }

    const fn case_name(self) -> &'static str {
        match self {
            Self::AsymmetricTail => "asymmetric-tail",
            Self::Baseline => "baseline",
            Self::SignedZeros => "signed-zeros",
            Self::Ties => "ties",
            Self::TwoPoint => "two-point",
        }
    }

    const fn fixture(self) -> &'static str {
        match self {
            Self::AsymmetricTail => {
                include_str!("../../data/fixtures/r/transform/asymmetric-tail.csv")
            }
            Self::Baseline => {
                include_str!("../../data/fixtures/r/transform/baseline.csv")
            }
            Self::SignedZeros => {
                include_str!("../../data/fixtures/r/transform/signed-zeros.csv")
            }
            Self::Ties => {
                include_str!("../../data/fixtures/r/transform/ties.csv")
            }
            Self::TwoPoint => {
                include_str!("../../data/fixtures/r/transform/two-point.csv")
            }
        }
    }
}

/// One transform fixture and its R reference values.
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

    /// Observed values in original fixture order.
    #[must_use]
    pub fn observations(&self) -> &[f64] {
        &self.observations
    }

    /// Reference values aligned with [`Self::observations`].
    #[must_use]
    pub fn statistics(&self) -> &[Statistics] {
        &self.statistics
    }
}

/// Three statistical transformations of one observation, computed by R.
#[derive(Debug, Clone, Copy)]
pub struct Statistics {
    demeaned: f64,
    z_score: f64,
    rank: f64,
}

impl Statistics {
    /// Observation minus the sample mean.
    #[must_use]
    pub const fn demeaned(&self) -> f64 {
        self.demeaned
    }

    /// Centered observation divided by the sample standard deviation.
    #[must_use]
    pub const fn z_score(&self) -> f64 {
        self.z_score
    }

    /// One-based rank, with equal values receiving their average rank.
    #[must_use]
    pub const fn rank(&self) -> f64 {
        self.rank
    }
}

struct Parser {
    dataset: Dataset,
}

impl Parser {
    const REFERENCE_HEADER: &'static str = "case,observation,demeaned,z_score,rank";

    const fn new(dataset: Dataset) -> Self {
        Self { dataset }
    }

    fn parse(self) -> Reference {
        let observations = parse_value_fixture(self.dataset.fixture(), "transform fixture");
        let statistics = self.parse_statistics();

        assert!(
            observations.len() >= 2,
            "transform fixture should contain at least two observations"
        );

        assert_eq!(
            statistics.len(),
            observations.len(),
            "transform reference count should match the fixture"
        );

        Reference { dataset: self.dataset, observations, statistics }
    }

    fn parse_statistics(&self) -> Vec<Statistics> {
        let case = self.dataset.case_name();
        let mut reference = Dataset::REFERENCES.lines();

        let header = reference
            .next()
            .expect("transform reference CSV should contain a header");

        assert_eq!(
            header,
            Self::REFERENCE_HEADER,
            "transform reference CSV header should match the expected schema"
        );

        let mut statistics = Vec::new();

        for line in reference {
            let fields: Vec<_> = line.split(',').collect();

            assert_eq!(
                fields.len(),
                5,
                "transform reference rows should contain exactly five fields"
            );

            if fields[0] != case {
                continue;
            }

            let observation = parse_usize(fields[1], "reference observation");

            assert_eq!(
                observation,
                statistics.len(),
                "transform reference observations should be in fixture order"
            );

            let demeaned = parse_finite(fields[2], "reference demeaned value");
            let z_score = parse_finite(fields[3], "reference z-score");
            let rank = parse_finite(fields[4], "reference rank");

            assert!(rank >= 1.0, "reference ranks should start at one");

            statistics.push(Statistics { demeaned, z_score, rank });
        }

        statistics
    }
}

#[cfg(test)]
mod tests {
    use super::Dataset;

    #[test]
    fn all_datasets_load_with_aligned_observations() {
        let datasets = [
            (Dataset::AsymmetricTail, 7),
            (Dataset::Baseline, 8),
            (Dataset::SignedZeros, 4),
            (Dataset::Ties, 5),
            (Dataset::TwoPoint, 2),
        ];

        for (dataset, n) in datasets {
            let reference = dataset.load();

            assert_eq!(reference.dataset(), dataset);
            assert_eq!(reference.observations().len(), n);
            assert_eq!(reference.statistics().len(), n);
        }
    }
}
