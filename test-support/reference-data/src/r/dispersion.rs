//! R conformance fixtures and reference values for dispersion statistics.
//!
//! Select a [`Dataset`] and call [`Dataset::load`] to obtain its observations
//! and corresponding [`Statistics`]. Fixtures and reference values are
//! embedded; loading them does not run R or require network access.

use super::parsing::{parse_finite, parse_usize, parse_value_fixture};

/// A bundled R dispersion dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dataset {
    /// Uneven tails.
    AsymmetricTail,

    /// Ordinary decimal observations.
    Baseline,

    /// Four observations with interpolated quartiles.
    MinimumShapeSample,

    /// Repeated integer observations and quantile boundaries.
    Ties,
}

impl Dataset {
    const REFERENCES: &'static str = include_str!("../../data/r/dispersion.csv");

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
                include_str!("../../data/fixtures/r/dispersion/asymmetric-tail.csv")
            }
            Self::Baseline => {
                include_str!("../../data/fixtures/r/dispersion/baseline.csv")
            }
            Self::MinimumShapeSample => {
                include_str!("../../data/fixtures/r/dispersion/minimum-shape-sample.csv")
            }
            Self::Ties => {
                include_str!("../../data/fixtures/r/dispersion/ties.csv")
            }
        }
    }
}

/// One dispersion fixture and its R reference statistics.
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

/// Reference dispersion statistics computed by R.
#[derive(Debug, Clone, Copy)]
pub struct Statistics {
    n: usize,
    range: f64,
    iqr: f64,
    mad: f64,
    std_dev: f64,
    cv: f64,
}

impl Statistics {
    /// Number of observations in the dataset.
    #[must_use]
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Difference between the maximum and minimum observations.
    #[must_use]
    pub const fn range(&self) -> f64 {
        self.range
    }

    /// Interquartile range using Type 7 quartiles.
    #[must_use]
    pub const fn iqr(&self) -> f64 {
        self.iqr
    }

    /// Median absolute deviation from the sample median, without scaling.
    #[must_use]
    pub const fn mad(&self) -> f64 {
        self.mad
    }

    /// Sample standard deviation using the variance denominator `n - 1`.
    #[must_use]
    pub const fn std_dev(&self) -> f64 {
        self.std_dev
    }

    /// Signed coefficient of variation: sample standard deviation divided by
    /// the arithmetic mean, expressed as a ratio.
    #[must_use]
    pub const fn cv(&self) -> f64 {
        self.cv
    }
}

struct Parser {
    dataset: Dataset,
}

impl Parser {
    const REFERENCE_HEADER: &'static str = "case,n,range,iqr,mad,std_dev,cv";

    const fn new(dataset: Dataset) -> Self {
        Self { dataset }
    }

    fn parse(self) -> Reference {
        let observations = parse_value_fixture(self.dataset.fixture(), "dispersion fixture");
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
            .expect("dispersion reference CSV should contain a header");

        assert_eq!(
            header,
            Self::REFERENCE_HEADER,
            "dispersion reference CSV header should match the expected schema"
        );

        let row = reference
            .find(|line| {
                line.split_once(',')
                    .is_some_and(|(row_case, _)| row_case == case)
            })
            .expect("dispersion reference data should contain the requested case");

        let fields: Vec<_> = row.split(',').collect();
        assert_eq!(
            fields.len(),
            7,
            "dispersion reference row should contain exactly seven fields"
        );

        let n = parse_usize(fields[1], "reference n");
        let range = parse_finite(fields[2], "reference range");
        let iqr = parse_finite(fields[3], "reference iqr");
        let mad = parse_finite(fields[4], "reference mad");
        let std_dev = parse_finite(fields[5], "reference std_dev");
        let cv = parse_finite(fields[6], "reference cv");

        Statistics { n, range, iqr, mad, std_dev, cv }
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
            assert_eq!(statistics.n(), reference.observations().len());
        }
    }
}
