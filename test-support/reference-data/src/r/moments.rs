//! R conformance fixtures and reference values for moment statistics.
//!
//! Select a [`Dataset`] and call [`Dataset::load`] to obtain its observations
//! and corresponding [`Statistics`].
//!
//! References cover the arithmetic mean, variance with denominators `n - 1`,
//! `n`, and `n - 2`, and Joanes-Gill Type 1/2/3 skewness and kurtosis. Excess
//! kurtosis is computed by `e1071`; Pearson kurtosis adds three to each
//! corresponding estimate.
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
    population_variance: f64,
    variance_ddof_2: f64,
    skewness_type_1: f64,
    skewness_type_2: f64,
    skewness_type_3: f64,
    excess_kurtosis_type_1: f64,
    excess_kurtosis_type_2: f64,
    excess_kurtosis_type_3: f64,
    pearson_kurtosis_type_1: f64,
    pearson_kurtosis_type_2: f64,
    pearson_kurtosis_type_3: f64,
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

    /// Population variance with denominator `n`.
    #[must_use]
    pub const fn population_variance(&self) -> f64 {
        self.population_variance
    }

    /// Variance with denominator `n - 2`, providing a custom-`ddof` reference.
    #[must_use]
    pub const fn variance_ddof_2(&self) -> f64 {
        self.variance_ddof_2
    }

    /// Moment coefficient of skewness using Joanes-Gill Type 1.
    #[must_use]
    pub const fn skewness_type_1(&self) -> f64 {
        self.skewness_type_1
    }

    /// Adjusted skewness using Joanes-Gill Type 2.
    #[must_use]
    pub const fn skewness_type_2(&self) -> f64 {
        self.skewness_type_2
    }

    /// Skewness standardized by the sample standard deviation, using
    /// Joanes-Gill Type 3.
    #[must_use]
    pub const fn skewness_type_3(&self) -> f64 {
        self.skewness_type_3
    }

    /// Moment coefficient of excess kurtosis using Joanes-Gill Type 1.
    #[must_use]
    pub const fn excess_kurtosis_type_1(&self) -> f64 {
        self.excess_kurtosis_type_1
    }

    /// Adjusted excess kurtosis using Joanes-Gill Type 2.
    #[must_use]
    pub const fn excess_kurtosis_type_2(&self) -> f64 {
        self.excess_kurtosis_type_2
    }

    /// Excess kurtosis standardized by the sample standard deviation, using
    /// Joanes-Gill Type 3.
    #[must_use]
    pub const fn excess_kurtosis_type_3(&self) -> f64 {
        self.excess_kurtosis_type_3
    }

    /// Pearson kurtosis using Joanes-Gill Type 1.
    #[must_use]
    pub const fn pearson_kurtosis_type_1(&self) -> f64 {
        self.pearson_kurtosis_type_1
    }

    /// Adjusted Pearson kurtosis using Joanes-Gill Type 2.
    #[must_use]
    pub const fn pearson_kurtosis_type_2(&self) -> f64 {
        self.pearson_kurtosis_type_2
    }

    /// Pearson kurtosis standardized by the sample standard deviation, using
    /// Joanes-Gill Type 3.
    #[must_use]
    pub const fn pearson_kurtosis_type_3(&self) -> f64 {
        self.pearson_kurtosis_type_3
    }
}

struct Parser {
    dataset: Dataset,
}

impl Parser {
    const REFERENCE_HEADER: &'static str = concat!(
        "case,n,mean,variance,population_variance,variance_ddof_2,",
        "skewness_type_1,skewness_type_2,skewness_type_3,",
        "excess_kurtosis_type_1,excess_kurtosis_type_2,excess_kurtosis_type_3,",
        "pearson_kurtosis_type_1,pearson_kurtosis_type_2,pearson_kurtosis_type_3",
    );

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
            15,
            "moments reference row should contain exactly fifteen fields"
        );

        let n = parse_usize(fields[1], "reference n");
        let mean = parse_finite(fields[2], "reference mean");
        let variance = parse_finite(fields[3], "reference variance");
        let population_variance = parse_finite(fields[4], "reference population_variance");
        let variance_ddof_2 = parse_finite(fields[5], "reference variance_ddof_2");
        let skewness_type_1 = parse_finite(fields[6], "reference skewness_type_1");
        let skewness_type_2 = parse_finite(fields[7], "reference skewness_type_2");
        let skewness_type_3 = parse_finite(fields[8], "reference skewness_type_3");
        let excess_kurtosis_type_1 = parse_finite(fields[9], "reference excess_kurtosis_type_1");
        let excess_kurtosis_type_2 = parse_finite(fields[10], "reference excess_kurtosis_type_2");
        let excess_kurtosis_type_3 = parse_finite(fields[11], "reference excess_kurtosis_type_3");
        let pearson_kurtosis_type_1 = parse_finite(fields[12], "reference pearson_kurtosis_type_1");
        let pearson_kurtosis_type_2 = parse_finite(fields[13], "reference pearson_kurtosis_type_2");
        let pearson_kurtosis_type_3 = parse_finite(fields[14], "reference pearson_kurtosis_type_3");

        Statistics {
            n,
            mean,
            variance,
            population_variance,
            variance_ddof_2,
            skewness_type_1,
            skewness_type_2,
            skewness_type_3,
            excess_kurtosis_type_1,
            excess_kurtosis_type_2,
            excess_kurtosis_type_3,
            pearson_kurtosis_type_1,
            pearson_kurtosis_type_2,
            pearson_kurtosis_type_3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Dataset;

    #[test]
    fn variance_references_match_hand_calculated_sample() {
        let reference = Dataset::MinimumShapeSample.load();
        let statistics = reference.statistics();

        assert_eq!(reference.observations(), &[1.0, 2.0, 3.0, 6.0]);
        assert_eq!(statistics.n(), 4);

        // Mean = 3, centered sum of squares = 4 + 1 + 0 + 9 = 14.
        // Dividing by n = 4 or n - 2 = 2 gives exact binary results.
        assert_eq!(statistics.population_variance(), 3.5);
        assert_eq!(statistics.variance_ddof_2(), 7.0);
    }

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
