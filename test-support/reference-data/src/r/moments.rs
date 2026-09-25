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
    const FIXTURE_HEADER: &'static str = "value";
    const REFERENCE_HEADER: &'static str = "case,n,mean,variance,skewness,excess_kurtosis";

    const fn new(dataset: Dataset) -> Self {
        Self { dataset }
    }

    fn parse(self) -> Reference {
        let observations = self.parse_fixture();
        let statistics = self.parse_statistics();

        assert_eq!(
            observations.len(),
            statistics.n,
            "fixture observation count should match the R reference count"
        );

        Reference { dataset: self.dataset, observations, statistics }
    }

    fn parse_fixture(&self) -> Vec<f64> {
        let mut fixture = self.dataset.fixture().lines();

        let header = fixture
            .next()
            .expect("moments fixture should contain a header");

        assert_eq!(
            header,
            Self::FIXTURE_HEADER,
            "moments fixture CSV header should match the expected schema "
        );

        let observations: Vec<f64> = fixture
            .map(|line| {
                let value: f64 = line
                    .trim()
                    .parse()
                    .expect("fixture values should be numerical");

                assert!(value.is_finite(), "fixture values should be finite");

                value
            })
            .collect();

        assert!(
            !observations.is_empty(),
            "moments fixture should contain observations"
        );

        observations
    }

    fn parse_statistics(&self) -> Statistics {
        let case = self.dataset.case_name();

        let mut reference = Dataset::REFERENCES.lines();

        let header = reference
            .next()
            .expect("moments reference csv should contain a header");

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

        let mut fields = row.split(',');

        let row_case = fields
            .next()
            .expect("moments reference row should contain a case");

        assert_eq!(
            row_case, case,
            "moments reference row case should match the requested case"
        );

        let n = fields
            .next()
            .expect("moments reference row should contain n")
            .parse()
            .expect("n should be numerical");

        let mean = fields
            .next()
            .expect("moments reference row should contain a mean")
            .parse()
            .expect("mean should be numerical");

        let variance = fields
            .next()
            .expect("moments reference row should contain a variance")
            .parse()
            .expect("variance should be numerical");

        let skewness = fields
            .next()
            .expect("moments reference row should contain a skewness")
            .parse()
            .expect("skewness should be numerical");

        let excess_kurtosis = fields
            .next()
            .expect("moments reference row should contain an excess kurtosis")
            .parse()
            .expect("excess kurtosis should numerical");

        assert!(
            fields.next().is_none(),
            "moments reference row should contain exactly six fields"
        );

        Statistics { n, mean, variance, skewness, excess_kurtosis }
    }
}
