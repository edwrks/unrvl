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

/// A bundled R quantiles dataset.
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

/// One quantiles fixture and its R reference statistics.
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
    const FIXTURE_HEADER: &'static str = "value";
    const REFERENCE_HEADER: &'static str = "case,probability,value";

    const fn new(dataset: Dataset) -> Self {
        Self { dataset }
    }

    fn parse(self) -> Reference {
        let observations = self.parse_fixture();
        let statistics = self.parse_statistics();

        Reference { dataset: self.dataset, observations, statistics }
    }

    fn parse_fixture(&self) -> Vec<f64> {
        let mut fixture = self.dataset.fixture().lines();

        let header = fixture
            .next()
            .expect("quantiles fixture should contain a header");

        assert_eq!(
            header,
            Self::FIXTURE_HEADER,
            "quantiles fixture CSV header should match the expected schema"
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
            "quantiles fixture should contain observations"
        );

        observations
    }

    fn parse_statistics(&self) -> Vec<Statistics> {
        let case = self.dataset.case_name();
        let mut reference = Dataset::REFERENCES.lines();

        let header = reference
            .next()
            .expect("quantiles reference CSV should contain a header");

        assert_eq!(
            header,
            Self::REFERENCE_HEADER,
            "quantiles reference CSV header should match the expected schema"
        );

        let mut statistics = Vec::new();

        for line in reference {
            let mut fields = line.split(',');

            let row_case = fields
                .next()
                .expect("quantiles reference row should contain a case");

            if row_case != case {
                continue;
            }

            let probability: f64 = fields
                .next()
                .expect("quantiles reference row should contain a probability")
                .parse()
                .expect("probability should be numerical");

            let value: f64 = fields
                .next()
                .expect("quantiles reference row should contain a value")
                .parse()
                .expect("quantile value should be numerical");

            assert!(
                fields.next().is_none(),
                "quantiles reference row should contain exactly three fields"
            );

            assert!(
                probability.is_finite() && (0.0..=1.0).contains(&probability),
                "reference probability should be finite and in [0, 1]"
            );

            assert!(value.is_finite(), "reference quantile should be finite");

            statistics.push(Statistics { probability, value });
        }

        assert!(
            !statistics.is_empty(),
            "quantiles reference data should contain the requested case"
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
