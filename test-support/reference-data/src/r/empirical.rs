//! R conformance fixtures and reference values for empirical distributions.
//!
//! Select a [`Dataset`] and call [`Dataset::load`] to obtain its observations
//! and corresponding [`Statistics`]. These contain ECDF and survival
//! evaluations, together with Hyndman–Fan Type 1 inverse ECDF quantiles.
//!
//! Fixtures and reference values are embedded; loading them does not run R or
//! require network access.

/// A bundled R empirical distribution dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dataset {
    /// Ordinary decimal observations.
    Baseline,

    /// Identical observations.
    Constant,

    /// A single observation.
    Singleton,

    /// Repeated integer observations.
    Ties,

    /// Two distinct observations.
    TwoPoint,
}

impl Dataset {
    const EVALUATIONS: &'static str = include_str!("../../data/r/empirical.csv");
    const QUANTILES: &'static str = include_str!("../../data/r/empirical-quantiles.csv");

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
            Self::Baseline => "baseline",
            Self::Constant => "constant",
            Self::Singleton => "singleton",
            Self::Ties => "ties",
            Self::TwoPoint => "two-point",
        }
    }

    const fn fixture(self) -> &'static str {
        match self {
            Self::Baseline => {
                include_str!("../../data/fixtures/r/empirical/baseline.csv")
            }
            Self::Constant => {
                include_str!("../../data/fixtures/r/empirical/constant.csv")
            }
            Self::Singleton => {
                include_str!("../../data/fixtures/r/empirical/singleton.csv")
            }
            Self::Ties => {
                include_str!("../../data/fixtures/r/empirical/ties.csv")
            }
            Self::TwoPoint => {
                include_str!("../../data/fixtures/r/empirical/two-point.csv")
            }
        }
    }
}

/// One empirical distribution fixture and its R reference statistics.
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

/// Reference evaluations and inverse ECDF quantiles computed by R.
#[derive(Debug)]
pub struct Statistics {
    evaluations: Vec<Evaluation>,
    quantiles: Vec<Quantile>,
}

impl Statistics {
    /// Reference ECDF and survival values in ascending query order.
    #[must_use]
    pub fn evaluations(&self) -> &[Evaluation] {
        &self.evaluations
    }

    /// Reference Hyndman–Fan Type 1 inverse ECDF values in ascending
    /// probability order.
    #[must_use]
    pub fn quantiles(&self) -> &[Quantile] {
        &self.quantiles
    }
}

/// One query and its empirical cumulative and survival probabilities.
#[derive(Debug, Clone, Copy)]
pub struct Evaluation {
    query: f64,
    cdf: f64,
    sf: f64,
}

impl Evaluation {
    /// Finite value at which the empirical distribution is evaluated.
    #[must_use]
    pub const fn query(&self) -> f64 {
        self.query
    }

    /// Fraction of observations less than or equal to the query, in `[0, 1]`.
    #[must_use]
    pub const fn cdf(&self) -> f64 {
        self.cdf
    }

    /// Fraction of observations strictly greater than the query, in `[0, 1]`.
    #[must_use]
    pub const fn sf(&self) -> f64 {
        self.sf
    }
}

/// One probability and its Hyndman–Fan Type 1 inverse ECDF quantile computed by
/// R.
#[derive(Debug, Clone, Copy)]
pub struct Quantile {
    probability: f64,
    value: f64,
}

impl Quantile {
    /// Probability in the closed interval `[0, 1]`.
    #[must_use]
    pub const fn probability(&self) -> f64 {
        self.probability
    }

    /// Reference inverse ECDF value, selected without interpolation.
    ///
    /// Probability zero returns the sample minimum; probability one returns
    /// the maximum.
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
    const EVALUATION_HEADER: &'static str = "case,query,cdf,sf";
    const QUANTILE_HEADER: &'static str = "case,probability,value";

    const fn new(dataset: Dataset) -> Self {
        Self { dataset }
    }

    fn parse(self) -> Reference {
        let observations = self.parse_fixture();
        let evaluations = self.parse_evaluations();
        let quantiles = self.parse_quantiles();
        let statistics = Statistics { evaluations, quantiles };

        Reference { dataset: self.dataset, observations, statistics }
    }

    fn parse_fixture(&self) -> Vec<f64> {
        let mut fixture = self.dataset.fixture().lines();

        let header = fixture
            .next()
            .expect("empirical fixture should contain a header");

        assert_eq!(
            header,
            Self::FIXTURE_HEADER,
            "empirical fixture CSV header should match the expected schema"
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
            "empirical fixture should contain observations"
        );

        observations
    }

    fn parse_evaluations(&self) -> Vec<Evaluation> {
        let case = self.dataset.case_name();
        let mut reference = Dataset::EVALUATIONS.lines();

        let header = reference
            .next()
            .expect("empirical evaluation CSV should contain a header");

        assert_eq!(
            header,
            Self::EVALUATION_HEADER,
            "empirical evaluation CSV header should match the expected schema"
        );

        let mut evaluations = Vec::new();

        for line in reference {
            let mut fields = line.split(',');

            let row_case = fields
                .next()
                .expect("empirical evaluation row should contain a case");

            if row_case != case {
                continue;
            }

            let query: f64 = fields
                .next()
                .expect("empirical evaluation row should contain a query")
                .parse()
                .expect("query should be numerical");

            let cdf: f64 = fields
                .next()
                .expect("empirical evaluation row should contain an ECDF value")
                .parse()
                .expect("ECDF value should be numerical");

            let sf: f64 = fields
                .next()
                .expect("empirical evaluation row should contain a survival value")
                .parse()
                .expect("survival value should be numerical");

            assert!(
                fields.next().is_none(),
                "empirical evaluation row should contain exactly four fields"
            );

            assert!(query.is_finite(), "reference query should be finite");

            assert!(
                is_probability(cdf),
                "reference ECDF probability should be finite and in [0, 1]"
            );

            assert!(
                is_probability(sf),
                "reference survival probability should be finite and in [0, 1]"
            );

            evaluations.push(Evaluation { query, cdf, sf });
        }

        assert!(
            !evaluations.is_empty(),
            "empirical evaluation data should contain the requested case"
        );

        assert!(
            evaluations
                .windows(2)
                .all(|pair| pair[0].query < pair[1].query),
            "reference queries should be unique and in ascending order"
        );

        assert!(
            evaluations
                .windows(2)
                .all(|pair| pair[0].cdf <= pair[1].cdf),
            "reference ECDF probabilities should be nondecreasing"
        );

        assert!(
            evaluations.windows(2).all(|pair| pair[0].sf >= pair[1].sf),
            "reference survival probabilities should be nonincreasing"
        );

        evaluations
    }

    fn parse_quantiles(&self) -> Vec<Quantile> {
        let case = self.dataset.case_name();
        let mut reference = Dataset::QUANTILES.lines();

        let header = reference
            .next()
            .expect("empirical quantile CSV should contain a header");

        assert_eq!(
            header,
            Self::QUANTILE_HEADER,
            "empirical quantile CSV header should match the expected schema"
        );

        let mut quantiles = Vec::new();

        for line in reference {
            let mut fields = line.split(',');

            let row_case = fields
                .next()
                .expect("empirical quantile row should contain a case");

            if row_case != case {
                continue;
            }

            let probability: f64 = fields
                .next()
                .expect("empirical quantile row should contain a probability")
                .parse()
                .expect("probability should be numerical");

            let value: f64 = fields
                .next()
                .expect("empirical quantile row should contain a value")
                .parse()
                .expect("quantile value should be numerical");

            assert!(
                fields.next().is_none(),
                "empirical quantile row should contain exactly three fields"
            );

            assert!(
                is_probability(probability),
                "reference probability should be finite and in [0, 1]"
            );

            assert!(value.is_finite(), "reference quantile should be finite");

            quantiles.push(Quantile { probability, value });
        }

        assert!(
            !quantiles.is_empty(),
            "empirical quantile data should contain the requested case"
        );

        assert!(
            quantiles
                .windows(2)
                .all(|pair| pair[0].probability < pair[1].probability),
            "reference probabilities should be unique and in ascending order"
        );

        assert!(
            quantiles
                .windows(2)
                .all(|pair| pair[0].value <= pair[1].value),
            "reference quantiles should be nondecreasing"
        );

        quantiles
    }
}

fn is_probability(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

#[cfg(test)]
mod tests {
    use super::Dataset;

    #[test]
    fn all_datasets_load() {
        let datasets = [
            Dataset::Baseline,
            Dataset::Constant,
            Dataset::Singleton,
            Dataset::Ties,
            Dataset::TwoPoint,
        ];

        for dataset in datasets {
            let reference = dataset.load();
            let statistics = reference.statistics();

            assert_eq!(reference.dataset(), dataset);
            assert_ne!(reference.observations(), []);
            assert!(!statistics.evaluations().is_empty());
            assert!(!statistics.quantiles().is_empty());
        }
    }
}
