//! [NIST univariate Statistical Reference Datasets] with certified reference
//! statistics.
//!
//! Select a [`Dataset`] and call [`Dataset::load`] to obtain its observations
//! and published statistics. References include the observation count, sample
//! mean, sample standard deviation, and lag-one autocorrelation.
//!
//! Data is embedded in the crate. Loading requires no network access and
//! checks that the parsed observation count matches the published count.
//!
//! [NIST univariate Statistical Reference Datasets]: https://www.itl.nist.gov/div898/strd/univ/homepage.html

/// A bundled NIST univariate Statistical Reference Dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dataset {
    /// `Lew` dataset.
    Lew,

    /// `Lottery` dataset.
    Lottery,

    /// `Mavro` dataset.
    Mavro,

    /// `Michelso` dataset.
    Michelso,

    /// `NumAcc1` dataset.
    NumAcc1,

    /// `NumAcc2` dataset.
    NumAcc2,

    /// `NumAcc3` dataset.
    NumAcc3,

    /// `NumAcc4` dataset.
    NumAcc4,

    /// `PiDigits` dataset.
    PiDigits,
}

impl Dataset {
    /// Loads a bundled dataset and its certified reference statistics.
    ///
    /// # Panics
    ///
    /// Panics if the bundled NIST data is malformed or contains a different
    /// number of observations than its certified count.
    #[must_use]
    pub fn load(self) -> Reference {
        Parser::new(self).parse()
    }

    const fn input(self) -> &'static str {
        match self {
            Self::Lew => include_str!("../../data/nist/univariate/Lew.dat"),
            Self::Lottery => include_str!("../../data/nist/univariate/Lottery.dat"),
            Self::Mavro => include_str!("../../data/nist/univariate/Mavro.dat"),
            Self::Michelso => include_str!("../../data/nist/univariate/Michelso.dat"),
            Self::NumAcc1 => include_str!("../../data/nist/univariate/NumAcc1.dat"),
            Self::NumAcc2 => include_str!("../../data/nist/univariate/NumAcc2.dat"),
            Self::NumAcc3 => include_str!("../../data/nist/univariate/NumAcc3.dat"),
            Self::NumAcc4 => include_str!("../../data/nist/univariate/NumAcc4.dat"),
            Self::PiDigits => include_str!("../../data/nist/univariate/PiDigits.dat"),
        }
    }
}

/// A bundled NIST univariate dataset and its certified reference statistics.
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

    /// NIST reference statistics.
    #[must_use]
    pub const fn statistics(&self) -> &Statistics {
        &self.statistics
    }
}

/// NIST univariate reference statistics.
#[derive(Debug, Clone, Copy)]
pub struct Statistics {
    n: usize,
    sample_mean: f64,
    sample_standard_deviation: f64,
    sample_lag1_autocorrelation: f64,
}

impl Statistics {
    /// Number of observations in the dataset.
    #[must_use]
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Certified sample mean.
    #[must_use]
    pub const fn sample_mean(&self) -> f64 {
        self.sample_mean
    }

    /// Certified sample standard deviation using `n - 1` degrees of freedom.
    #[must_use]
    pub const fn sample_standard_deviation(&self) -> f64 {
        self.sample_standard_deviation
    }

    /// Certified lag-1 sample autocorrelation coefficient.
    #[must_use]
    pub const fn sample_lag1_autocorrelation(&self) -> f64 {
        self.sample_lag1_autocorrelation
    }
}

struct Parser {
    dataset: Dataset,
}

impl Parser {
    const OBSERVATION_COUNT: &'static str = "Number of Observations";
    const SAMPLE_MEAN: &'static str = "Sample Mean";
    const SAMPLE_STANDARD_DEVIATION: &'static str = "Sample Standard Deviation";
    const SAMPLE_AUTOCORRELATION: &'static str = "Sample Autocorrelation Coefficient";

    const fn new(dataset: Dataset) -> Self {
        Self { dataset }
    }

    fn parse(self) -> Reference {
        let observations = self.observations();
        let statistics = Statistics {
            n: self.field(Self::OBSERVATION_COUNT),
            sample_mean: self.field(Self::SAMPLE_MEAN),
            sample_standard_deviation: self.field(Self::SAMPLE_STANDARD_DEVIATION),
            sample_lag1_autocorrelation: self.field(Self::SAMPLE_AUTOCORRELATION),
        };

        assert_eq!(
            observations.len(),
            statistics.n,
            "observation length mismatch"
        );

        Reference { dataset: self.dataset, observations, statistics }
    }

    fn observations(&self) -> Vec<f64> {
        let mut lines = self.dataset.input().lines();

        lines
            .find(|line| line.starts_with("Data: Y"))
            .expect("NIST file should contain Data: Y");

        let sep = lines
            .next()
            .expect("NIST file should contain data separator");

        assert!(
            sep.chars().all(|c| c == '-'),
            "separator should be correctly formed"
        );

        lines
            .take_while(|line| !line.trim().is_empty())
            .map(|line| {
                line.trim()
                    .parse()
                    .expect("NIST observations must be numerical")
            })
            .collect()
    }

    fn field<T>(&self, label: &str) -> T
    where
        T: std::str::FromStr,
        T::Err: std::fmt::Debug,
    {
        self.dataset
            .input()
            .lines()
            .find_map(|line| {
                let (key, value) = line.rsplit_once(':')?;

                key.trim_start()
                    .starts_with(label)
                    .then_some(value)?
                    .split_whitespace()
                    .next()
            })
            .unwrap_or_else(|| panic!("NIST file is missing {label}"))
            .parse()
            .unwrap_or_else(|err| panic!("failed to parse {label}: {err:?}"))
    }
}
