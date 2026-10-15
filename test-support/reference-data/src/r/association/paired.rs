//! Paired observations and R association references.
//!
//! Select a [`Dataset`] and call [`Dataset::load`] to obtain two aligned
//! observation vectors and their [`Statistics`]. All observations and reference
//! values are finite. Covariance uses the sample denominator `n - 1`;
//! correlations are Pearson, Spearman with average ranks, and Kendall tau-b.
//!
//! ```
//! use reference_data::r::association::paired::Dataset;
//!
//! let reference = Dataset::Baseline.load();
//! assert_eq!(reference.x().len(), reference.y().len());
//! assert_eq!(reference.statistics().n(), reference.x().len());
//! ```

use crate::r::parsing::{parse_correlation, parse_finite, parse_usize};

/// A bundled R paired association dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dataset {
    /// Ordinary decimal observations, including ties.
    Baseline,

    /// Mixed concordant and discordant pairs without ties.
    MixedOrder,

    /// A strictly increasing nonlinear relationship.
    MonotonicNonlinear,

    /// Perfect negative linear association.
    PerfectNegative,

    /// Perfect positive linear association.
    PerfectPositive,

    /// Unequal ties in both variables, including repeated pairs.
    TiesBothVariables,

    /// Ties in only the first variable.
    TiesOneVariable,

    /// Varying observations with zero covariance.
    ZeroCovariance,
}

impl Dataset {
    const REFERENCES: &'static str = include_str!("../../../data/r/association.csv");

    /// Loads the paired fixture and its R reference statistics.
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
            Self::MixedOrder => "mixed-order",
            Self::MonotonicNonlinear => "monotonic-nonlinear",
            Self::PerfectNegative => "perfect-negative",
            Self::PerfectPositive => "perfect-positive",
            Self::TiesBothVariables => "ties-both-variables",
            Self::TiesOneVariable => "ties-one-variable",
            Self::ZeroCovariance => "zero-covariance",
        }
    }

    const fn fixture(self) -> &'static str {
        match self {
            Self::Baseline => {
                include_str!("../../../data/fixtures/r/association/baseline.csv")
            }
            Self::MixedOrder => {
                include_str!("../../../data/fixtures/r/association/mixed-order.csv")
            }
            Self::MonotonicNonlinear => {
                include_str!("../../../data/fixtures/r/association/monotonic-nonlinear.csv")
            }
            Self::PerfectNegative => {
                include_str!("../../../data/fixtures/r/association/perfect-negative.csv")
            }
            Self::PerfectPositive => {
                include_str!("../../../data/fixtures/r/association/perfect-positive.csv")
            }
            Self::TiesBothVariables => {
                include_str!("../../../data/fixtures/r/association/ties-both-variables.csv")
            }
            Self::TiesOneVariable => {
                include_str!("../../../data/fixtures/r/association/ties-one-variable.csv")
            }
            Self::ZeroCovariance => {
                include_str!("../../../data/fixtures/r/association/zero-covariance.csv")
            }
        }
    }
}

/// One paired fixture and its R reference statistics.
#[derive(Debug)]
pub struct Reference {
    dataset: Dataset,
    x: Vec<f64>,
    y: Vec<f64>,
    statistics: Statistics,
}

impl Reference {
    /// The bundled dataset this reference was loaded from.
    #[must_use]
    pub const fn dataset(&self) -> Dataset {
        self.dataset
    }

    /// First variable, in fixture observation order and aligned with
    /// [`Self::y`].
    #[must_use]
    pub fn x(&self) -> &[f64] {
        &self.x
    }

    /// Second variable, in fixture observation order and aligned with
    /// [`Self::x`].
    #[must_use]
    pub fn y(&self) -> &[f64] {
        &self.y
    }

    /// Reference statistics computed by R.
    #[must_use]
    pub const fn statistics(&self) -> &Statistics {
        &self.statistics
    }
}

/// Scalar association statistics computed by R.
#[derive(Debug, Clone, Copy)]
pub struct Statistics {
    n: usize,
    covariance: f64,
    pearson: f64,
    spearman: f64,
    kendall: f64,
}

impl Statistics {
    /// Number of paired observations.
    #[must_use]
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Sample covariance with denominator `n - 1`.
    #[must_use]
    pub const fn covariance(&self) -> f64 {
        self.covariance
    }

    /// Pearson correlation in `[-1, 1]`.
    #[must_use]
    pub const fn pearson(&self) -> f64 {
        self.pearson
    }

    /// Spearman correlation using average ranks for ties, in `[-1, 1]`.
    #[must_use]
    pub const fn spearman(&self) -> f64 {
        self.spearman
    }

    /// Kendall tau-b accounting for ties, in `[-1, 1]`.
    #[must_use]
    pub const fn kendall(&self) -> f64 {
        self.kendall
    }
}

struct Parser {
    dataset: Dataset,
}

impl Parser {
    const FIXTURE_HEADER: &'static str = "x,y";
    const REFERENCE_HEADER: &'static str = "case,n,covariance,pearson,spearman,kendall";

    const fn new(dataset: Dataset) -> Self {
        Self { dataset }
    }

    fn parse(self) -> Reference {
        let (x, y) = self.parse_fixture();
        let statistics = self.parse_statistics();

        assert_eq!(
            statistics.n,
            x.len(),
            "fixture observation count should match the R reference count"
        );

        Reference { dataset: self.dataset, x, y, statistics }
    }

    fn parse_fixture(&self) -> (Vec<f64>, Vec<f64>) {
        let mut lines = self.dataset.fixture().lines();

        let header = lines
            .next()
            .expect("paired fixture should contain a header");

        assert_eq!(
            header,
            Self::FIXTURE_HEADER,
            "paired fixture CSV header should match the expected schema"
        );

        let mut x = Vec::new();
        let mut y = Vec::new();

        for line in lines {
            let fields: Vec<_> = line.split(',').collect();

            assert_eq!(
                fields.len(),
                2,
                "paired fixture rows should contain exactly two fields"
            );

            x.push(parse_finite(fields[0], "fixture x"));
            y.push(parse_finite(fields[1], "fixture y"));
        }

        assert!(
            x.len() >= 2,
            "paired fixture should contain at least two observations"
        );

        (x, y)
    }

    fn parse_statistics(&self) -> Statistics {
        let case = self.dataset.case_name();

        let mut reference = Dataset::REFERENCES.lines();

        let header = reference
            .next()
            .expect("paired reference CSV should contain a header");

        assert_eq!(
            header,
            Self::REFERENCE_HEADER,
            "paired reference CSV header should match the expected schema"
        );

        let row = reference
            .find(|line| {
                line.split_once(',')
                    .is_some_and(|(row_case, _)| row_case == case)
            })
            .expect("paired reference data should contain the requested case");

        let fields: Vec<_> = row.split(',').collect();

        assert_eq!(
            fields.len(),
            6,
            "paired reference row should contain exactly six fields"
        );

        let n = parse_usize(fields[1], "reference n");

        Statistics {
            n,
            covariance: parse_finite(fields[2], "reference covariance"),
            pearson: parse_correlation(fields[3], "reference Pearson correlation"),
            spearman: parse_correlation(fields[4], "reference Spearman correlation"),
            kendall: parse_correlation(fields[5], "reference Kendall tau-b"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Dataset;

    #[test]
    fn all_datasets_load() {
        let datasets = [
            (Dataset::Baseline, 8),
            (Dataset::MixedOrder, 6),
            (Dataset::MonotonicNonlinear, 5),
            (Dataset::PerfectNegative, 5),
            (Dataset::PerfectPositive, 5),
            (Dataset::TiesBothVariables, 8),
            (Dataset::TiesOneVariable, 6),
            (Dataset::ZeroCovariance, 4),
        ];

        for (dataset, n) in datasets {
            let reference = dataset.load();
            assert_eq!(reference.dataset(), dataset);
            assert_eq!(reference.x().len(), n);
            assert_eq!(reference.y().len(), n);
            assert_eq!(reference.statistics().n(), n);
        }
    }
}
