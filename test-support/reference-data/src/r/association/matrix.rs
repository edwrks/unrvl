//! Aligned variable series and R association matrix references.
//!
//! Select a [`Dataset`] and call [`Dataset::load`] to obtain named variables,
//! variable series, and [`Statistics`]. Variable order follows the fixture
//! header. Reference entries use zero-based variable indices and include the
//! diagonal and both triangles in row-major order.
//!
//! ```
//! use reference_data::r::association::matrix::Dataset;
//!
//! let reference = Dataset::Baseline.load();
//! let statistics = reference.statistics();
//! assert_eq!(reference.series().len(), statistics.dimension());
//! assert!(
//!     reference
//!         .series()
//!         .iter()
//!         .all(|series| series.len() == statistics.n())
//! );
//! assert_eq!(reference.variables().len(), statistics.dimension());
//! assert_eq!(statistics.entries().len(), statistics.dimension().pow(2));
//! ```

use crate::r::parsing::{parse_correlation, parse_finite, parse_usize};

/// A bundled R association matrix dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dataset {
    /// Three variables with ordinary observations.
    Baseline,

    /// Variables with different numerical scales.
    DifferentScales,

    /// Nonconstant series with exact linear dependence.
    LinearlyDependent,

    /// A single nonconstant variable and one-by-one reference matrices.
    SingleVariable,
}

impl Dataset {
    const REFERENCES: &'static str = include_str!("../../../data/r/association-matrix.csv");

    /// Loads aligned variable series and their R reference entries.
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
            Self::DifferentScales => "different-scales",
            Self::LinearlyDependent => "linearly-dependent",
            Self::SingleVariable => "single-variable",
        }
    }

    const fn fixture(self) -> &'static str {
        match self {
            Self::Baseline => {
                include_str!("../../../data/fixtures/r/association-matrix/baseline.csv")
            }
            Self::DifferentScales => {
                include_str!("../../../data/fixtures/r/association-matrix/different-scales.csv")
            }
            Self::LinearlyDependent => {
                include_str!("../../../data/fixtures/r/association-matrix/linearly-dependent.csv")
            }
            Self::SingleVariable => {
                include_str!("../../../data/fixtures/r/association-matrix/single-variable.csv")
            }
        }
    }
}

/// Aligned variable series and their R reference matrix statistics.
#[derive(Debug)]
pub struct Reference {
    dataset: Dataset,
    variables: Vec<String>,
    series: Vec<Vec<f64>>,
    statistics: Statistics,
}

impl Reference {
    /// The bundled dataset this reference was loaded from.
    #[must_use]
    pub const fn dataset(&self) -> Dataset {
        self.dataset
    }

    /// Unique variable names, in fixture-header order.
    ///
    /// These indices identify series in [`Self::series`] and both coordinates
    /// in each reference [`Entry`].
    #[must_use]
    pub fn variables(&self) -> &[String] {
        &self.variables
    }

    /// Finite aligned series, one per variable in fixture-header order.
    ///
    /// The outer slice follows [`Self::variables`] order. Each inner vector
    /// contains observations in fixture-row order. There is at least one
    /// series, and all series have the same length of at least two.
    /// Series index `i` identifies row and column `i` of each reference matrix.
    #[must_use]
    pub fn series(&self) -> &[Vec<f64>] {
        &self.series
    }

    /// Reference covariance and Pearson correlation matrix entries.
    #[must_use]
    pub const fn statistics(&self) -> &Statistics {
        &self.statistics
    }
}

/// Matrix dimensions and reference entries computed by R.
#[derive(Debug)]
pub struct Statistics {
    n: usize,
    dimension: usize,
    entries: Vec<Entry>,
}

impl Statistics {
    /// Number of aligned observations per series used by R.
    #[must_use]
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Number of variables, and the width/height of each square reference
    /// matrix.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Every matrix coordinate exactly once, ordered by row then column.
    ///
    /// Contains `dimension() * dimension()` entries, including the diagonal
    /// and both triangles. Coordinates index variables, not observation rows.
    #[must_use]
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }
}

/// Reference covariance and Pearson correlation for a pair of variables.
#[derive(Debug, Clone, Copy)]
pub struct Entry {
    row: usize,
    column: usize,
    covariance: f64,
    correlation: f64,
}

impl Entry {
    /// Zero-based row variable index in fixture-header order.
    #[must_use]
    pub const fn row(&self) -> usize {
        self.row
    }

    /// Zero-based column variable index in fixture-header order.
    #[must_use]
    pub const fn column(&self) -> usize {
        self.column
    }

    /// Sample covariance with denominator `n - 1`.
    #[must_use]
    pub const fn covariance(&self) -> f64 {
        self.covariance
    }

    /// Pearson correlation in `[-1, 1]`.
    #[must_use]
    pub const fn correlation(&self) -> f64 {
        self.correlation
    }
}

struct Parser {
    dataset: Dataset,
}

impl Parser {
    const REFERENCE_HEADER: &'static str = "case,n,row,column,covariance,correlation";

    const fn new(dataset: Dataset) -> Self {
        Self { dataset }
    }

    fn parse(self) -> Reference {
        let (variables, series) = self.parse_fixture();
        let (n, entries) = self.parse_statistics();
        let dimension = variables.len();

        assert_eq!(
            n,
            series[0].len(),
            "fixture observation count should match the R reference count"
        );

        let expected = dimension
            .checked_mul(dimension)
            .expect("matrix entry count should fit usize");

        assert_eq!(
            entries.len(),
            expected,
            "matrix reference should contain every coordinate"
        );

        for (index, entry) in entries.iter().enumerate() {
            assert!(
                entry.row < dimension && entry.column < dimension,
                "matrix reference indices should be in bounds"
            );

            assert_eq!(
                (entry.row, entry.column),
                (index / dimension, index % dimension),
                "matrix reference coordinates should be unique and in row-major order"
            );
        }

        let statistics = Statistics { n, dimension, entries };

        Reference {
            dataset: self.dataset,
            variables,
            series,
            statistics,
        }
    }

    fn parse_fixture(&self) -> (Vec<String>, Vec<Vec<f64>>) {
        let mut lines = self.dataset.fixture().lines();

        let header = lines
            .next()
            .expect("matrix fixture should contain a header");

        let variables: Vec<_> = header.split(',').map(str::to_owned).collect();

        for (index, name) in variables.iter().enumerate() {
            assert!(
                !name.trim().is_empty(),
                "matrix variable names should be nonempty"
            );

            assert!(
                !variables[..index].contains(name),
                "matrix variable names should be unique"
            );
        }

        // Keep the CSV's observation rows on disk, but collect each variable
        // directly into the series layout consumed by the Rust matrix API.
        let mut series = vec![Vec::new(); variables.len()];
        for line in lines {
            let fields: Vec<_> = line.split(',').collect();
            assert_eq!(
                fields.len(),
                variables.len(),
                "matrix fixture rows should match the variable count"
            );

            for (values, field) in series.iter_mut().zip(fields) {
                values.push(parse_finite(field, "matrix observation"));
            }
        }

        assert!(
            series[0].len() >= 2,
            "matrix fixture should contain at least two observations"
        );

        (variables, series)
    }

    fn parse_statistics(&self) -> (usize, Vec<Entry>) {
        let case = self.dataset.case_name();

        let mut reference = Dataset::REFERENCES.lines();

        let header = reference
            .next()
            .expect("matrix reference CSV should contain a header");

        assert_eq!(
            header,
            Self::REFERENCE_HEADER,
            "matrix reference CSV header should match the expected schema"
        );

        let mut count = None;
        let mut entries = Vec::new();

        for line in reference {
            let fields: Vec<_> = line.split(',').collect();
            if fields[0] != case {
                continue;
            }

            assert_eq!(
                fields.len(),
                6,
                "matrix reference row should contain exactly six fields"
            );

            let n = parse_usize(fields[1], "reference n");

            match count {
                Some(c) => assert_eq!(n, c, "matrix rows should have the same observation count"),
                None => count = Some(n),
            }

            let row = parse_usize(fields[2], "reference row");
            let column = parse_usize(fields[3], "reference column");
            let covariance = parse_finite(fields[4], "reference covariance");
            let correlation = parse_correlation(fields[5], "reference Pearson correlation");

            entries.push(Entry { row, column, covariance, correlation });
        }

        let n = count.expect("matrix reference data should contain the requested case");

        (n, entries)
    }
}

#[cfg(test)]
mod tests {
    use super::Dataset;

    #[test]
    fn series_follow_fixture_header_and_observation_order() {
        let reference = Dataset::Baseline.load();
        assert_eq!(reference.variables(), ["a", "b", "c"]);
        assert_eq!(
            reference.series(),
            [
                vec![1.0, 2.0, 4.0, 5.0],
                vec![4.0, 1.0, 3.0, 2.0],
                vec![2.0, 5.0, 1.0, 3.0],
            ]
        );
    }

    #[test]
    fn all_datasets_load() {
        for (dataset, dimension) in [
            (Dataset::Baseline, 3),
            (Dataset::DifferentScales, 3),
            (Dataset::LinearlyDependent, 3),
            (Dataset::SingleVariable, 1),
        ] {
            let reference = dataset.load();
            let stats = reference.statistics();

            assert_eq!(reference.dataset(), dataset);
            assert_eq!(reference.variables().len(), dimension);
            assert_eq!(reference.series().len(), dimension);
            assert!(reference.series().iter().all(|series| series.len() == 4));
            assert_eq!(stats.n(), 4);
            assert_eq!(stats.dimension(), dimension);
            assert_eq!(stats.entries().len(), dimension * dimension);
        }
    }
}
