//! NIST regression references for one predictor and an intercept.
//!
//! Currently exposes only Norris.

/// A bundled NIST simple linear regression dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dataset {
    /// Ozone monitor calibration: 36 pairs, one predictor, and an intercept.
    Norris,
}

impl Dataset {
    /// Loads paired observations and certified fit statistics.
    ///
    /// # Panics
    ///
    /// Panics if bundled data is malformed or does not match its certified
    /// observation count.
    #[must_use]
    pub fn load(self) -> Reference {
        let input = self.input();
        let (x, y) = parse_observations(input);
        let statistics = parse_statistics(input);

        assert_eq!(x.len(), statistics.n, "NIST observation count mismatch");

        Reference { dataset: self, x, y, statistics }
    }

    const fn input(self) -> &'static str {
        match self {
            Self::Norris => include_str!("../../data/nist/linear-regression/Norris.dat"),
        }
    }
}

/// A bundled NIST regression dataset and certified statistics.
#[derive(Debug)]
pub struct Reference {
    dataset: Dataset,
    x: Vec<f64>,
    y: Vec<f64>,
    statistics: Statistics,
}

impl Reference {
    /// Dataset identity.
    #[must_use]
    pub const fn dataset(&self) -> Dataset {
        self.dataset
    }

    /// Predictor values in published observation order.
    #[must_use]
    pub fn x(&self) -> &[f64] {
        &self.x
    }

    /// Response values aligned with [`Self::x`].
    #[must_use]
    pub fn y(&self) -> &[f64] {
        &self.y
    }

    /// Certified NIST statistics.
    #[must_use]
    pub const fn statistics(&self) -> &Statistics {
        &self.statistics
    }
}

/// Certified NIST simple linear regression statistics.
#[derive(Debug, Clone, Copy)]
pub struct Statistics {
    n: usize,
    intercept: f64,
    slope: f64,
    r_squared: f64,
}

impl Statistics {
    /// Certified observation count.
    #[must_use]
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Certified intercept (B0).
    #[must_use]
    pub const fn intercept(&self) -> f64 {
        self.intercept
    }

    /// Certified slope (B1).
    #[must_use]
    pub const fn slope(&self) -> f64 {
        self.slope
    }

    /// Certified coefficient of determination.
    #[must_use]
    pub const fn r_squared(&self) -> f64 {
        self.r_squared
    }
}

fn parse_observations(input: &str) -> (Vec<f64>, Vec<f64>) {
    let mut lines = input.lines();

    // Skip the descriptive header to reach the published observation table.
    lines
        .find(|line| line.split_whitespace().eq(["Data:", "y", "x"]))
        .expect("NIST file should contain the y x data header");

    lines
        .take_while(|line| !line.trim().is_empty())
        .map(|line| {
            let fields: Vec<_> = line.split_whitespace().collect();
            assert_eq!(fields.len(), 2, "NIST observations should contain y then x");

            // NIST stores y before x; return predictors before responses.
            let y = parse_finite(fields[0]);
            let x = parse_finite(fields[1]);
            (x, y)
        })
        .unzip()
}

fn parse_statistics(input: &str) -> Statistics {
    let statistics = Statistics {
        n: parse_observation_count(input),
        intercept: parse_field(input, "B0"),
        slope: parse_field(input, "B1"),
        r_squared: parse_field(input, "R-Squared"),
    };

    assert!(
        (0.0..=1.0).contains(&statistics.r_squared),
        "NIST R² should be bounded"
    );

    statistics
}

fn parse_observation_count(input: &str) -> usize {
    // The metadata contains a line such as "36 Observations".
    for line in input.lines() {
        let mut fields = line.split_whitespace();
        let Some(count) = fields.next() else {
            continue;
        };

        if fields.next() == Some("Observations") && fields.next().is_none() {
            return count.parse().expect("NIST count should be an integer");
        }
    }

    panic!("NIST file should contain its observation count");
}

fn parse_field(input: &str, label: &str) -> f64 {
    // Certified rows start with a label, followed by its numerical value.
    let value = input
        .lines()
        .find_map(|line| {
            let mut fields = line.split_whitespace();
            if fields.next() != Some(label) {
                return None;
            }

            fields.next()
        })
        .unwrap_or_else(|| panic!("NIST file is missing {label}"));

    parse_finite(value)
}

fn parse_finite(value: &str) -> f64 {
    let value: f64 = value.parse().expect("NIST value should be numerical");
    assert!(value.is_finite(), "NIST value should be finite");
    value
}

#[cfg(test)]
mod tests {
    use super::Dataset;

    #[test]
    fn norris_loads_with_correct_predictor_and_response_order() {
        let reference = Dataset::Norris.load();

        assert_eq!(reference.dataset(), Dataset::Norris);
        assert_eq!(reference.statistics().n(), 36);
        assert_eq!(reference.x().len(), 36);
        assert_eq!(reference.y().len(), 36);
        assert_eq!(reference.x()[0], 0.2);
        assert_eq!(reference.y()[0], 0.1);
    }
}
