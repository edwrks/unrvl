//! Shared scalar parsing and single-column fixture validation for R loaders.

pub(super) fn parse_finite(value: &str, field: &str) -> f64 {
    let value: f64 = value
        .parse()
        .unwrap_or_else(|_| panic!("{field} should be numerical"));

    assert!(value.is_finite(), "{field} should be finite");

    value
}

pub(super) fn parse_probability(value: &str, field: &str) -> f64 {
    let value = parse_finite(value, field);

    assert!((0.0..=1.0).contains(&value), "{field} should be in [0, 1]");

    value
}

pub(super) fn parse_correlation(value: &str, field: &str) -> f64 {
    let value = parse_finite(value, field);

    assert!(
        (-1.0..=1.0).contains(&value),
        "{field} should be in [-1, 1]"
    );

    value
}

pub(super) fn parse_usize(value: &str, field: &str) -> usize {
    value
        .parse()
        .unwrap_or_else(|_| panic!("{field} should be an unsigned integer"))
}

pub(super) fn parse_value_fixture(input: &str, context: &str) -> Vec<f64> {
    let mut lines = input.lines();
    assert_eq!(
        lines.next(),
        Some("value"),
        "{context} CSV header should match the expected schema"
    );

    let observations: Vec<_> = lines
        .map(|line| parse_finite(line.trim(), context))
        .collect();

    assert!(
        !observations.is_empty(),
        "{context} should contain observations"
    );

    observations
}
