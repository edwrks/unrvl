//! Shared scalar parsing, reference-row lookup, and fixture validation for R
//! loaders.

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

/// Selects exactly one data row for a case. The caller validates the CSV header
/// and the selected row's fields.
pub(super) fn unique_reference_row<'a>(
    rows: impl Iterator<Item = &'a str>,
    case: &str,
    context: &str,
) -> &'a str {
    let mut matches = rows.filter(|line| {
        line.split_once(',')
            .is_some_and(|(row_case, _)| row_case == case)
    });

    let row = matches
        .next()
        .unwrap_or_else(|| panic!("{context}: missing case {case}"));

    assert!(matches.next().is_none(), "{context}: duplicate case {case}");

    row
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

#[cfg(test)]
mod tests {
    use super::unique_reference_row;

    #[test]
    #[should_panic(expected = "test reference: missing case missing")]
    fn rejects_missing_reference_case() {
        let rows = ["other,1"];

        let _ = unique_reference_row(rows.into_iter(), "missing", "test reference");
    }

    #[test]
    #[should_panic(expected = "test reference: duplicate case baseline")]
    fn rejects_duplicate_reference_case() {
        let rows = ["baseline,1", "other,2", "baseline,3"];

        let _ = unique_reference_row(rows.into_iter(), "baseline", "test reference");
    }

    #[test]
    fn selects_requested_reference_case() {
        let rows = ["first,1", "baseline,2", "last,3"];
        let row = unique_reference_row(rows.into_iter(), "baseline", "test reference");

        assert_eq!(row, "baseline,2");
    }
}
