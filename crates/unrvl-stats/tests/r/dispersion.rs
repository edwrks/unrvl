use reference_data::r::dispersion::Dataset;
use test_utils::approx::{Tolerance, assert_approx_eq};
use unrvl_stats::{Normalization, dispersion};

#[test]
fn asymmetric_tail() {
    run(Dataset::AsymmetricTail);
}

#[test]
fn baseline() {
    run(Dataset::Baseline);
}

#[test]
fn minimum_shape_sample() {
    run(Dataset::MinimumShapeSample);
}

#[test]
fn ties() {
    run(Dataset::Ties);
}

#[test]
fn two_point() {
    run(Dataset::TwoPoint);
}

fn run(dataset: Dataset) {
    let reference = dataset.load();
    let xs = reference.observations();
    let stats = reference.statistics();

    let iqr = dispersion::iqr(xs);
    let mad = dispersion::mad(xs);
    let range = dispersion::range(xs);

    assert_approx_eq(iqr, stats.iqr(), Tolerance::STRICT);
    assert_approx_eq(mad, stats.mad(), Tolerance::STRICT);
    assert_approx_eq(range, stats.range(), Tolerance::STRICT);

    let std_dev_cases = [
        (Normalization::Sample, stats.std_dev()),
        (Normalization::Population, stats.population_std_dev()),
    ];

    for (normalization, expected) in std_dev_cases {
        let actual = dispersion::std_dev(xs, normalization);

        assert_approx_eq(actual, expected, Tolerance::STRICT);
    }

    let cv_cases = [
        (Normalization::Sample, stats.cv()),
        (Normalization::Population, stats.population_cv()),
    ];

    for (normalization, expected) in cv_cases {
        let actual = dispersion::cv(xs, normalization);

        assert_approx_eq(actual, expected, Tolerance::STRICT);
    }
}
