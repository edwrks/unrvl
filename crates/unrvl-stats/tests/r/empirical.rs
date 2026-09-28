use reference_data::r::empirical::Dataset;
use test_utils::approx::{Tolerance, assert_approx_eq};
use unrvl_stats::distribution::empirical::Ecdf;

#[test]
fn baseline() {
    run(Dataset::Baseline);
}

#[test]
fn constant() {
    run(Dataset::Constant);
}

#[test]
fn singleton() {
    run(Dataset::Singleton);
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
    let distribution = Ecdf::new(reference.observations()).unwrap();
    let statistics = reference.statistics();

    for expected in statistics.evaluations() {
        let cdf = distribution.cdf(expected.query());
        let sf = distribution.sf(expected.query());

        assert_approx_eq(cdf, expected.cdf(), Tolerance::STRICT);
        assert_approx_eq(sf, expected.sf(), Tolerance::STRICT);
    }

    for expected in statistics.quantiles() {
        let quantile = distribution.inverse_cdf(expected.probability());

        // Hyndman–Fan Type 1 selects an observation; no interpolation tolerance
        // is needed.
        assert_eq!(quantile, expected.value());
    }
}
