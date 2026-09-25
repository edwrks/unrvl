use reference_data::r::moments::Dataset;
use test_utils::approx::{Tolerance, assert_approx_eq};
use unrvl_stats::moments;

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
fn symmetric() {
    run(Dataset::Symmetric);
}

#[test]
fn ties() {
    run(Dataset::Ties);
}

fn run(dataset: Dataset) {
    let reference = dataset.load();
    let xs = reference.observations();
    let stats = reference.statistics();

    let mean = moments::mean(xs);
    let variance = moments::var(xs);
    let skewness = moments::skewness(xs);
    let excess_kurtosis = moments::excess_kurtosis(xs);

    assert_approx_eq(mean, stats.mean(), Tolerance::STRICT);
    assert_approx_eq(variance, stats.variance(), Tolerance::STRICT);
    assert_approx_eq(skewness, stats.skewness(), Tolerance::STRICT);
    assert_approx_eq(excess_kurtosis, stats.excess_kurtosis(), Tolerance::STRICT);
}
