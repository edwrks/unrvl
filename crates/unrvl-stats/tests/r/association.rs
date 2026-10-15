mod matrix;

use reference_data::r::association::paired::Dataset;
use test_utils::approx::{Tolerance, assert_approx_eq};
use unrvl_stats::association;

#[test]
fn baseline() {
    run(Dataset::Baseline);
}

#[test]
fn mixed_order() {
    run(Dataset::MixedOrder);
}

#[test]
fn monotonic_nonlinear() {
    run(Dataset::MonotonicNonlinear);
}

#[test]
fn perfect_negative() {
    run(Dataset::PerfectNegative);
}

#[test]
fn perfect_positive() {
    run(Dataset::PerfectPositive);
}

#[test]
fn ties_both_variables() {
    run(Dataset::TiesBothVariables);
}

#[test]
fn ties_one_variable() {
    run(Dataset::TiesOneVariable);
}

#[test]
fn zero_covariance() {
    run(Dataset::ZeroCovariance);
}

fn run(dataset: Dataset) {
    let reference = dataset.load();
    let stats = reference.statistics();

    let covariance = association::covariance(reference.x(), reference.y());
    let kendall = association::kendall(reference.x(), reference.y());
    let pearson = association::pearson(reference.x(), reference.y());
    let spearman = association::spearman(reference.x(), reference.y());

    assert_approx_eq(covariance, stats.covariance(), Tolerance::STRICT);
    assert_approx_eq(pearson, stats.pearson(), Tolerance::STRICT);
    assert_approx_eq(kendall, stats.kendall(), Tolerance::STRICT);
    assert_approx_eq(spearman, stats.spearman(), Tolerance::STRICT);
}
