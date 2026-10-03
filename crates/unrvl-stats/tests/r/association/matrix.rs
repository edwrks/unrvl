use reference_data::r::association::matrix::Dataset;
use test_utils::approx::{Tolerance, assert_approx_eq};
use unrvl_stats::association::{correlation_matrix, covariance_matrix};

#[test]
fn baseline() {
    run(Dataset::Baseline);
}

#[test]
fn different_scales() {
    run(Dataset::DifferentScales);
}

#[test]
fn linearly_dependent() {
    run(Dataset::LinearlyDependent);
}

#[test]
fn single_variable() {
    run(Dataset::SingleVariable);
}

fn run(dataset: Dataset) {
    let reference = dataset.load();
    let series = reference.series();
    let stats = reference.statistics();
    let dimension = stats.dimension();

    let tol = Tolerance::STRICT;

    let covariance = covariance_matrix(series).unwrap();
    let correlation = correlation_matrix(series).unwrap();

    for matrix in [&covariance, &correlation] {
        assert_eq!(matrix.len(), dimension);
        assert!(matrix.iter().all(|row| row.len() == dimension));
    }

    for entry in stats.entries() {
        let row = entry.row();
        let col = entry.column();

        assert_approx_eq(covariance[row][col], entry.covariance(), tol);
        assert_approx_eq(correlation[row][col], entry.correlation(), tol);
    }
}
