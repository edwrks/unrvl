use reference_data::r::transform::{Dataset, Statistics};
use test_utils::approx::{Tolerance, assert_all_approx_eq};
use unrvl_stats::{Normalization, transform};

#[test]
fn asymmetric_tail() {
    run(Dataset::AsymmetricTail);
}

#[test]
fn baseline() {
    run(Dataset::Baseline);
}

#[test]
fn signed_zeros() {
    run(Dataset::SignedZeros);
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
    let expected_demeaned: Vec<_> = reference
        .statistics()
        .iter()
        .map(Statistics::demeaned)
        .collect();

    let expected_sample_z_scores: Vec<_> = reference
        .statistics()
        .iter()
        .map(Statistics::z_score)
        .collect();

    let expected_population_z_scores: Vec<_> = reference
        .statistics()
        .iter()
        .map(Statistics::population_z_score)
        .collect();

    let expected_ranks: Vec<_> = reference
        .statistics()
        .iter()
        .map(Statistics::rank)
        .collect();

    let demeaned = transform::demean(xs);
    let sample_z_scores = transform::z_score(xs, Normalization::Sample);
    let population_z_scores = transform::z_score(xs, Normalization::Population);
    let ranks = transform::ranks(xs);

    assert_all_approx_eq(&demeaned, &expected_demeaned, Tolerance::STRICT);
    assert_all_approx_eq(
        &sample_z_scores,
        &expected_sample_z_scores,
        Tolerance::STRICT,
    );
    assert_all_approx_eq(
        &population_z_scores,
        &expected_population_z_scores,
        Tolerance::STRICT,
    );
    assert_eq!(ranks, expected_ranks);
}
