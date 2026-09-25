use reference_data::r::quantiles::Dataset;
use test_utils::approx::{Tolerance, assert_approx_eq};
use unrvl_stats::quantile;

const PROBABILITIES: [f64; 9] = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0];

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

fn run(dataset: Dataset) {
    let reference = dataset.load();
    let xs = reference.observations();
    let stats = reference.statistics();

    // Ensure the committed data covers the complete declared grid.
    assert_eq!(stats.len(), PROBABILITIES.len());

    let batched = quantile::quantiles(xs, &PROBABILITIES);
    assert_eq!(batched.len(), PROBABILITIES.len());

    for (index, (&probability, expected)) in PROBABILITIES.iter().zip(stats).enumerate() {
        // Probability must parse identically.
        assert_eq!(expected.probability().to_bits(), probability.to_bits());

        let individual = quantile::quantile(xs, probability);

        assert_approx_eq(individual, expected.value(), Tolerance::STRICT);
        assert_approx_eq(batched[index], expected.value(), Tolerance::STRICT);
    }

    // The probability-grid assertion above establishes that index 4 is 0.5.
    assert_approx_eq(quantile::median(xs), stats[4].value(), Tolerance::STRICT);
}
