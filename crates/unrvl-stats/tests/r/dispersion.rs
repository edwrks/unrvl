use reference_data::r::dispersion::Dataset;
use test_utils::approx::{Tolerance, assert_approx_eq};
use unrvl_stats::dispersion;

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

    let iqr = dispersion::iqr(xs);
    let mad = dispersion::mad(xs);
    let range = dispersion::range(xs);
    let std_dev = dispersion::std_dev(xs);
    let cv = dispersion::cv(xs);

    assert_approx_eq(iqr, stats.iqr(), Tolerance::STRICT);
    assert_approx_eq(mad, stats.mad(), Tolerance::STRICT);
    assert_approx_eq(range, stats.range(), Tolerance::STRICT);
    assert_approx_eq(std_dev, stats.std_dev(), Tolerance::STRICT);
    assert_approx_eq(cv, stats.cv(), Tolerance::STRICT);
}
