use reference_data::r::regression::Dataset;
use test_utils::approx::{Tolerance, assert_all_approx_eq, assert_approx_eq};
use unrvl_stats::regression::simple_linear::SimpleLinearRegression;

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
fn perfect_negative() {
    run(Dataset::PerfectNegative);
}

#[test]
fn perfect_positive() {
    run(Dataset::PerfectPositive);
}

fn run(dataset: Dataset) {
    let reference = dataset.load();
    let statistics = reference.statistics();
    let model = SimpleLinearRegression::fit(reference.x(), reference.y()).unwrap();

    assert_eq!(model.observation_count(), statistics.n());
    assert_approx_eq(model.intercept(), statistics.intercept(), Tolerance::STRICT);
    assert_approx_eq(model.slope(), statistics.slope(), Tolerance::STRICT);
    assert_approx_eq(model.r_squared(), statistics.r_squared(), Tolerance::STRICT);

    let predictions: Vec<_> = reference.x().iter().map(|&x| model.predict(x)).collect();

    assert_all_approx_eq(&predictions, reference.predictions(), Tolerance::STRICT);
    assert_all_approx_eq(
        &model.residuals(reference.x(), reference.y()).unwrap(),
        reference.residuals(),
        Tolerance::STRICT,
    );
}
