use reference_data::nist::linear_regression::Dataset;
use test_utils::approx::{Tolerance, assert_approx_eq};
use unrvl_stats::regression::simple_linear::SimpleLinearRegression;

#[test]
fn norris() {
    let reference = Dataset::Norris.load();
    let expected = reference.statistics();

    let model = SimpleLinearRegression::fit(reference.x(), reference.y()).unwrap();
    assert_eq!(model.observation_count(), expected.n());

    // Published estimates are rounded to 15 significant decimal digits.
    // Intercept subtraction also amplifies the last bits of the slope.
    assert_approx_eq(
        model.intercept(),
        expected.intercept(),
        Tolerance::new(1e-12, 1e-12),
    );
    assert_approx_eq(model.slope(), expected.slope(), Tolerance::STRICT);
    assert_approx_eq(model.r_squared(), expected.r_squared(), Tolerance::STRICT);
}
