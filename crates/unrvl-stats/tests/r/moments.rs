use reference_data::r::moments::Dataset;
use test_utils::approx::{Tolerance, assert_approx_eq};
use unrvl_stats::moments::{Kurtosis, ShapeEstimator};
use unrvl_stats::{Normalization, moments};

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

    assert_approx_eq(mean, stats.mean(), Tolerance::STRICT);

    let variance_cases = [
        (Normalization::Sample, stats.variance()),
        (Normalization::Population, stats.population_variance()),
        (Normalization::Ddof(2), stats.variance_ddof_2()),
    ];

    for (normalization, expected) in variance_cases {
        assert_approx_eq(moments::var(xs, normalization), expected, Tolerance::STRICT);
    }

    let skewness_cases = [
        (ShapeEstimator::Type1, stats.skewness_type_1()),
        (ShapeEstimator::Type2, stats.skewness_type_2()),
        (ShapeEstimator::Type3, stats.skewness_type_3()),
    ];

    for (estimator, expected) in skewness_cases {
        let actual = moments::skewness(xs, estimator);

        assert_approx_eq(actual, expected, Tolerance::STRICT);
    }

    let kurtosis_cases = [
        (
            ShapeEstimator::Type1,
            stats.excess_kurtosis_type_1(),
            stats.pearson_kurtosis_type_1(),
        ),
        (
            ShapeEstimator::Type2,
            stats.excess_kurtosis_type_2(),
            stats.pearson_kurtosis_type_2(),
        ),
        (
            ShapeEstimator::Type3,
            stats.excess_kurtosis_type_3(),
            stats.pearson_kurtosis_type_3(),
        ),
    ];

    for (estimator, excess, pearson) in kurtosis_cases {
        let expectations = [(Kurtosis::Excess, excess), (Kurtosis::Pearson, pearson)];

        for (convention, expected) in expectations {
            let actual = moments::kurtosis(xs, estimator, convention);

            assert_approx_eq(actual, expected, Tolerance::STRICT);
        }
    }
}
