use reference_data::nist::univariate::Dataset;
use test_utils::approx::{Tolerance, assert_approx_eq};

#[test]
fn lew() {
    run(Dataset::Lew);
}

#[test]
fn lottery() {
    run(Dataset::Lottery);
}

#[test]
fn mavro() {
    run(Dataset::Mavro);
}

#[test]
fn michelso() {
    run(Dataset::Michelso);
}

#[test]
fn num_acc_1() {
    run(Dataset::NumAcc1);
}

#[test]
fn num_acc_2() {
    run(Dataset::NumAcc2);
}

#[test]
fn num_acc_3() {
    run(Dataset::NumAcc3);
}

#[test]
fn num_acc_4() {
    run(Dataset::NumAcc4);
}

#[test]
fn pi_digitis() {
    run(Dataset::PiDigits);
}

fn run(dataset: Dataset) {
    let reference = dataset.load();
    let stats = reference.statistics();
    let xs = reference.observations();
    let mean = unrvl_stats::moments::mean(xs);
    let std_dev = unrvl_stats::dispersion::std_dev(xs);

    // NIST certifies the decimal inputs. Parsing the large-offset NumAcc
    // observations into f64 perturbs their spread. Allow half an input ULP
    // for these fixtures, while retaining strict checks for the others.
    let std_dev_tolerance = match dataset {
        Dataset::NumAcc3 => Tolerance::new(2.0_f64.powi(-34), 1e-14),
        Dataset::NumAcc4 => Tolerance::new(2.0_f64.powi(-30), 1e-14),
        _ => Tolerance::STRICT,
    };

    assert_approx_eq(mean, stats.sample_mean(), Tolerance::VERY_STRICT);
    assert_approx_eq(
        std_dev,
        stats.sample_standard_deviation(),
        std_dev_tolerance,
    );
}
