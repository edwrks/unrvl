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
    let mean = unrvl_stats::moments::mean(reference.observations());

    assert_approx_eq(mean, stats.sample_mean(), Tolerance::VERY_STRICT);
}
