use reference_data::nist::univariate::Dataset;
use reference_data::r::{self};
use unrvl::stats::moments;

fn main() {
    let lew = Dataset::Lew.load();
    println!("{lew:?}");

    let obs = lew.statistics().n();
    println!("{obs:?}");

    let mean = moments::mean(lew.observations());
    println!("Mean: {mean:?}");

    let r = r::moments::Dataset::Baseline.load();
    let r_mean = r.statistics().mean();
    println!("r_ref: {r:?}");

    let mean = moments::mean(r.observations());
    println!("ref: {r_mean}, ours: {mean}");
}
