use unrvl_stats::prelude::*;
use unrvl_stats::{moments, quantile};

const XS: &[f64] = &[1.0, 2.0, 3.0, 4.0, 5.0];

#[test]
fn moment_functions_and_extension_methods_are_available_externally() {
    let _: f64 = moments::mean(XS);
    let _: f64 = moments::var(XS);
    let _: f64 = moments::skewness(XS);
    let _: f64 = moments::excess_kurtosis(XS);

    let _: f64 = XS.mean();
    let _: f64 = XS.var();
    let _: f64 = XS.skewness();
    let _: f64 = XS.excess_kurtosis();
}

#[test]
fn quantile_functions_and_extension_methods_are_available_externally() {
    let probabilities = [0.0, 0.5, 1.0];

    let _: f64 = quantile::quantile(XS, 0.9);
    let _: Vec<f64> = quantile::quantiles(XS, &probabilities);
    let _: f64 = quantile::median(XS);

    let _: f64 = XS.quantile(0.9);
    let _: Vec<f64> = XS.quantiles(&probabilities);
    let _: f64 = XS.median();
}
