use unrvl_stats::distribution::empirical::Ecdf;
use unrvl_stats::prelude::*;
use unrvl_stats::{dispersion, moments, quantile};

const XS: &[f64] = &[1.0, 2.0, 3.0, 4.0, 5.0];

#[test]
fn empirical_distribution_is_available_externally() {
    let distribution = Ecdf::new(XS).unwrap();

    let _: f64 = distribution.cdf(3.0);
    let _: f64 = distribution.sf(3.0);
    let _: f64 = distribution.inverse_cdf(0.5);
}

#[test]
fn dispersion_functions_and_extension_methods_are_available_externally() {
    let _: f64 = dispersion::iqr(XS);
    let _: f64 = dispersion::mad(XS);
    let _: f64 = dispersion::range(XS);
    let _: f64 = dispersion::std_dev(XS);
    let _: f64 = dispersion::cv(XS);

    let _: f64 = XS.iqr();
    let _: f64 = XS.mad();
    let _: f64 = XS.range();
    let _: f64 = XS.std_dev();
    let _: f64 = XS.cv();
}

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
