use unrvl_stats::distribution::empirical::Ecdf;
use unrvl_stats::prelude::*;
use unrvl_stats::{association, dispersion, moments, quantile};

const XS: &[f64] = &[1.0, 2.0, 3.0, 4.0, 5.0];
const YS: &[f64] = &[2.0, 3.0, 4.0, 5.0, 6.0];

#[test]
fn association_functions_and_extension_methods_are_available_externally() {
    let _ = association::covariance(XS, YS);
    let _ = association::kendall(XS, YS);
    let _ = association::pearson(XS, YS);
    let _ = association::spearman(XS, YS);

    let _ = XS.covariance(YS);
    let _ = XS.kendall(YS);
    let _ = XS.pearson(YS);
    let _ = XS.spearman(YS);

    let result = association::covariance_matrix(&[XS, YS]);
    assert!(result.is_ok());

    let result = association::correlation_matrix(&[XS, YS]);
    assert!(result.is_ok());
}

#[test]
fn empirical_distribution_is_available_externally() {
    let distribution = Ecdf::new(XS).unwrap();

    let _ = distribution.cdf(3.0);
    let _ = distribution.sf(3.0);
    let _ = distribution.inverse_cdf(0.5);
}

#[test]
fn dispersion_functions_and_extension_methods_are_available_externally() {
    let _ = dispersion::iqr(XS);
    let _ = dispersion::mad(XS);
    let _ = dispersion::range(XS);
    let _ = dispersion::std_dev(XS);
    let _ = dispersion::cv(XS);

    let _ = XS.iqr();
    let _ = XS.mad();
    let _ = XS.range();
    let _ = XS.std_dev();
    let _ = XS.cv();
}

#[test]
fn moment_functions_and_extension_methods_are_available_externally() {
    let _ = moments::mean(XS);
    let _ = moments::var(XS);
    let _ = moments::skewness(XS);
    let _ = moments::excess_kurtosis(XS);

    let _ = XS.mean();
    let _ = XS.var();
    let _ = XS.skewness();
    let _ = XS.excess_kurtosis();
}

#[test]
fn quantile_functions_and_extension_methods_are_available_externally() {
    let ps = [0.0, 0.5, 1.0];

    let _ = quantile::quantile(XS, 0.9);
    let _ = quantile::quantiles(XS, &ps);
    let _ = quantile::median(XS);

    let _ = XS.quantile(0.9);
    let _ = XS.quantiles(&ps);
    let _ = XS.median();
}
