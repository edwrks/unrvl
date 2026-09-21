use unrvl_stats::prelude::*;

const XS: &[f64] = &[1.0, 2.0, 3.0, 4.0, 5.0];

#[test]
fn moment_functions_and_extension_methods_are_available_externally() {
    let _: f64 = unrvl_stats::moments::mean(XS);
    let _: f64 = unrvl_stats::moments::var(XS);
    let _: f64 = unrvl_stats::moments::skewness(XS);
    let _: f64 = unrvl_stats::moments::excess_kurtosis(XS);

    let _: f64 = XS.mean();
    let _: f64 = XS.var();
    let _: f64 = XS.skewness();
    let _: f64 = XS.excess_kurtosis();
}
