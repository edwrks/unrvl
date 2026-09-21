use unrvl_numerics::convert::usize_to_f64;
use unrvl_numerics::dot::compensated_dot;
use unrvl_numerics::extrema;
use unrvl_numerics::summation::NeumaierSum;

#[test]
fn scalar_helpers_are_available_to_external_callers() {
    let _: f64 = usize_to_f64(42);
}

#[test]
fn compensated_dot_product_is_available_externally() {
    let xs = [1.0, 2.0, 3.0];
    let ys = [4.0, 5.0, 6.0];

    let _ = compensated_dot(&xs, &ys);
}

#[test]
fn extrema_is_available_externally() {
    let xs = &[1.0, 2.0, -3.0];

    let _ = extrema::max_abs(xs);
    let _ = extrema::min_abs(xs);
    let _ = extrema::argmax(xs);
    let _ = extrema::argmin(xs);
    let _ = extrema::argmax_abs(xs);
    let _ = extrema::argmin_abs(xs);
}

#[test]
fn neumaier_sum_is_available_externally() {
    let mut acc = NeumaierSum::new();
    acc.extend([1.0, 2.0, 3.0]);

    let _: NeumaierSum = [1.0, 2.0, 3.0].into_iter().collect();
    let _: NeumaierSum = [1.0, 2.0, 3.0].iter().sum();
}

#[test]
fn extension_traits_are_available_from_the_prelude() {
    use unrvl_numerics::prelude::*;

    let xs = [1.0, 2.0, 3.0];
    let _ = xs.iter().neumaier_sum();
}
