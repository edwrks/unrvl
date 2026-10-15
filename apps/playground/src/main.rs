use reference_data::r::association::matrix;
use unrvl::stats::association::{correlation_matrix, covariance_matrix};

fn main() {
    let data = matrix::Dataset::Baseline.load();
    println!("Variables: {:?}", data.variables());
    println!(
        "Covariance: {:?}",
        covariance_matrix(data.series()).unwrap()
    );
    println!(
        "Correlation: {:?}",
        correlation_matrix(data.series()).unwrap()
    );
}
