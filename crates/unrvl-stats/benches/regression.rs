use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::SeedableRng;
use rand::rngs::StdRng;
use unrvl_stats::regression::simple_linear::SimpleLinearRegression;

#[path = "support/data.rs"]
mod data;
use data::{CASES, SEED, generate_samples};

fn benchmark_regression(c: &mut Criterion) {
    let sizes = [32_usize, 1_024, 32_768, 1_048_576];

    for (name, offset) in CASES {
        let mut group = c.benchmark_group(format!("regression/simple_linear/{name}"));

        for n in sizes {
            let mut rng = StdRng::seed_from_u64(SEED);

            let xs = generate_samples(n, offset, &mut rng);
            let ys = generate_samples(n, offset, &mut rng);

            let model = SimpleLinearRegression::fit(&xs, &ys).unwrap();

            group.throughput(Throughput::Elements(n as u64));

            group.bench_function(BenchmarkId::new("fit", n), |b| {
                b.iter(|| SimpleLinearRegression::fit(black_box(&xs), black_box(&ys)).unwrap());
            });

            group.bench_function(BenchmarkId::new("residuals", n), |b| {
                b.iter(|| model.residuals(black_box(&xs), black_box(&ys)).unwrap());
            });
        }

        group.finish();
    }

    let model = SimpleLinearRegression::fit(&[0.0, 1.0], &[1.0, 3.0]).unwrap();
    c.bench_function("regression/simple_linear/predict", |b| {
        b.iter(|| black_box(model).predict(black_box(2.0)));
    });
}

criterion_group!(benches, benchmark_regression);
criterion_main!(benches);
