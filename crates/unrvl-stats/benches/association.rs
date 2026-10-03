use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::SeedableRng;
use rand::rngs::StdRng;
use unrvl_stats::association::{
    correlation_matrix, covariance, covariance_matrix, kendall, pearson, spearman,
};

#[path = "support/data.rs"]
mod data;
use data::{CASES, SEED, generate_samples};

fn benchmark_paired(c: &mut Criterion) {
    let sizes = [32_usize, 1_024, 32_768, 1_048_576];

    for (name, offset) in CASES {
        let mut group = c.benchmark_group(format!("association/paired/{name}"));

        for n in sizes {
            let mut rng = StdRng::seed_from_u64(SEED);
            let xs = generate_samples(n, offset, &mut rng);
            let ys = generate_samples(n, offset, &mut rng);

            // Elements count paired observation positions, each containing x
            // and y.
            group.throughput(Throughput::Elements(n as u64));

            group.bench_function(BenchmarkId::new("covariance", n), |b| {
                b.iter(|| covariance(black_box(&xs), black_box(&ys)));
            });

            group.bench_function(BenchmarkId::new("pearson", n), |b| {
                b.iter(|| pearson(black_box(&xs), black_box(&ys)));
            });

            group.bench_function(BenchmarkId::new("spearman", n), |b| {
                b.iter(|| spearman(black_box(&xs), black_box(&ys)));
            });
        }

        group.finish();
    }
}

fn benchmark_kendall(c: &mut Criterion) {
    // Include intermediate sizes as well as larger sorting workloads.
    let sizes = [32_usize, 256, 1_024, 4_096, 32_768, 1_048_576];

    for (name, offset) in CASES {
        let mut group = c.benchmark_group(format!("association/kendall/{name}"));

        for n in sizes {
            let mut rng = StdRng::seed_from_u64(SEED);
            let xs = generate_samples(n, offset, &mut rng);
            let ys = generate_samples(n, offset, &mut rng);

            // Elements count paired observation positions, each containing x
            // and y.
            group.throughput(Throughput::Elements(n as u64));
            group.bench_function(BenchmarkId::new("tau_b", n), |b| {
                b.iter(|| kendall(black_box(&xs), black_box(&ys)));
            });
        }

        group.finish();
    }
}

fn benchmark_matrices(c: &mut Criterion) {
    // Pairwise work grows quadratically with the number of variables and
    // linearly with the number of observations per series.
    let shapes = [(32, 2), (1_024, 2), (1_024, 8), (1_024, 32), (32_768, 8)];

    for (name, offset) in CASES {
        let mut group = c.benchmark_group(format!("association/matrix/{name}"));

        for (n, variables) in shapes {
            let mut rng = StdRng::seed_from_u64(SEED);
            let series: Vec<_> = (0..variables)
                .map(|_| generate_samples(n, offset, &mut rng))
                .collect();

            let shape = format!("{variables}series_x_{n}observations");

            // Elements count all input values. Matrix shape is part of the ID.
            group.throughput(Throughput::Elements((n * variables) as u64));

            group.bench_function(BenchmarkId::new("covariance", &shape), |b| {
                b.iter(|| covariance_matrix(black_box(&series)).unwrap());
            });

            group.bench_function(BenchmarkId::new("correlation", &shape), |b| {
                b.iter(|| correlation_matrix(black_box(&series)).unwrap());
            });
        }

        group.finish();
    }
}

criterion_group!(
    benches,
    benchmark_paired,
    benchmark_kendall,
    benchmark_matrices
);
criterion_main!(benches);
