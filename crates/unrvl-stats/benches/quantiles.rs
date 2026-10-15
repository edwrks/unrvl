use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::SeedableRng;
use rand::rngs::StdRng;
use unrvl_stats::quantile::{median, quantile, quantiles};

#[path = "support/data.rs"]
mod data;
use data::{CASES, SEED, generate_samples};

fn benchmark_quantiles(c: &mut Criterion) {
    let sizes = [32_usize, 1_024, 32_768, 1_048_576];
    let probabilities = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99];

    for (name, offset) in CASES {
        let mut group = c.benchmark_group(format!("quantiles/{name}"));

        for n in sizes {
            let mut rng = StdRng::seed_from_u64(SEED);
            let xs = generate_samples(n, offset, &mut rng);

            group.throughput(Throughput::Elements(n as u64));

            group.bench_function(BenchmarkId::new("median", n), |b| {
                b.iter(|| median(black_box(xs.as_slice())));
            });

            group.bench_function(BenchmarkId::new("quantile_95", n), |b| {
                b.iter(|| quantile(black_box(xs.as_slice()), black_box(0.95)));
            });

            // Both workloads return the same seven Type 7 quantiles in a Vec.
            // The separate calls each copy and sort; the batch sorts once.
            group.bench_function(BenchmarkId::new("seven_separate_calls", n), |b| {
                b.iter(|| {
                    let sample = black_box(xs.as_slice());
                    black_box(probabilities.as_slice())
                        .iter()
                        .map(|&p| quantile(sample, p))
                        .collect::<Vec<_>>()
                });
            });

            group.bench_function(BenchmarkId::new("seven_batched_quantiles", n), |b| {
                b.iter(|| quantiles(black_box(xs.as_slice()), black_box(&probabilities)));
            });
        }

        group.finish();
    }
}

criterion_group!(benches, benchmark_quantiles);
criterion_main!(benches);
