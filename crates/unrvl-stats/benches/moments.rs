use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::SeedableRng;
use rand::rngs::StdRng;
use unrvl_stats::moments::{excess_kurtosis, mean, skewness, var};

#[path = "support/data.rs"]
mod data;
use data::{CASES, SEED, generate_samples};

fn benchmark_moments(c: &mut Criterion) {
    let sizes = [32_usize, 1_024, 32_768, 1_048_576];

    for (name, offset) in CASES {
        let mut group = c.benchmark_group(format!("moments/{name}"));

        for n in sizes {
            let mut rng = StdRng::seed_from_u64(SEED);
            let xs = generate_samples(n, offset, &mut rng);

            // Report throughput in input elements per second.
            group.throughput(Throughput::Elements(n as u64));

            group.bench_function(BenchmarkId::new("mean", n), |b| {
                b.iter(|| mean(black_box(xs.as_slice())));
            });

            group.bench_function(BenchmarkId::new("variance", n), |b| {
                b.iter(|| var(black_box(xs.as_slice())));
            });

            group.bench_function(BenchmarkId::new("skewness", n), |b| {
                b.iter(|| skewness(black_box(xs.as_slice())));
            });

            group.bench_function(BenchmarkId::new("excess_kurtosis", n), |b| {
                b.iter(|| excess_kurtosis(black_box(xs.as_slice())));
            });

            // Baseline for a caller who needs all four statistics.
            group.bench_function(BenchmarkId::new("four_separate_calls", n), |b| {
                b.iter(|| {
                    let sample = black_box(xs.as_slice());

                    (
                        mean(sample),
                        var(sample),
                        skewness(sample),
                        excess_kurtosis(sample),
                    )
                });
            });
        }

        group.finish();
    }
}

criterion_group!(benches, benchmark_moments);
criterion_main!(benches);
