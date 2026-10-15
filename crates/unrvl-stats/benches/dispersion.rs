use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::SeedableRng;
use rand::rngs::StdRng;
use unrvl_stats::dispersion::{cv, iqr, mad, range, std_dev};

#[path = "support/data.rs"]
mod data;
use data::{CASES, SEED, generate_samples};

fn benchmark_dispersion(c: &mut Criterion) {
    let sizes = [32_usize, 1_024, 32_768, 1_048_576];

    for (name, offset) in CASES {
        let mut group = c.benchmark_group(format!("dispersion/{name}"));

        for n in sizes {
            let mut rng = StdRng::seed_from_u64(SEED);
            let xs = generate_samples(n, offset, &mut rng);
            group.throughput(Throughput::Elements(n as u64));

            group.bench_function(BenchmarkId::new("range", n), |b| {
                b.iter(|| range(black_box(xs.as_slice())));
            });

            group.bench_function(BenchmarkId::new("iqr", n), |b| {
                b.iter(|| iqr(black_box(xs.as_slice())));
            });

            group.bench_function(BenchmarkId::new("mad", n), |b| {
                b.iter(|| mad(black_box(xs.as_slice())));
            });

            group.bench_function(BenchmarkId::new("std_dev", n), |b| {
                b.iter(|| std_dev(black_box(xs.as_slice())));
            });

            group.bench_function(BenchmarkId::new("cv", n), |b| {
                b.iter(|| cv(black_box(xs.as_slice())));
            });

            group.bench_function(BenchmarkId::new("five_separate_calls", n), |b| {
                b.iter(|| {
                    let sample = black_box(xs.as_slice());
                    (
                        range(sample),
                        iqr(sample),
                        mad(sample),
                        std_dev(sample),
                        cv(sample),
                    )
                });
            });
        }

        group.finish();
    }
}

criterion_group!(benches, benchmark_dispersion);
criterion_main!(benches);
