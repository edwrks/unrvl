use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::SeedableRng;
use rand::rngs::StdRng;
use unrvl_stats::transform::{demean, ranks, z_score};

#[path = "support/data.rs"]
mod data;
use data::{CASES, SEED, generate_samples};

fn benchmark_transform(c: &mut Criterion) {
    let sizes = [32_usize, 1_024, 32_768, 1_048_576];

    for (name, offset) in CASES {
        let mut group = c.benchmark_group(format!("transform/{name}"));

        for n in sizes {
            let mut rng = StdRng::seed_from_u64(SEED);
            let xs = generate_samples(n, offset, &mut rng);

            group.throughput(Throughput::Elements(n as u64));

            group.bench_function(BenchmarkId::new("demean", n), |b| {
                b.iter(|| demean(black_box(xs.as_slice())));
            });

            group.bench_function(BenchmarkId::new("z_score", n), |b| {
                b.iter(|| z_score(black_box(xs.as_slice())));
            });

            group.bench_function(BenchmarkId::new("ranks", n), |b| {
                b.iter(|| ranks(black_box(xs.as_slice())));
            });
        }

        group.finish();
    }
}

criterion_group!(benches, benchmark_transform);
criterion_main!(benches);
