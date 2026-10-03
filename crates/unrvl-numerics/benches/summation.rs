use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::SeedableRng;
use rand::rngs::StdRng;
use unrvl_numerics::summation::{NeumaierSum, NeumaierSumExt};

#[path = "support/data.rs"]
mod data;
use data::{CASES, SEED, SIZES, generate_samples};

fn benchmark_summation(c: &mut Criterion) {
    for (name, offset) in CASES {
        benchmark_case(c, name, |n, rng| generate_samples(n, offset, rng));
    }
    benchmark_case(c, "positive", |n, rng| generate_samples(n, 6.0, rng));
    benchmark_case(c, "cancellation", |n, _| {
        // Each block sums to 2 in exact arithmetic. Ordinary accumulation
        // loses small contributions when they are added to 1e16.
        [1e16, 1.0, -1e16, 1.0]
            .into_iter()
            .cycle()
            .take(n)
            .collect()
    });
}

fn benchmark_case(
    c: &mut Criterion,
    name: &str,
    generate: impl Fn(usize, &mut StdRng) -> Vec<f64>,
) {
    let mut group = c.benchmark_group(format!("numerics/summation/{name}"));

    for n in SIZES {
        let mut rng = StdRng::seed_from_u64(SEED);
        let xs = generate(n, &mut rng);
        group.throughput(Throughput::Elements(n as u64));

        group.bench_function(BenchmarkId::new("neumaier_iterator", n), |b| {
            b.iter(|| black_box(xs.as_slice()).iter().neumaier_sum());
        });
        group.bench_function(BenchmarkId::new("neumaier_accumulator", n), |b| {
            b.iter(|| {
                let mut sum = NeumaierSum::new();
                for &x in black_box(xs.as_slice()) {
                    sum.add(x);
                }
                sum.total()
            });
        });
        group.bench_function(BenchmarkId::new("ordinary_sum", n), |b| {
            b.iter(|| black_box(xs.as_slice()).iter().copied().sum::<f64>());
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_summation);
criterion_main!(benches);
