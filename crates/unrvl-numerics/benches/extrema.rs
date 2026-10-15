use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::SeedableRng;
use rand::rngs::StdRng;
use unrvl_numerics::convert::usize_to_f64;
use unrvl_numerics::extrema::{argmax, argmax_abs, argmin, argmin_abs, max_abs, min_abs, min_max};

#[path = "support/data.rs"]
mod data;
use data::{CASES, SEED, SIZES, generate_samples};

fn benchmark_extrema(c: &mut Criterion) {
    for (name, offset) in CASES {
        benchmark_case(c, name, |n, rng| generate_samples(n, offset, rng));
    }
    benchmark_case(c, "ascending", |n, _| (0..n).map(usize_to_f64).collect());
    benchmark_case(c, "descending", |n, _| {
        (0..n).rev().map(usize_to_f64).collect()
    });
    benchmark_case(c, "with_nan", |n, rng| {
        let mut xs = generate_samples(n, 0.0, rng);
        // Includes a leading NaN and one every sixteen positions afterwards.
        for x in xs.iter_mut().step_by(16) {
            *x = f64::NAN;
        }
        xs
    });
}

fn benchmark_case(
    c: &mut Criterion,
    name: &str,
    generate: impl Fn(usize, &mut StdRng) -> Vec<f64>,
) {
    let mut group = c.benchmark_group(format!("numerics/extrema/{name}"));

    for n in SIZES {
        let mut rng = StdRng::seed_from_u64(SEED);
        let xs = generate(n, &mut rng);
        group.throughput(Throughput::Elements(n as u64));

        group.bench_function(BenchmarkId::new("max_abs", n), |b| {
            b.iter(|| max_abs(black_box(&xs)));
        });
        group.bench_function(BenchmarkId::new("min_abs", n), |b| {
            b.iter(|| min_abs(black_box(&xs)));
        });
        group.bench_function(BenchmarkId::new("argmax", n), |b| {
            b.iter(|| argmax(black_box(&xs)));
        });
        group.bench_function(BenchmarkId::new("argmin", n), |b| {
            b.iter(|| argmin(black_box(&xs)));
        });
        group.bench_function(BenchmarkId::new("argmax_abs", n), |b| {
            b.iter(|| argmax_abs(black_box(&xs)));
        });
        group.bench_function(BenchmarkId::new("argmin_abs", n), |b| {
            b.iter(|| argmin_abs(black_box(&xs)));
        });
        group.bench_function(BenchmarkId::new("min_max", n), |b| {
            b.iter(|| min_max(black_box(&xs)));
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_extrema);
criterion_main!(benches);
