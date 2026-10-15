use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::SeedableRng;
use rand::rngs::StdRng;
use unrvl_numerics::dot::compensated_dot;
use unrvl_numerics::summation::NeumaierSumExt;

#[path = "support/data.rs"]
mod data;
use data::{CASES, SEED, SIZES, generate_samples};

fn benchmark_dot(c: &mut Criterion) {
    for (name, offset) in CASES {
        benchmark_case(c, name, |n, rng| {
            let xs = generate_samples(n, offset, rng);
            let ys = generate_samples(n, offset, rng);
            (xs, ys)
        });
    }

    benchmark_case(c, "sum_cancellation", |n, _| {
        // Multiplication by 1 is exact; only accumulation needs compensation.
        let xs = [1e16, 1.0, -1e16, 1.0]
            .into_iter()
            .cycle()
            .take(n)
            .collect();
        (xs, vec![1.0; n])
    });

    benchmark_case(c, "product_cancellation", |n, _| {
        // Each pair contributes (1 + d)*(1 - d) - 1 = -d² exactly.
        // The first product rounds to 1, so compensating only its sum cannot
        // recover that lost contribution. Full compensation uses mul_add.
        let delta = 2.0_f64.powi(-27);
        let xs = [1.0 + delta, -1.0].into_iter().cycle().take(n).collect();
        let ys = [1.0 - delta, 1.0].into_iter().cycle().take(n).collect();
        (xs, ys)
    });
}

fn benchmark_case(
    c: &mut Criterion,
    name: &str,
    generate: impl Fn(usize, &mut StdRng) -> (Vec<f64>, Vec<f64>),
) {
    let mut group = c.benchmark_group(format!("numerics/dot/{name}"));

    for n in SIZES {
        let mut rng = StdRng::seed_from_u64(SEED);
        let (xs, ys) = generate(n, &mut rng);
        // Count aligned pairs, rather than individual values across both
        // slices.
        group.throughput(Throughput::Elements(n as u64));

        group.bench_function(BenchmarkId::new("compensated_dot", n), |b| {
            b.iter(|| compensated_dot(black_box(&xs), black_box(&ys)));
        });
        group.bench_function(BenchmarkId::new("compensated_sum_of_products", n), |b| {
            b.iter(|| {
                black_box(xs.as_slice())
                    .iter()
                    .zip(black_box(ys.as_slice()))
                    .map(|(&x, &y)| x * y)
                    .neumaier_sum()
            });
        });
        group.bench_function(BenchmarkId::new("ordinary_dot", n), |b| {
            b.iter(|| {
                black_box(xs.as_slice())
                    .iter()
                    .zip(black_box(ys.as_slice()))
                    .map(|(&x, &y)| x * y)
                    .sum::<f64>()
            });
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_dot);
criterion_main!(benches);
