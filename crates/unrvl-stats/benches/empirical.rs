use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::SeedableRng;
use rand::rngs::StdRng;
use unrvl_stats::distribution::empirical::Ecdf;

#[path = "support/data.rs"]
mod data;
use data::{CASES, SEED, generate_samples};

fn benchmark_construction(c: &mut Criterion) {
    let sizes = [32_usize, 1_024, 32_768, 1_048_576];

    for (name, offset) in CASES {
        let mut group = c.benchmark_group(format!("empirical/{name}/construction"));

        for n in sizes {
            let mut rng = StdRng::seed_from_u64(SEED);
            let xs = generate_samples(n, offset, &mut rng);

            group.throughput(Throughput::Elements(n as u64));

            group.bench_function(BenchmarkId::new("new", n), |b| {
                // Includes validation, copying, sorting, and dropping the
                // result.
                b.iter(|| Ecdf::new(black_box(xs.as_slice())).unwrap());
            });
        }

        group.finish();
    }
}

fn benchmark_queries(c: &mut Criterion) {
    let sizes = [32_usize, 1_024, 32_768, 1_048_576];

    for (name, offset) in CASES {
        let mut group = c.benchmark_group(format!("empirical/{name}/queries"));

        for n in sizes {
            let mut rng = StdRng::seed_from_u64(SEED);
            let xs = generate_samples(n, offset, &mut rng);
            let distribution = Ecdf::new(&xs).unwrap();

            // Repeatedly query one observed value.
            let query = xs[n / 2];

            // Each single-query iteration performs one query, regardless of n.
            group.throughput(Throughput::Elements(1));

            group.bench_function(BenchmarkId::new("cdf", n), |b| {
                b.iter(|| black_box(&distribution).cdf(black_box(query)));
            });

            group.bench_function(BenchmarkId::new("sf", n), |b| {
                b.iter(|| black_box(&distribution).sf(black_box(query)));
            });

            group.bench_function(BenchmarkId::new("inverse_cdf_95", n), |b| {
                b.iter(|| black_box(&distribution).inverse_cdf(black_box(0.95)));
            });

            // Mix existing observations, between-value queries, and both tails.
            // Generate these once, outside the timed work.
            let minimum = distribution.inverse_cdf(0.0);
            let maximum = distribution.inverse_cdf(1.0);
            let span = maximum - minimum;
            let queries = [
                minimum - 1.0,
                minimum,
                xs[n / 4],
                span.mul_add(0.12345, minimum),
                query,
                span.mul_add(0.54321, minimum),
                xs[3 * n / 4],
                maximum,
                maximum + 1.0,
            ];

            group.throughput(Throughput::Elements(queries.len() as u64));

            group.bench_function(BenchmarkId::new("cdf_varied_batch", n), |b| {
                b.iter(|| {
                    let distribution = black_box(&distribution);
                    black_box(queries).map(|query| distribution.cdf(query))
                });
            });

            group.bench_function(BenchmarkId::new("sf_varied_batch", n), |b| {
                b.iter(|| {
                    let distribution = black_box(&distribution);
                    black_box(queries).map(|query| distribution.sf(query))
                });
            });
        }

        group.finish();
    }
}

criterion_group!(benches, benchmark_construction, benchmark_queries);
criterion_main!(benches);
