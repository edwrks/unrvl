use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use unrvl_numerics::scaling::{binary_exponent, power_of_two_unit, scale_power_of_two};

fn benchmark_scaling(c: &mut Criterion) {
    // Exercise the normal, subnormal-input, extreme-shift, and subnormal-output
    // paths without input generation or allocation in the timed loops.
    let cases = [
        ("normal", 3.25, 8),
        ("subnormal_input", f64::from_bits(3), 1),
        ("extreme_shift", f64::from_bits(1), 2097),
        ("subnormal_output", 1.5, -1075),
    ];

    for (name, value, exponent) in cases {
        let mut group = c.benchmark_group(format!("numerics/scaling/{name}"));

        group.throughput(Throughput::Elements(1));

        group.bench_function("power_of_two_unit", |b| {
            b.iter(|| power_of_two_unit(black_box(value)));
        });

        group.bench_function("binary_exponent", |b| {
            b.iter(|| binary_exponent(black_box(value)));
        });

        group.bench_function("scale_power_of_two", |b| {
            b.iter(|| scale_power_of_two(black_box(value), black_box(exponent)));
        });

        group.finish();
    }
}

criterion_group!(benches, benchmark_scaling);
criterion_main!(benches);
