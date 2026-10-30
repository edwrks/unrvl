# Numerics benchmarks

Run each category in Cargo's benchmark profile:

```sh
cargo bench -p unrvl-numerics --bench summation
cargo bench -p unrvl-numerics --bench dot
cargo bench -p unrvl-numerics --bench extrema
cargo bench -p unrvl-numerics --bench scaling
```

Filter a group, or exercise all benchmarks once without timing measurements:

```sh
cargo bench -p unrvl-numerics --bench dot -- 'numerics/dot/product_cancellation'
cargo bench -p unrvl-numerics -- --test
```

Criterion saves measurements under `target/criterion`. Group names start with
`numerics/`. Full measurement runs can take several minutes per category.

## Workloads

Reduction categories use 32, 1,024, 32,768, and 1,048,576 elements. Inputs are
generated before timing and reused across iterations.

| Target      | Operations                                                                           |
| ----------- | ------------------------------------------------------------------------------------ |
| `summation` | Neumaier iterator, explicit Neumaier accumulator, ordinary iterator sum              |
| `dot`       | Compensated dot product, compensated sum of rounded products, ordinary dot product   |
| `extrema`   | Absolute extrema, signed and absolute extremum indices, combined minimum and maximum |
| `scaling`   | Scalar unit selection, binary exponent extraction, and power-of-two scaling          |

Each reduction target includes uniform random inputs in [-5, 5) and the same draws shifted
by `1e12`. Each input dataset starts a `StdRng` seeded with `42`; dot-product
inputs draw successive samples from that RNG. Shared input generation lives in
`support/data.rs`.

The positive summation input shifts random observations into [1, 11).
Cancellation inputs repeat `[1e16, 1, -1e16, 1]`, whose exact sum per block
is 2. The dot product's summation case multiplies these values by ones. These
explicit stress inputs are generated outside the timed loops too.

The product-rounding case repeats `(1 + d)*(1 - d) - 1`, with `d = 2^-27`. Each
two-product block contributes `-d²` in exact arithmetic, but its first product
rounds to 1. Compensating only the sum cannot recover that lost value.
The compensated dot-product workload also exercises `NeumaierSum::add_product`.

Scaling uses four fixed scalar cases: normal values, subnormal inputs, an
exponent whose standalone factor overflows, and a subnormal result. Each
iteration processes one value.

Ascending and descending extrema inputs exercise opposite patterns of updates to
the running minimum and maximum. The NaN case replaces every sixteenth random
value, starting with the first, with NaN.

## Timing and throughput

Scalar reductions count `n` input values per iteration. Dot products count `n`
aligned pairs, each containing x and y. Each iteration measures the complete
reduction; input generation and allocation happen before timing.
Scaling throughput counts one scalar value per iteration.
