# unrvl

An opinionated collection of quantitative analysis tools for Rust, with a focus
on financial data.

## Crates

`unrvl` is split into focused component crates behind a small facade:

- **`unrvl`**: optional access to the component crates through a facade.
- **`unrvl-numerics`**: numerical primitives and floating-point utilities.
- **`unrvl-stats`**: descriptive statistics and statistical analysis.
- **`unrvl-timeseries`**: exact timestamps, validated series, time-range
  selection, supplied session schedules, alignment, and sampling coverage.
- **`unrvl-risk`**: in development; no public API yet.

The facade has no default features. Components can be enabled individually:

```toml
[dependencies]
unrvl = { version = "0.0.3", features = ["stats"] }
```

or fully enabled:

```toml
[dependencies]
unrvl = { version = "0.0.3", features = ["full"] }
```

and accessed through their corresponding module:

```rust
use unrvl::stats::{Normalization, moments};

let xs = [1.0, 2.0, 3.0, 4.0, 5.0];

let mean = moments::mean(&xs);
let variance = moments::var(&xs, Normalization::Sample);
```

Extension traits are available from the component crate's prelude:

```rust
use unrvl::stats::prelude::*;

let xs = [1.0, 2.0, 3.0, 4.0, 5.0];

let mean = xs.mean();
let variance = xs.var(Normalization::Sample);
```

## Approach

The APIs operate deliberately on `f64` values and document their estimator
conventions, input requirements, and undefined results.

## Numerical Behavior

The implementations use compensated summation and scaling where appropriate
to reduce rounding error and avoid intermediate overflow and underflow. Results
remain subject to the range and precision of `f64`.

Scalar statistics generally return `NaN` for invalid samples or undefined
results. Paired slice statistics and dot products panic for unequal lengths.
Matrix operations, empirical distribution construction, and regression fitting
report invalid inputs through typed `Result`s. Individual functions document
their input requirements and numerical limitations.

## Validation

The [reference-data guide](test-support/reference-data/README.md) describes
comparisons against NIST certified values and committed R/e1071 outputs. The
[R generator guide](test-support/reference-data/scripts/r/README.md) documents
estimator conventions, pinned reference software versions, and regeneration
commands. Unit and property tests cover mathematical invariants and
floating-point boundaries, including cancellation and extreme magnitudes.

Run the workspace tests and documentation examples from the repository root:

```sh
cargo test --locked --workspace --all-targets --all-features
cargo test --locked --workspace --doc --all-features
```

The [numerics](crates/unrvl-numerics/benches/README.md) and
[statistics](crates/unrvl-stats/benches/README.md) benchmark guides describe
workloads and commands for measuring performance.

## Status

`unrvl` is experimental and under active development.

The API is not yet stable, and functionality is added selectively as the
numerical and statistical foundations mature.

## License

Licensed under either of:

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

at your option.

