# unrvl

An  opinionated collection of quantitative analysis tools for Rust, with a focus
on financial data.

## Crates

`unrvl` is split into focused component crates behind a small facade:

- **`unrvl`**: optional access to the component crates through a facade.
- **`unrvl-numerics`**: numerical primitives and floating-point utilities.
- **`unrvl-stats`**: descriptive statistics and statistical analysis.
- **`unrvl-timeseries`**: time-series transformations, rolling analysis, and
  temporal statistics.
- **`unrvl-risk`**: risk measures and analytical tools for financial data.

The facade has no default features. Components can be enabled individually:

```toml
[dependencies]
unrvl = { version = "0.0.2", features = ["stats"] }
```

or fully enabled:

```toml
[dependencies]
unrvl = { version = "0.0.2", features = ["full"] }
```

and accessed through their corresponding module:

```rust
use unrvl::stats::moments;

let xs = [1.0, 2.0, 3.0, 4.0, 5.0];

let mean = moments::mean(&xs);
let variance = moments::var(&xs);
```

Extension traits are available from the component crate's prelude:

```rust
use unrvl::stats::prelude::*;

let xs = [1.0, 2.0, 3.0, 4.0, 5.0];

let mean = xs.mean();
let variance = xs.var();
```

## Approach

The APIs operate deliberately on `f64` values and document their estimator
conventions, input requirements, and undefined results.

Implementations are checked against independent reference data and mathematical
properties.

## Numerical Behavior

The implementations use compensated summation and scaling where appropriate
to reduce rounding error and avoid intermediate overflow and underflow. Results
remain subject to the range and precision of `f64`.

Individual functions document their input requirements and numerical limitations.
Undefined statistical results generally return `NaN`; invalid input shapes may
panic.

## Status

`unrvl` is experimental and under active development.

The API is not yet stable, and functionality is added selectively as the
numerical and statistical foundations mature.

## License

Licensed under either of:

- Apache License, Version 2.0
- MIT License

at your option.

