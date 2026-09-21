# unrvl

Quantitative analysis tools for Rust.

`unrvl` is an opinionated collection of numerical, statistical, time-series, and
risk-analysis tools for working with financial data.

It is purposefully _not_ a comprehensive statistics framework. The project
grows only as the need arises.

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
unrvl = { version = "...", features = ["stats"] }
```

or fully enabled:

```toml
[dependencies]
unrvl = { version = "...", features = ["full"] }
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

`unrvl` favours small, composable functions with explicit definitions.

A few principles guide the project:

- keep numerical primitives, statistical methods, time-series operations, and
  risk measures separate;
- use `f64` deliberately rather than generalising over numeric types;
- make estimator conventions, assumptions, and undefined cases explicit;
- improve numerical robustness where practical;
- validate important implementations against independent reference data and
  mathematical properties; and
- add abstractions and configuration only when they earn their place.

Where multiple valid conventions exist, `unrvl` chooses a well-motivated default
for its intended analytical use, documents that choice, and only exposes
alternatives when they serve a concrete purpose.

For statistics intended to characterise an underlying population or process, the
default functions use documented sample estimators. For example, variance is
Bessel-corrected sample variance, while skewness and excess kurtosis use
bias-corrected sample estimators.

The goal is not to hide these choices behind an API. A function should make it
possible to understand what quantity is being calculated and what that quantity
means.

## Numerical Behaviour

Floating-point arithmetic can complicate otherwise simple statistical formulas.

Where useful, `unrvl` uses techniques such as compensated summation, corrected
centring, and scaled deviations to reduce avoidable loss of precision, overflow,
and underflow.

These techniques improve numerical behaviour but do not remove the finite range
and precision of `f64`. Individual functions document relevant behaviour for
cases such as:

- insufficient observations;
- non-finite inputs;
- constant samples;
- undefined statistics;
- overflow and underflow; and
- estimator-specific sample requirements.

Undefined statistical results generally return `NaN`. Programmer errors, such as
incompatible input shapes, may panic.

## Status

`unrvl` is experimental and under active development.

The API is not yet stable, and functionality is added selectively as the
numerical and statistical foundations mature.

## License

Licensed under either of:

- Apache License, Version 2.0
- MIT License

at your option.

