# unrvl

Numerical, statistical, and time-series foundations for Rust. The API is
experimental.

Enable the components your application uses. No features are enabled by default:

```toml
[dependencies]
unrvl = { version = "0.0.3", features = ["stats"] }
```

```rust
use unrvl::stats::prelude::*;

let values = [2.0, 4.0, 6.0];

assert_eq!(values.mean(), 4.0);
assert_eq!(values.std_dev(Normalization::Sample), 2.0);
```

The `numerics`, `stats`, and `timeseries` features expose the corresponding
modules. The `full` feature enables every component. The `risk` component is
under development and currently has no public API.

See the [API documentation](https://docs.rs/unrvl) for usage guides, estimator
conventions, and input contracts. The
[repository's validation guide](https://github.com/edwrks/unrvl#validation)
links to reference comparisons and benchmark instructions.
