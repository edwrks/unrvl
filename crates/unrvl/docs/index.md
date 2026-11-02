Numerical, statistical, time-series, and risk analysis tools for Rust.

`unrvl` provides access to focused component crates through feature-gated
modules. Numerical and statistical operations work with `f64` values, slices,
and iterators. The API is experimental.

# Features

No features are enabled by default. Enable the modules your application uses:

```toml
[dependencies]
unrvl = { version = "0.0.3", features = ["numerics", "stats"] }
```

| Feature      | Module              | Availability      |
| ------------ | ------------------- | ----------------- |
| `numerics`   | `unrvl::numerics`   | Public API        |
| `stats`      | `unrvl::stats`      | Public API        |
| `timeseries` | `unrvl::timeseries` | No public API yet |
| `risk`       | `unrvl::risk`       | No public API yet |

The `full` feature enables all four modules. Enabling `stats` alone does not
expose `unrvl::numerics`.

# Using Statistics

With the `stats` feature enabled, import `unrvl::stats::prelude::*` for slice
methods:

```rust
# #[cfg(feature = "stats")]
# {
# use unrvl::stats::prelude::*;
let values = [2.0, 4.0, 6.0];

assert_eq!(values.mean(), 4.0);
assert_eq!(values.std_dev(Normalization::Sample), 2.0);
# }
```

Free functions are available through the same modules, such as
`unrvl::stats::moments::mean`.

