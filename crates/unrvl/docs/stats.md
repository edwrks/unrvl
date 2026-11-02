Statistical routines for slices of `f64`.

Enable the `stats` feature to use this module.

# Working with Samples

Free functions live in their statistical modules. For example, import
`unrvl::stats::moments::mean` to calculate the average:

```rust
# use unrvl::stats::moments::mean;
let values = [2.0, 4.0, 6.0];

assert_eq!(mean(&values), 4.0);
```

Import `unrvl::stats::prelude::*` for slice methods:

```rust
# use unrvl::stats::prelude::*;
let values = [2.0, 4.0, 6.0];

assert_eq!(values.mean(), 4.0);
assert_eq!(values.std_dev(Normalization::Sample), 2.0);
```

# Paired Samples and Fitted Objects

Paired functions match observations by position. Matrix inputs contain one slice
per variable, each with the same observation count. Output row and column
indices follow the input variable order:

```rust
# use unrvl::stats::Normalization;
# use unrvl::stats::association::{MatrixError, covariance_matrix};
let x = [1.0, 2.0, 3.0];
let y = [2.0, 4.0, 6.0];
let matrix = covariance_matrix(&[x.as_slice(), y.as_slice()], Normalization::Sample)?;

assert_eq!(matrix, vec![vec![1.0, 2.0], vec![2.0, 4.0]]);
# Ok::<(), MatrixError>(())
```

[`Ecdf`](distribution::empirical::Ecdf) validates and sorts a copied sample once
for repeated queries. [Quantile functions](quantile) can interpolate between
observations; [`Ecdf::inverse_cdf`](distribution::empirical::Ecdf::inverse_cdf)
selects an observed value.

[`SimpleLinearRegression`](regression::simple_linear::SimpleLinearRegression)
stores a fitted line without retaining its input observations. Its R² and
adjusted R² describe the training sample; prediction and residual queries do not
update them.

# Undefined Results and Errors

Scalar statistics generally return `NaN` for undefined results, such as an
insufficient sample or an invalid divisor. Matrix operations, empirical
distribution construction, and regression fitting return typed `Result`s for
invalid inputs. Valid input can still yield an undefined statistic: a constant
response gives a valid horizontal fit with `NaN` R² and adjusted R².

Scaling and compensated arithmetic reduce numerical error where appropriate.
Results remain subject to `f64` rounding, underflow, and overflow.

