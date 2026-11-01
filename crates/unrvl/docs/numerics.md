Floating-point primitives for building numerical algorithms.

Enable the `numerics` feature to use this module. Import
`unrvl::numerics::prelude::*` for iterator extension methods, or access the
individual modules directly.

# Summing Observations

The iterator extension works with both owned `f64` values and borrowed `&f64`
values:

```rust
# use unrvl::numerics::prelude::*;
let values = [1.0, 2.0, 3.0];
let total = values.iter().neumaier_sum();

assert_eq!(total, 6.0);
```

For incremental work, [`summation::NeumaierSum`] supports `add`, `add_product`,
`extend`, and `total`. An empty accumulator or iterator sum returns zero.

# Working with Paired Values

[`dot::compensated_dot`] multiplies corresponding values and sums the products:

```rust
# use unrvl::numerics::dot::compensated_dot;
let x = [1.0, 2.0, 3.0];
let y = [4.0, 5.0, 6.0];

assert_eq!(compensated_dot(&x, &y), 32.0);
```

Use [`scaling`] when an algorithm needs to work in scaled units. Its helpers can
apply powers of two without first constructing a factor that might overflow or
underflow.

# Numerical Behavior

Compensation reduces rounding error within the range of `f64`. Compensated sums
and dot products are intended for finite operands; non-finite inputs or
intermediate overflow can produce non-finite results.

Input contracts vary by operation. Dot products panic for unequal slice lengths,
extrema panic for empty slices, and checked conversion panics outside the
consecutive exact integer range. Extrema ignore NaN when a non-NaN value is
present.

