Exact timestamps and validated numerical series for time-series analysis.

Enable the `timeseries` feature to use this module. [`Timestamp`] stores signed
Unix nanoseconds; negative values precede `1970-01-01T00:00:00Z`.

Constructors from seconds, milliseconds, and microseconds return
[`TimestampError`] when the nanosecond count is out of range. Nanosecond
construction and access preserve the full `i64` value exactly.

Use [`std::time::Duration`] for nonnegative elapsed time. Checked addition and
subtraction return `None` for an out-of-range timestamp; checked elapsed
differences return `None` when the supplied earlier timestamp is later.

# Using Timestamps

```rust
# use std::time::Duration;
# use unrvl::timeseries::{Timestamp, TimestampError};
let timestamp = Timestamp::from_unix_secs(1)?;

assert_eq!(timestamp.as_unix_nanos(), 1_000_000_000);

let elapsed = Duration::from_secs(2);
let later = timestamp.checked_add(elapsed).unwrap();

assert_eq!(later.checked_duration_since(timestamp), Some(elapsed));
# Ok::<(), TimestampError>(())
```

# Working with Series

[`Series`] owns corresponding timestamp and value vectors. Construction
requires matching lengths, finite values, and strictly increasing timestamps;
[`SeriesError`] identifies invalid input. Empty samples, signed values, and
irregular observation intervals are valid.

```rust
# use unrvl::timeseries::{Series, Timestamp};
let timestamps = vec![
    Timestamp::from_unix_secs(0)?,
    Timestamp::from_unix_secs(1)?,
    Timestamp::from_unix_secs(3)?,
];
let values = vec![100.0, 102.0, 101.0];
let series = Series::new(timestamps, values)?;
let view = series.view();

assert_eq!(view.len(), 3);
assert_eq!(view.values(), &[100.0, 102.0, 101.0]);

let start = Timestamp::from_unix_secs(1)?;
let selected = view.between(start..)?;

assert_eq!(selected.values(), &[102.0, 101.0]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

[`Series::view`] borrows the stored observations without allocation or
revalidation. When borrowing existing slices, use [`SeriesView::new`] to
validate them directly. Both types expose immutable timestamp and value
slices. Structural validation leaves cadence, coverage, and session
membership to the corresponding operations.

[`SeriesView::between`] selects observations by timestamp using Rust ranges:
`start..end` excludes the end and `start..=end` includes it. Bounds may be
unbounded or extend beyond the data. Valid ranges with no matching
observations return empty views; reversed bounds return an error. Selections
borrow the corresponding slices without allocation or revalidation.
