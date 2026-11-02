Exact timestamps for time-series analysis.

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
