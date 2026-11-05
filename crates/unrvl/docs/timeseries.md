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

[`Series`] owns corresponding timestamp and value vectors. Construction requires
matching lengths, finite values, and strictly increasing timestamps;
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
validate them directly. Both types expose immutable timestamp and value slices.
Structural validation leaves cadence, coverage, and session membership to the
corresponding operations.

[`SeriesView::between`] selects observations by timestamp using Rust ranges:
`start..end` excludes the end and `start..=end` includes it. Bounds may be
unbounded or extend beyond the data. Valid ranges with no matching observations
return empty views; reversed bounds return an error. Selections borrow the
corresponding slices without allocation or revalidation.

# Working with Intervals

[`TimeInterval`] represents a nonempty `[start, end)` between supplied
timestamps. It includes the start and excludes the end, and exposes the exact
elapsed duration. [`IntervalError`] reports endpoints that fail `start < end`.
An interval can describe one continuous session open period.

```rust
# use unrvl::timeseries::{TimeInterval, Timestamp};
let open = Timestamp::from_unix_secs(9 * 60 * 60)?;
let close = Timestamp::from_unix_secs(16 * 60 * 60)?;
let interval = TimeInterval::new(open, close)?;

assert!(interval.contains(open));
assert!(!interval.contains(close));
# Ok::<(), Box<dyn std::error::Error>>(())
```

Intervals implement [`std::ops::RangeBounds`], so they can be passed directly to
[`SeriesView::between`] to select observations within the same bounds.

# Working with Sessions

[`Session`] owns a nonempty identifier and one or more scheduled open periods.
Periods must be ordered and nonoverlapping; adjacent periods and periods that
cross UTC midnight are valid. Gaps between periods represent breaks.
[`SessionError`] reports invalid construction with the session identifier,
interval indices, and endpoints where applicable.

```rust
# use std::time::Duration;
# use unrvl::timeseries::{ScheduleStatus, Session, SessionSchedule, TimeInterval, Timestamp};
let morning = TimeInterval::new(
    Timestamp::from_unix_secs(9 * 3600)?,
    Timestamp::from_unix_secs(12 * 3600)?,
)?;

let afternoon = TimeInterval::new(
    Timestamp::from_unix_secs(13 * 3600)?,
    Timestamp::from_unix_secs(16 * 3600)?,
)?;

let session = Session::new("regular", vec![morning, afternoon])?;

assert!(session.is_open_at(Timestamp::from_unix_secs(10 * 3600)?));
assert!(!session.is_open_at(Timestamp::from_unix_secs(12 * 3600)?));

let coverage = TimeInterval::new(
    Timestamp::UNIX_EPOCH,
    Timestamp::from_unix_secs(24 * 3600)?,
)?;

let schedule = SessionSchedule::new(coverage, vec![session])?;

let open_session = schedule.session_at(morning.start());
assert_eq!(open_session.map(Session::id), Some("regular"));
assert_eq!(schedule.status_at(morning.start()), ScheduleStatus::Open);
assert_eq!(schedule.status_at(morning.end()), ScheduleStatus::Closed);
assert_eq!(schedule.status_at(coverage.end()), ScheduleStatus::OutsideCoverage);
assert_eq!(schedule.open_duration(coverage), Some(Duration::from_secs(6 * 3600)));

let (_, close) = schedule.close_labels().next().unwrap();
assert_eq!(close, afternoon.end());
# Ok::<(), Box<dyn std::error::Error>>(())
```

Identifiers accept `&str` or `String` and preserve the supplied text exactly.
They carry no inferred date or timezone meaning. [`Session::is_open_at`] tests
only this session's open periods.

[`SessionSchedule`] owns sessions within declared coverage. Identifiers must be
unique, and session envelopes -- from first opening to final close -- must be
ordered, nonoverlapping, and contained in the coverage. Adjacent envelopes are
valid; another session cannot occupy a break. [`ScheduleError`] reports the
first invalid session with its identifier, index, and relevant interval bounds.

Coverage is the time range for which you supply all scheduled sessions, such as
a whole month. [`SessionSchedule::status_at`] reports `Open` during a session's
open hours and `Closed` during breaks or other covered times. An empty schedule
is closed throughout coverage. `OutsideCoverage` means the schedule has no
information for that timestamp.

Use [`SessionSchedule::session_at`] to borrow the open session itself. Use
`status_at` when you need to distinguish a closure from a time outside coverage.

[`SessionSchedule::previous_session`] finds the last session whose final close
is at or before a timestamp. [`SessionSchedule::next_session`] finds the first
session that opens strictly later. Both search the supplied sessions even for
timestamps outside coverage.

[`SessionSchedule::open_duration`] adds the open time within a range. Breaks
and gaps contribute zero. It returns `None` when any part of the range lies
outside coverage.

[`SessionSchedule::close_labels`] pairs each session with its final close.
These are the timestamps to look for when you expect one closing observation
per session. A session that closes at the end of coverage still has a close
label at that timestamp.

Construction cannot detect omitted sessions.
