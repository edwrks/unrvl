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

# Checking Grid Sampling

[`cadence::FixedGrid`] describes timestamps separated by a fixed elapsed
duration, anchored at a supplied origin.

```rust
# use std::time::Duration;
# use unrvl::timeseries::{SeriesView, Timestamp};
# use unrvl::timeseries::cadence::{FixedGrid, GridCoverage};
let origin = Timestamp::from_unix_secs(9 * 3600)?;
let grid = FixedGrid::new(
    origin,
    Duration::from_secs(3600),
    Duration::from_secs(2 * 60),
)?;

let timestamps = [
    Timestamp::from_unix_secs(9 * 3600 + 60)?,
    Timestamp::from_unix_secs(11 * 3600 - 60)?,
];
let prices = SeriesView::new(&timestamps, &[100.0, 102.0])?;
let end = Timestamp::from_unix_secs(11 * 3600)?;
let coverage = GridCoverage::new(prices, grid, origin..=end)?;

assert_eq!(coverage.expected_count(), 3);
assert_eq!(coverage.missing_count(), 1);
assert_eq!(coverage.gaps()[0].first, Timestamp::from_unix_secs(10 * 3600)?);
# Ok::<(), Box<dyn std::error::Error>>(())
```

This describes hourly timestamps such as 08:00, 09:00, and 10:00, allowing up
to two minutes of deviation. The origin anchors the grid in both directions;
analysis bounds are separate.

Spacing must be positive. Tolerance is inclusive and strictly less than half
the spacing, so an accepted timestamp can belong to only one slot. Zero
tolerance requires exact timestamps. [`cadence::GridError`] reports invalid
configuration.

[`cadence::GridCoverage`] maps every supplied observation to a distinct grid
slot within explicit bounds. Bounds select expected grid labels: the example
accepts observations at 09:01 and 10:59 for the 09:00 and 11:00 slots. The
10:00 slot is missing. Original timestamps and values are preserved alongside
their slot labels and deviations.

Missing slots are reported in a successful result as compact consecutive
ranges with inclusive first and last labels. Counts describe expected,
observed, and missing slots; `is_complete()` checks that none are missing.
Expected and missing counts use `u128` to cover the full timestamp range.
[`cadence::CadenceError`] reports off-grid observations, labels outside the
bounds, collisions, or unrepresentable slot timestamps. No observations are
silently dropped or filled.

# Checking Session Closes

[`cadence::SessionCloseCoverage`] checks for one observation at each selected
session's exact final close. For example, a schedule containing Friday and
Monday sessions expects two closing observations across that period; the
weekend does not create missing samples.

Pass a series, its supplied schedule, and close-label bounds to
[`cadence::SessionCloseCoverage::new`]. Bounds may include or exclude either
endpoint. `..` selects the whole schedule, including a final close at its
coverage end. Explicit endpoints outside coverage return an error because the
schedule cannot establish expected sessions there.

The result preserves source observations and their sessions. Expected,
observed, and missing counts describe the selected closes. `missing()` yields
missing sessions with their identifiers and final close timestamps; `gaps()`
reports consecutive missing runs as ranges into the supplied schedule.
Missing closes are valid coverage outcomes. Use `is_complete()` when every
selected session must have an observation.

Every supplied observation must exactly match a selected final close. An
intermediate close before a session break does not qualify. Observations are
not dropped or filled, and nearby observations are not substituted for closes.
[`cadence::SessionCloseError`] identifies invalid bounds or the first unexpected
observation. Open-period membership is separate: a closing observation labels
the session that ended, including at a shared close/open boundary.

# Aligning Series

[`align::inner`] pairs observations from two [`SeriesView`]s at exactly equal
timestamps. Both series may have irregular spacing. Iterate over the result
to get each matched timestamp, both values, and their indices in the supplied
views. [`align::InnerAlignment`] also reports the number of observations left
unmatched on each side. Series with no shared timestamps give an empty result.

[`align::inner_slots`] pairs observations with equal checked grid labels from
two [`cadence::GridCoverage`] inputs. For example, observations one second
before and after 10:00 can match the same 10:00 slot when each is within its
grid's tolerance. Each pair preserves the shared label and both original
timestamps, values, source indices, and deviations.

Grids must have the same spacing and origins separated by whole grid steps.
Tolerances and analysis bounds may differ. Unmatched counts include all
supplied observations, including labels outside the other input's bounds.
[`align::SlotAlignmentError`] identifies incompatible grids. Empty
intersections are valid. Slot matching describes sampling positions;
information availability is assessed separately.

[`align::as_of_backward`] keeps every left observation and looks for the latest
right observation at or before its timestamp. The match must be within the
supplied maximum age, inclusive. For example, a trade at 10:03 can match a quote
at 10:01 with a two-minute limit. [`align::AsOfOutcome`] distinguishes matches,
missing prior observations, and candidates that are too old. Matched rows
include the right observation's index, timestamp, value, and elapsed age.

# Recording Availability

[`AvailabilityView`] pairs a series with caller-supplied timestamps recording
when each observation became available. For example, a price observed at 10:00
may become available at 10:07. A historical decision at 10:03 could not yet use
that price.

```rust
# use unrvl::timeseries::{AvailabilityView, SeriesView, Timestamp};
let observed_at = [Timestamp::from_unix_secs(10 * 60 * 60)?];
let available_at = [Timestamp::from_unix_secs(10 * 60 * 60 + 7 * 60)?];
let prices = SeriesView::new(&observed_at, &[100.0])?;
let availability = AvailabilityView::new(prices, &available_at)?;

assert_eq!(availability.available_at(), &available_at);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Construction requires one availability timestamp per observation, at or after
its observation time. Availability times may repeat or be out of order. The
view borrows the series and availability slice without allocation;
[`AvailabilityError`] identifies invalid input. Use a consistent meaning for
availability, such as provider publication or receipt by your application.

[`align::as_of_available`] matches each left observation to the latest right
observation already available at its timestamp, within a supplied maximum age.
In the example above, the price is excluded at 10:03 and eligible at 10:08
with a ten-minute limit. Age is measured from observation time, so its age at
10:08 is eight minutes.

Every left row is preserved. [`align::AvailableAsOfOutcome`] distinguishes
matches, missing preceding observations, observations not yet available, and
available observations that are too old. Matched rows include the source
index, value, observation timestamp, availability timestamp, age, and
availability delay. A late older observation cannot replace a newer eligible
observation.

[`align::as_of_backward`] uses observation timestamps only. Use
`as_of_available` when eligibility also depends on the supplied availability
times.
