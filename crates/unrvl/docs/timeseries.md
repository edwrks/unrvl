Exact timestamps, validated numerical series, supplied sessions, and temporal
alignment for time-series analysis. Enable the `timeseries` feature to use this
module.

[`Timestamp`] stores signed Unix nanoseconds. Use its checked constructors for
seconds, milliseconds, and microseconds, and [`std::time::Duration`] for elapsed
time. Date parsing, timezones, and holiday calendars belong at the application
boundary.

# Working with Series

[`Series`] owns timestamp and value vectors; [`SeriesView`] borrows existing
slices. Both require matching lengths, finite values, and strictly increasing
timestamps. Empty series, signed values, and irregular spacing are valid.

```rust
# use unrvl::timeseries::{Series, Timestamp};
let timestamps = vec![
    Timestamp::from_unix_secs(0)?,
    Timestamp::from_unix_secs(60)?,
    Timestamp::from_unix_secs(180)?,
];
let values = vec![100.0, 102.0, 101.0];
let series = Series::new(timestamps, values)?;
let view = series.view();

let start = Timestamp::from_unix_secs(60)?;
let selected = view.between(start..)?;
assert_eq!(selected.values(), &[102.0, 101.0]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Views and time-range selections borrow the original buffers without copying or
revalidating observations. Their value slices can be passed directly to
statistical functions. Construction establishes structure; sampling checks
establish whether expected observations are present.

# Working with Sessions

[`TimeInterval`] represents the half-open interval `[start, end)`: the start is
included and the end is excluded. [`Session`] groups one or more ordered open
periods under a caller-supplied identifier. Gaps between its periods represent
breaks.

[`SessionSchedule`] describes all supplied sessions within an explicit coverage
interval. It distinguishes scheduled open time, scheduled closure, and time
outside its knowledge:

```rust
# use unrvl::timeseries::{ScheduleStatus, Session, SessionSchedule, TimeInterval, Timestamp};
let open = Timestamp::from_unix_secs(9 * 3600)?;
let close = Timestamp::from_unix_secs(16 * 3600)?;
let coverage_end = Timestamp::from_unix_secs(17 * 3600)?;

let session = Session::new("regular", vec![TimeInterval::new(open, close)?])?;
let coverage = TimeInterval::new(open, coverage_end)?;
let schedule = SessionSchedule::new(coverage, vec![session])?;

assert_eq!(schedule.status_at(open), ScheduleStatus::Open);
assert_eq!(schedule.status_at(close), ScheduleStatus::Closed);
assert_eq!(schedule.status_at(coverage_end), ScheduleStatus::OutsideCoverage);
# Ok::<(), Box<dyn std::error::Error>>(())
```

The schedule supports session lookup, navigation, and scheduled-open duration.
It describes planned hours; construction cannot detect omitted sessions or
establish that a data source captured every observation.

[`SessionSchedule::close_labels`] supplies the timestamps for expected closing
observations. A close label identifies the session that ended, including when it
equals the end of coverage. Open-period membership remains half-open.

# Checking Sampling

Use [`cadence::FixedGrid`] and [`cadence::GridCoverage`] when observations are
expected at regular elapsed-time timestamps, such as every minute. A tolerance
allows small timestamp deviations. Coverage preserves the original observations
and reports missing grid slots as compact gap ranges.

Use [`cadence::SessionCloseCoverage`] when you expect one observation at each
supplied session's exact final close.

Missing samples are valid coverage outcomes. Require `is_complete()` when your
calculation needs every expected observation. These operations check supplied
observations without filling gaps or constructing closing prices.

# Grouping Observations

Use [`resample::FixedBuckets`] to group observations into elapsed-time
intervals, such as hourly buckets containing minute prices. A grid checks for
observations near expected timestamps; a bucket accepts every observation inside
its interval.

An irregular series can be assigned directly to fixed buckets. Grid checks and
session grouping are optional operations for different sampling and scheduling
requirements.

[`resample::FixedBuckets::assign`] borrows a series for iteration over all
buckets, including empty ones. Each [`resample::Bucket`] provides its original
observations, source index range, full interval, and the portion inside the
analysis bounds. Its value slice can be passed to a statistic.

Use [`resample::SessionAssignment`] to group observations by an existing
[`SessionSchedule`]. It yields one [`resample::SessionBucket`] per session whose
envelope intersects the analysis bounds, retaining identity, open periods,
source ranges, and empty or partial sessions. Bounds must lie within known
schedule coverage, and observations during breaks or other closed time are
rejected. An envelope spans from the first opening to the final close, including
any breaks; [`SessionSchedule::open_duration`] measures scheduled open time.

Choose a start or end output label with [`resample::BucketLabel`]. Partial edge
buckets remain identified, and a populated bucket does not establish complete
sampling.

[`first`](resample::BucketAssignment::first) and
[`last`](resample::BucketAssignment::last) select the earliest and latest
observation in each bucket. [`min`](resample::BucketAssignment::min) and
[`max`](resample::BucketAssignment::max) select the smallest and largest values,
choosing the earliest observation when values tie. Results retain the output
label, bucket context, and the selected observation's original timestamp and
source index. Empty buckets have no selected observation.

[`mean`](resample::BucketAssignment::mean) calculates an arithmetic mean with
equal weight per observation; [`sum`](resample::BucketAssignment::sum) totals
observed values, such as signed event amounts. Elapsed time does not weight
either operation. [`resample::AggregateOutcome`] distinguishes empty buckets,
finite values, and numerical failures; an empty bucket remains distinct from an
observed total of zero.

Both assignment types offer these six methods. Fixed-bucket results use
[`resample::BucketSelection`] and [`resample::BucketAggregate`]; session results
use [`resample::SessionSelection`] and [`resample::SessionAggregate`], retaining
the original schedule index and a borrowed session reference.

# Aligning Series

Choose alignment according to what makes two observations comparable:

- [`align::inner`] matches exactly equal timestamps.
- [`align::inner_slots`] matches equal labels from checked grids, preserving
  both original timestamps and their deviations.
- [`align::as_of_backward`] finds the latest right observation at or before each
  left timestamp, within an inclusive maximum age.
- [`align::as_of_available`] also requires that the right observation was
  available by the left timestamp.

Results retain source indices and values, with unmatched counts or explicit
lookup outcomes. Stale as-of outcomes retain the candidate's right index and age
for diagnostics. Slot matching establishes sampling positions; historical
information eligibility requires availability data.

# Recording Availability

[`AvailabilityView`] pairs observation timestamps with caller-supplied times at
which those observations became usable. For example, a price observed at 10:00
and published at 10:07 is unavailable to a decision at 10:03.

[`AvailabilityView::between`] selects by observation time and keeps the paired
availability timestamps, including those outside the selected range.

Pass this view to [`align::as_of_available`] to enforce that distinction. Match
age is measured from observation time: at 10:08 the price is eight minutes old,
with a seven-minute availability delay. Use a consistent meaning for supplied
availability, such as provider publication or receipt by your application.

