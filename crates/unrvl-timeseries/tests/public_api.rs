use std::time::Duration;

use unrvl_timeseries::{
    Series, SeriesView, Session, SessionSchedule, TimeInterval, Timestamp, TimestampError,
};

#[test]
fn timestamps_and_checked_conversions_are_available_externally() {
    const EPOCH: Timestamp = Timestamp::from_unix_nanos(0);
    const SECOND: Result<Timestamp, TimestampError> = Timestamp::from_unix_secs(1);

    let _ = Timestamp::MIN;
    let _ = Timestamp::MAX;
    let _ = Timestamp::UNIX_EPOCH;

    let _ = EPOCH.as_unix_nanos();
    let _ = SECOND.unwrap();
    let _ = Timestamp::from_unix_millis(1).unwrap();
    let _ = Timestamp::from_unix_micros(1).unwrap();
}

#[test]
fn checked_timestamp_arithmetic_is_available_externally() {
    let start = Timestamp::UNIX_EPOCH;
    let duration = Duration::from_secs(1);
    let end = start.checked_add(duration).unwrap();

    let _ = end.checked_sub(duration).unwrap();
    let _ = end.checked_duration_since(start).unwrap();
}

#[test]
fn series_views_are_available_externally() {
    let timestamps = [0, 1, 3].map(Timestamp::from_unix_nanos);
    let values = [-2.0, 0.0, 5.0];
    let result = SeriesView::new(&timestamps, &values);
    let series = result.unwrap();

    let _ = series.timestamps();
    let _ = series.values();
    let _ = series.len();
    let _ = series.is_empty();
    let _ = series.between(..).unwrap();
}

#[test]
fn owned_series_are_available_externally() {
    let timestamps = [0, 1, 3].map(Timestamp::from_unix_nanos).to_vec();
    let values = vec![-2.0, 0.0, 5.0];
    let result = Series::new(timestamps, values);
    let series = result.unwrap();

    let _ = series.timestamps();
    let _ = series.values();
    let _ = series.len();
    let _ = series.is_empty();
    let _ = series.view();
}

#[test]
fn time_intervals_are_available_externally() {
    let start = Timestamp::UNIX_EPOCH;
    let end = Timestamp::from_unix_nanos(2);
    let result = TimeInterval::new(start, end);
    let interval = result.unwrap();

    let _ = interval.start();
    let _ = interval.end();
    let _ = interval.contains(start);
    let _ = interval.duration();

    let timestamps = [start, end];
    let values = [1.0, 2.0];
    let series = SeriesView::new(&timestamps, &values).unwrap();
    let _ = series.between(interval).unwrap();
}

#[test]
fn sessions_are_available_externally() {
    let start = Timestamp::UNIX_EPOCH;
    let end = Timestamp::from_unix_nanos(2);
    let opening = TimeInterval::new(start, end).unwrap();
    let result = Session::new("regular", vec![opening]);
    let session = result.unwrap();

    let _ = session.id();
    let _ = session.intervals();
    let _ = session.is_open_at(start);
}

#[test]
fn session_schedules_are_available_externally() {
    let start = Timestamp::UNIX_EPOCH;
    let end = Timestamp::from_unix_nanos(2);
    let coverage = TimeInterval::new(start, end).unwrap();
    let session = Session::new("regular", vec![coverage]).unwrap();
    let result = SessionSchedule::new(coverage, vec![session]);
    let schedule = result.unwrap();

    let _ = schedule.coverage();
    let _ = schedule.sessions();
    let _ = schedule.close_labels().next();
    let _ = schedule.session_at(start);
    let _ = schedule.previous_session(end);
    let _ = schedule.next_session(start);
    let _ = schedule.status_at(start);
    let _ = schedule.open_duration(coverage);
}
