use std::time::Duration;

use unrvl_timeseries::{Series, SeriesError, SeriesView, Timestamp, TimestampError};

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
    let result: Result<SeriesView<'_>, SeriesError> = SeriesView::new(&timestamps, &values);
    let series = result.unwrap();

    let _: &[Timestamp] = series.timestamps();
    let _: &[f64] = series.values();
    let _ = series.len();
    let _ = series.is_empty();
    let _: SeriesView<'_> = series.between(..).unwrap();
}

#[test]
fn owned_series_are_available_externally() {
    let timestamps = [0, 1, 3].map(Timestamp::from_unix_nanos).to_vec();
    let values = vec![-2.0, 0.0, 5.0];
    let result: Result<Series, SeriesError> = Series::new(timestamps, values);
    let series = result.unwrap();

    let _: &[Timestamp] = series.timestamps();
    let _: &[f64] = series.values();
    let _ = series.len();
    let _ = series.is_empty();
    let _: SeriesView<'_> = series.view();
}
