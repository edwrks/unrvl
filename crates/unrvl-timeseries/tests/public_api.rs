use std::time::Duration;

use unrvl_timeseries::{Timestamp, TimestampError};

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
