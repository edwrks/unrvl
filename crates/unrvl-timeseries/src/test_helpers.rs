use crate::{Session, TimeInterval, Timestamp};

/// Creates a fixture timestamp from Unix nanoseconds.
pub fn timestamp(nanos: i64) -> Timestamp {
    Timestamp::from_unix_nanos(nanos)
}

/// Creates a valid half-open interval from Unix nanoseconds.
#[track_caller]
pub fn interval(start_nanos: i64, end_nanos: i64) -> TimeInterval {
    TimeInterval::new(timestamp(start_nanos), timestamp(end_nanos)).unwrap()
}

/// Creates a valid session with one open interval, using Unix nanoseconds.
#[track_caller]
pub fn session(id: &str, start_nanos: i64, end_nanos: i64) -> Session {
    Session::new(id, vec![interval(start_nanos, end_nanos)]).unwrap()
}
