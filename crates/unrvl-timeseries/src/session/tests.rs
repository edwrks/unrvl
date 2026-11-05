use super::*;

fn interval(start_nanos: i64, end_nanos: i64) -> TimeInterval {
    TimeInterval::new(
        Timestamp::from_unix_nanos(start_nanos),
        Timestamp::from_unix_nanos(end_nanos),
    )
    .unwrap()
}

#[test]
fn rejects_an_empty_identifier() {
    let error = Session::new("", vec![interval(0, 1)]).unwrap_err();

    assert_eq!(error, SessionError::EmptyId);
}

#[test]
fn rejects_empty_intervals_with_the_session_identifier() {
    let error = Session::new("regular", vec![]).unwrap_err();

    assert_eq!(error, SessionError::EmptyIntervals { id: "regular".into() });
}

#[test]
fn rejects_overlapping_unordered_and_duplicate_intervals() {
    let cases = [
        ("overlap", [interval(0, 4), interval(3, 6)]),
        ("out of order", [interval(4, 6), interval(0, 3)]),
        ("duplicate", [interval(0, 3), interval(0, 3)]),
    ];

    for (id, intervals) in cases {
        let error = Session::new(id, intervals.to_vec()).unwrap_err();

        assert_eq!(
            error,
            SessionError::InvalidIntervalSequence {
                id: id.into(),
                previous_index: 0,
                index: 1,
                previous: intervals[0],
                current: intervals[1],
            },
        );
    }
}

#[test]
fn reports_the_first_invalid_pair_after_valid_intervals() {
    let intervals = [
        interval(0, 2),
        interval(3, 5),
        interval(4, 6),
        interval(4, 7),
    ];
    let error = Session::new("regular", intervals.to_vec()).unwrap_err();

    assert_eq!(
        error,
        SessionError::InvalidIntervalSequence {
            id: "regular".into(),
            previous_index: 1,
            index: 2,
            previous: intervals[1],
            current: intervals[2],
        },
    );
}

#[test]
fn preserves_the_owned_identifier_and_interval_buffers() {
    let id = String::from("  venue:overnight  ");

    let openings = [interval(-2, 0), interval(1, 3)];
    let intervals = openings.to_vec();

    let id_ptr = id.as_ptr();
    let intervals_ptr = intervals.as_ptr();

    let session = Session::new(id, intervals).unwrap();

    assert_eq!(session.id(), "  venue:overnight  ");
    assert_eq!(session.id().as_ptr(), id_ptr);
    assert_eq!(session.intervals(), openings);
    assert_eq!(session.intervals().as_ptr(), intervals_ptr);
}

#[test]
fn membership_respects_openings_breaks_and_the_final_close() {
    let session = Session::new("regular", vec![interval(0, 3), interval(5, 8)]).unwrap();
    let cases = [
        (-1, false),
        (0, true),
        (2, true),
        (3, false),
        (4, false),
        (5, true),
        (7, true),
        (8, false),
        (9, false),
    ];

    for (nanos, expected) in cases {
        let timestamp = Timestamp::from_unix_nanos(nanos);

        assert_eq!(
            session.is_open_at(timestamp),
            expected,
            "timestamp={nanos} ns"
        );
    }
}

#[test]
fn accepts_adjacent_opening_intervals() {
    let session = Session::new("continuous", vec![interval(0, 3), interval(3, 6)]).unwrap();

    assert_eq!(session.intervals().len(), 2);
    assert!(session.is_open_at(Timestamp::from_unix_nanos(3)));
}

#[test]
fn supports_a_single_opening_across_utc_midnight() {
    // The Unix epoch is midnight UTC, between these opening and closing times.
    let open = Timestamp::from_unix_secs(-3600).unwrap();
    let close = Timestamp::from_unix_secs(3600).unwrap();
    let opening = TimeInterval::new(open, close).unwrap();
    let session = Session::new("overnight", vec![opening]).unwrap();

    assert_eq!(session.intervals(), [opening]);
    assert!(session.is_open_at(open));
    assert!(session.is_open_at(Timestamp::UNIX_EPOCH));
    assert!(!session.is_open_at(close));
}

#[test]
fn supports_membership_across_the_full_timestamp_span() {
    let opening = TimeInterval::new(Timestamp::MIN, Timestamp::MAX).unwrap();
    let session = Session::new("full span", vec![opening]).unwrap();

    assert!(session.is_open_at(Timestamp::MIN));
    assert!(session.is_open_at(Timestamp::UNIX_EPOCH));
    assert!(!session.is_open_at(Timestamp::MAX));
}

#[test]
fn describes_invalid_sequences_with_the_session_and_units() {
    let error = Session::new("regular", vec![interval(0, 3), interval(2, 5)]).unwrap_err();

    assert_eq!(
        error.to_string(),
        "session \"regular\" open period 1 starts at 2 ns, before period 0 ends at 3 ns",
    );
}
