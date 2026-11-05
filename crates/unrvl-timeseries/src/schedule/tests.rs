use super::*;

fn interval(start_nanos: i64, end_nanos: i64) -> TimeInterval {
    TimeInterval::new(
        Timestamp::from_unix_nanos(start_nanos),
        Timestamp::from_unix_nanos(end_nanos),
    )
    .unwrap()
}

fn session(id: &str, start_nanos: i64, end_nanos: i64) -> Session {
    Session::new(id, vec![interval(start_nanos, end_nanos)]).unwrap()
}

mod construction {
    use super::*;

    #[test]
    fn rejects_duplicate_identifiers_with_the_first_occurrence() {
        let coverage = interval(0, 10);
        let sessions = vec![
            session("regular", 0, 2),
            session("overnight", 3, 5),
            session("regular", 6, 8),
        ];
        let error = SessionSchedule::new(coverage, sessions).unwrap_err();

        assert_eq!(
            error,
            ScheduleError::DuplicateSessionId { id: "regular".into(), previous_index: 0, index: 2 },
        );
    }

    #[test]
    fn rejects_sessions_extending_beyond_either_coverage_endpoint() {
        let coverage = interval(0, 10);
        let cases = [
            ("starts before", interval(-1, 1)),
            ("ends after", interval(9, 11)),
            ("encloses coverage", interval(-1, 11)),
        ];

        for (id, opening) in cases {
            let session = Session::new(id, vec![opening]).unwrap();
            let error = SessionSchedule::new(coverage, vec![session]).unwrap_err();

            assert_eq!(
                error,
                ScheduleError::SessionOutsideCoverage {
                    id: id.into(),
                    index: 0,
                    envelope: opening,
                    coverage,
                },
            );
        }
    }

    #[test]
    fn rejects_overlapping_unordered_and_interleaved_session_envelopes() {
        let coverage = interval(0, 10);

        // The second session fits inside the first session's break, but their
        // envelopes overlap. Breaks cannot be assigned to another session.
        let with_break = Session::new("first", vec![interval(0, 2), interval(6, 8)]).unwrap();
        let cases = [
            (
                session("first", 0, 4),
                session("second", 3, 6),
                interval(0, 4),
            ),
            (
                session("first", 4, 6),
                session("second", 0, 3),
                interval(4, 6),
            ),
            (with_break, session("second", 3, 5), interval(0, 8)),
        ];

        for (first, second, previous) in cases {
            let current = second.intervals()[0];
            let error = SessionSchedule::new(coverage, vec![first, second]).unwrap_err();

            assert_eq!(
                error,
                ScheduleError::InvalidSessionSequence {
                    previous_id: "first".into(),
                    id: "second".into(),
                    previous_index: 0,
                    index: 1,
                    previous,
                    current,
                },
            );
        }
    }

    #[test]
    fn reports_the_first_invalid_session_before_later_failures() {
        let coverage = interval(0, 10);
        let sessions = vec![
            session("first", 0, 2),
            session("second", 3, 5),
            session("overlap", 4, 6),
            session("first", 8, 12),
        ];
        let error = SessionSchedule::new(coverage, sessions).unwrap_err();

        assert!(matches!(
            error,
            ScheduleError::InvalidSessionSequence { previous_index: 1, index: 2, .. }
        ));
    }

    #[test]
    fn accepts_coverage_without_sessions() {
        let coverage = interval(0, 10);
        let schedule = SessionSchedule::new(coverage, vec![]).unwrap();

        assert_eq!(schedule.coverage(), coverage);
        assert!(schedule.sessions().is_empty());
    }

    #[test]
    fn preserves_the_supplied_sessions_and_their_buffer() {
        let coverage = interval(0, 10);
        let with_break = Session::new("first", vec![interval(1, 2), interval(4, 6)]).unwrap();
        let sessions = vec![with_break, session("second", 7, 9)];
        let sessions_ptr = sessions.as_ptr();
        let schedule = SessionSchedule::new(coverage, sessions).unwrap();

        assert_eq!(schedule.coverage(), coverage);
        assert_eq!(schedule.sessions().as_ptr(), sessions_ptr);
        assert_eq!(schedule.sessions()[0].id(), "first");
        assert_eq!(schedule.sessions()[1].id(), "second");
        assert_eq!(
            schedule.sessions()[0].intervals(),
            [interval(1, 2), interval(4, 6)]
        );
    }

    #[test]
    fn accepts_adjacent_envelopes_at_the_coverage_boundaries() {
        let coverage = interval(0, 10);
        let sessions = vec![session("first", 0, 5), session("second", 5, 10)];
        let schedule = SessionSchedule::new(coverage, sessions).unwrap();

        assert_eq!(schedule.sessions().len(), 2);
    }

    #[test]
    fn compares_opaque_identifiers_exactly() {
        let coverage = interval(0, 3);
        let sessions = vec![
            session("venue", 0, 1),
            session("Venue", 1, 2),
            session(" venue ", 2, 3),
        ];
        let schedule = SessionSchedule::new(coverage, sessions).unwrap();
        let identifiers: Vec<_> = schedule.sessions().iter().map(Session::id).collect();

        assert_eq!(identifiers, ["venue", "Venue", " venue "]);
    }

    #[test]
    fn supports_overnight_sessions_and_the_full_timestamp_span() {
        let coverage = TimeInterval::new(Timestamp::MIN, Timestamp::MAX).unwrap();
        let sessions = vec![
            session("before", i64::MIN, -1),
            session("across midnight", -1, 1),
            session("after", 1, i64::MAX),
        ];
        let schedule = SessionSchedule::new(coverage, sessions).unwrap();

        assert_eq!(schedule.coverage(), coverage);
        assert_eq!(schedule.sessions().len(), 3);
        assert_eq!(
            schedule.open_duration(coverage),
            Some(Duration::from_nanos(u64::MAX))
        );
    }

    #[test]
    fn describes_invalid_sequences_with_both_identifiers_and_units() {
        let coverage = interval(0, 10);
        let sessions = vec![session("first", 0, 3), session("second", 2, 5)];
        let error = SessionSchedule::new(coverage, sessions).unwrap_err();

        assert_eq!(
            error.to_string(),
            "session \"second\" at index 1 starts at 2 ns, \
             before session \"first\" at index 0 ends at 3 ns",
        );
    }
}

mod lookup {
    use super::*;

    #[test]
    fn finds_the_open_session_across_a_break_and_shared_boundary() {
        let coverage = interval(0, 10);
        let split = Session::new("split", vec![interval(1, 3), interval(4, 6)]).unwrap();
        let sessions = vec![split, session("later", 6, 9)];
        let schedule = SessionSchedule::new(coverage, sessions).unwrap();

        let cases = [
            (-1, None),
            (0, None),
            (1, Some("split")),
            (3, None),
            (4, Some("split")),
            (6, Some("later")),
            (9, None),
            (10, None),
        ];

        for (nanos, expected) in cases {
            let session = schedule.session_at(Timestamp::from_unix_nanos(nanos));
            assert_eq!(session.map(Session::id), expected, "{nanos} ns");
        }
    }

    #[test]
    fn distinguishes_open_periods_breaks_gaps_and_uncovered_time() {
        let coverage = interval(0, 10);
        let split = Session::new("split", vec![interval(1, 3), interval(4, 6)]).unwrap();
        let sessions = vec![split, session("later", 7, 9)];
        let schedule = SessionSchedule::new(coverage, sessions).unwrap();

        let cases = [
            (-1, ScheduleStatus::OutsideCoverage),
            (0, ScheduleStatus::Closed),
            (1, ScheduleStatus::Open),
            (3, ScheduleStatus::Closed),
            (4, ScheduleStatus::Open),
            (6, ScheduleStatus::Closed),
            (7, ScheduleStatus::Open),
            (9, ScheduleStatus::Closed),
            (10, ScheduleStatus::OutsideCoverage),
        ];

        for (nanos, expected) in cases {
            assert_eq!(
                schedule.status_at(Timestamp::from_unix_nanos(nanos)),
                expected,
                "{nanos} ns"
            );
        }
    }

    #[test]
    fn stays_open_at_the_boundary_between_adjacent_sessions() {
        let coverage = interval(0, 10);
        let sessions = vec![session("first", 0, 5), session("second", 5, 10)];
        let schedule = SessionSchedule::new(coverage, sessions).unwrap();

        assert_eq!(
            schedule.status_at(Timestamp::from_unix_nanos(5)),
            ScheduleStatus::Open
        );
        assert_eq!(
            schedule.status_at(Timestamp::from_unix_nanos(10)),
            ScheduleStatus::OutsideCoverage
        );
    }

    #[test]
    fn an_empty_schedule_is_closed_within_coverage() {
        let coverage = interval(0, 10);
        let schedule = SessionSchedule::new(coverage, vec![]).unwrap();

        assert_eq!(
            schedule.status_at(Timestamp::UNIX_EPOCH),
            ScheduleStatus::Closed
        );
    }
}

mod navigation {
    use super::*;

    #[test]
    fn navigates_sessions_by_first_open_and_final_close() {
        let coverage = interval(0, 12);
        let split = Session::new("split", vec![interval(3, 5), interval(6, 8)]).unwrap();
        let sessions = vec![session("first", 1, 3), split, session("last", 9, 11)];
        let schedule = SessionSchedule::new(coverage, sessions).unwrap();

        let cases = [
            (-1, None, Some("first")),
            (1, None, Some("split")),
            (3, Some("first"), Some("last")),
            (5, Some("first"), Some("last")),
            (6, Some("first"), Some("last")),
            (8, Some("split"), Some("last")),
            (9, Some("split"), None),
            (11, Some("last"), None),
            (12, Some("last"), None),
        ];

        for (nanos, expected_previous, expected_next) in cases {
            let timestamp = Timestamp::from_unix_nanos(nanos);
            let previous = schedule.previous_session(timestamp).map(Session::id);
            let next = schedule.next_session(timestamp).map(Session::id);

            assert_eq!(previous, expected_previous, "previous at {nanos} ns");
            assert_eq!(next, expected_next, "next at {nanos} ns");
        }
    }

    #[test]
    fn an_empty_schedule_has_no_previous_or_next_session() {
        let coverage = interval(0, 10);
        let schedule = SessionSchedule::new(coverage, vec![]).unwrap();

        assert!(schedule.previous_session(Timestamp::UNIX_EPOCH).is_none());
        assert!(schedule.next_session(Timestamp::UNIX_EPOCH).is_none());
    }
}

mod open_duration {
    use super::*;

    #[test]
    fn sums_open_time_within_covered_ranges() {
        let coverage = interval(-2, 15);
        let split = Session::new("split", vec![interval(-1, 2), interval(4, 7)]).unwrap();
        let sessions = vec![split, session("adjacent", 7, 10), session("last", 12, 14)];
        let schedule = SessionSchedule::new(coverage, sessions).unwrap();

        let cases = [
            (coverage, 11),
            (interval(1, 8), 5),
            (interval(2, 4), 0),
            (interval(7, 9), 2),
            (interval(10, 12), 0),
            (interval(14, 15), 0),
        ];

        for (range, expected_nanos) in cases {
            assert_eq!(
                schedule.open_duration(range),
                Some(Duration::from_nanos(expected_nanos)),
                "range {range:?}"
            );
        }
    }

    #[test]
    fn open_duration_requires_the_whole_range_to_be_covered() {
        let coverage = interval(0, 10);
        let sessions = vec![session("regular", 1, 9)];
        let schedule = SessionSchedule::new(coverage, sessions).unwrap();

        for range in [interval(-1, 5), interval(5, 11), interval(10, 12)] {
            assert_eq!(schedule.open_duration(range), None, "range {range:?}");
        }
    }

    #[test]
    fn open_duration_is_zero_for_a_covered_range_without_sessions() {
        let coverage = interval(0, 10);
        let schedule = SessionSchedule::new(coverage, vec![]).unwrap();

        assert_eq!(schedule.open_duration(coverage), Some(Duration::ZERO));
    }
}

#[test]
fn labels_final_closes_at_shared_boundaries_and_the_end_of_coverage() {
    let coverage = interval(0, 10);
    let split = Session::new("split", vec![interval(1, 3), interval(4, 6)]).unwrap();
    let sessions = vec![split, session("later", 6, 10)];
    let schedule = SessionSchedule::new(coverage, sessions).unwrap();

    let labels: Vec<_> = schedule
        .close_labels()
        .map(|(session, close)| (session.id(), close.as_unix_nanos()))
        .collect();

    assert_eq!(labels, [("split", 6), ("later", 10)]);
}
