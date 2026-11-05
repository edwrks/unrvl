use super::*;

mod from_unix_secs {
    use super::*;

    #[test]
    fn rejects_unrepresentable_seconds() {
        // First whole-second magnitude that overflows when converted to
        // nanoseconds.
        let first_overflowing = i64::MAX / 1_000_000_000 + 1;
        let cases = [i64::MIN, -first_overflowing, first_overflowing, i64::MAX];

        for value in cases {
            let expected = TimestampError::SecondsOutOfRange { value };

            assert_eq!(Timestamp::from_unix_secs(value), Err(expected));
        }
    }

    #[test]
    fn converts_seconds_exactly_including_range_boundaries() {
        // Largest whole-second magnitude that still fits in signed
        // nanoseconds.
        let max_secs_in_range = i64::MAX / 1_000_000_000;
        let max_secs_in_nanos = 9_223_372_036_000_000_000;
        let cases = [
            (-max_secs_in_range, -max_secs_in_nanos),
            (-1, -1_000_000_000),
            (0, 0),
            (1, 1_000_000_000),
            (max_secs_in_range, max_secs_in_nanos),
        ];

        for (value, expected) in cases {
            let timestamp = Timestamp::from_unix_secs(value).unwrap();

            assert_eq!(timestamp.as_unix_nanos(), expected);
        }
    }
}

mod from_unix_millis {
    use super::*;

    #[test]
    fn rejects_unrepresentable_milliseconds() {
        // First whole-millisecond magnitude that overflows in nanoseconds.
        let first_overflowing = i64::MAX / 1_000_000 + 1;
        let cases = [i64::MIN, -first_overflowing, first_overflowing, i64::MAX];

        for value in cases {
            let expected = TimestampError::MillisecondsOutOfRange { value };

            assert_eq!(Timestamp::from_unix_millis(value), Err(expected));
        }
    }

    #[test]
    fn converts_milliseconds_exactly_including_range_boundaries() {
        // Largest whole-millisecond magnitude that still fits in signed
        // nanoseconds.
        let max_millis_in_range = i64::MAX / 1_000_000;
        let max_millis_in_nanos = 9_223_372_036_854_000_000;
        let cases = [
            (-max_millis_in_range, -max_millis_in_nanos),
            (-1, -1_000_000),
            (0, 0),
            (1, 1_000_000),
            (max_millis_in_range, max_millis_in_nanos),
        ];

        for (value, expected) in cases {
            let timestamp = Timestamp::from_unix_millis(value).unwrap();

            assert_eq!(timestamp.as_unix_nanos(), expected);
        }
    }
}

mod from_unix_micros {
    use super::*;

    #[test]
    fn rejects_unrepresentable_microseconds() {
        // First whole-microsecond magnitude that overflows in nanoseconds.
        let first_overflowing = i64::MAX / 1_000 + 1;
        let cases = [i64::MIN, -first_overflowing, first_overflowing, i64::MAX];

        for value in cases {
            let expected = TimestampError::MicrosecondsOutOfRange { value };

            assert_eq!(Timestamp::from_unix_micros(value), Err(expected));
        }
    }

    #[test]
    fn converts_microseconds_exactly_including_range_boundaries() {
        // Largest whole-microsecond magnitude that still fits in signed
        // nanoseconds.
        let max_micros_in_range = i64::MAX / 1_000;
        let max_micros_in_nanos = 9_223_372_036_854_775_000;
        let cases = [
            (-max_micros_in_range, -max_micros_in_nanos),
            (-1, -1_000),
            (0, 0),
            (1, 1_000),
            (max_micros_in_range, max_micros_in_nanos),
        ];

        for (value, expected) in cases {
            let timestamp = Timestamp::from_unix_micros(value).unwrap();

            assert_eq!(timestamp.as_unix_nanos(), expected);
        }
    }
}

mod checked_add {
    use super::*;

    #[test]
    fn returns_none_when_the_result_is_out_of_range() {
        // MIN to the epoch is 2^63 ns, one more than i64::MAX.
        let min_to_epoch = Duration::from_nanos(1_u64 << 63);

        // One nanosecond beyond the full i64 timestamp span.
        let beyond_full_span = Duration::from_nanos(u64::MAX) + Duration::from_nanos(1);

        let cases = [
            (Timestamp::MAX, Duration::from_nanos(1)),
            (Timestamp::UNIX_EPOCH, min_to_epoch),
            (Timestamp::MIN, beyond_full_span),
            (Timestamp::MIN, Duration::MAX),
        ];

        for (timestamp, duration) in cases {
            assert_eq!(timestamp.checked_add(duration), None);
        }
    }

    #[test]
    fn adds_exactly_across_epoch_and_range_boundaries() {
        // -1.500000001 s + 2.000000002 s = 0.500000001 s.
        let before_epoch_nanos = -1_500_000_001;
        let after_epoch_nanos = 500_000_001;
        let across_epoch = Duration::new(2, 2);

        // MIN to the epoch is 2^63 ns; MIN to MAX is 2^64 - 1 ns.
        let min_to_epoch = Duration::from_nanos(1_u64 << 63);
        let full_span = Duration::from_nanos(u64::MAX);

        let cases = [
            (before_epoch_nanos, across_epoch, after_epoch_nanos),
            (-1, Duration::from_nanos(1), 0),
            (0, Duration::from_nanos(1), 1),
            (i64::MIN, Duration::from_nanos(1), i64::MIN + 1),
            (i64::MAX - 1, Duration::from_nanos(1), i64::MAX),
            (i64::MIN, min_to_epoch, 0),
            (i64::MIN, full_span, i64::MAX),
        ];

        for (start, duration, expected) in cases {
            let timestamp = Timestamp::from_unix_nanos(start);
            let expected = Timestamp::from_unix_nanos(expected);

            assert_eq!(timestamp.checked_add(duration), Some(expected));
        }
    }

    #[test]
    fn zero_preserves_the_timestamp_including_range_limits() {
        let cases = [i64::MIN, -1, 0, 1, i64::MAX];

        for value in cases {
            let timestamp = Timestamp::from_unix_nanos(value);

            assert_eq!(timestamp.checked_add(Duration::ZERO), Some(timestamp));
        }
    }
}

mod checked_sub {
    use super::*;

    #[test]
    fn returns_none_when_the_result_is_out_of_range() {
        // Subtracting 2^63 ns from the epoch reaches MIN; starting at -1
        // overflows.
        let min_to_epoch = Duration::from_nanos(1_u64 << 63);
        // One nanosecond beyond the full i64 timestamp span.
        let beyond_full_span = Duration::from_nanos(u64::MAX) + Duration::from_nanos(1);

        let cases = [
            (Timestamp::MIN, Duration::from_nanos(1)),
            (Timestamp::from_unix_nanos(-1), min_to_epoch),
            (Timestamp::MAX, beyond_full_span),
            (Timestamp::MAX, Duration::MAX),
        ];

        for (timestamp, duration) in cases {
            assert_eq!(timestamp.checked_sub(duration), None);
        }
    }

    #[test]
    fn subtracts_exactly_across_epoch_and_range_boundaries() {
        // 0.500000001 s - 2.000000002 s = -1.500000001 s.
        let before_epoch_nanos = -1_500_000_001;
        let after_epoch_nanos = 500_000_001;
        let across_epoch = Duration::new(2, 2);

        // Epoch to MIN subtracts 2^63 ns; MAX to MIN subtracts 2^64 - 1 ns.
        let min_to_epoch = Duration::from_nanos(1_u64 << 63);
        let full_span = Duration::from_nanos(u64::MAX);

        let cases = [
            (after_epoch_nanos, across_epoch, before_epoch_nanos),
            (0, Duration::from_nanos(1), -1),
            (1, Duration::from_nanos(1), 0),
            (i64::MIN + 1, Duration::from_nanos(1), i64::MIN),
            (i64::MAX, Duration::from_nanos(1), i64::MAX - 1),
            (0, min_to_epoch, i64::MIN),
            (i64::MAX, full_span, i64::MIN),
        ];

        for (start, duration, expected) in cases {
            let timestamp = Timestamp::from_unix_nanos(start);
            let expected = Timestamp::from_unix_nanos(expected);

            assert_eq!(timestamp.checked_sub(duration), Some(expected));
        }
    }

    #[test]
    fn zero_preserves_the_timestamp_including_range_limits() {
        let cases = [i64::MIN, -1, 0, 1, i64::MAX];

        for value in cases {
            let timestamp = Timestamp::from_unix_nanos(value);

            assert_eq!(timestamp.checked_sub(Duration::ZERO), Some(timestamp));
        }
    }
}

mod checked_duration_since {
    use super::*;

    #[test]
    fn returns_none_for_reversed_timestamps() {
        let cases = [(i64::MIN, i64::MAX), (-1, 0), (0, 1)];

        for (later, earlier) in cases {
            let timestamp = Timestamp::from_unix_nanos(later);
            let earlier = Timestamp::from_unix_nanos(earlier);

            assert_eq!(timestamp.checked_duration_since(earlier), None);
        }
    }

    #[test]
    fn measures_exactly_across_epoch_and_range_boundaries() {
        // The fractional-second observations are 2 seconds and 2
        // nanoseconds apart.
        let before_epoch_nanos = -1_500_000_001;
        let after_epoch_nanos = 500_000_001;
        let across_epoch = Duration::new(2, 2);

        // Independently written as seconds/subseconds: 2^63, 2^63 - 1,
        // and 2^64 - 1 nanoseconds, respectively.
        let min_to_epoch = Duration::new(9_223_372_036, 854_775_808);
        let epoch_to_max = Duration::new(9_223_372_036, 854_775_807);
        let full_span = Duration::new(18_446_744_073, 709_551_615);

        let cases = [
            (after_epoch_nanos, before_epoch_nanos, across_epoch),
            (0, -1, Duration::from_nanos(1)),
            (1, 0, Duration::from_nanos(1)),
            (i64::MIN + 1, i64::MIN, Duration::from_nanos(1)),
            (i64::MAX, i64::MAX - 1, Duration::from_nanos(1)),
            (0, i64::MIN, min_to_epoch),
            (i64::MAX, 0, epoch_to_max),
            (i64::MAX, i64::MIN, full_span),
        ];

        for (later, earlier, expected) in cases {
            let timestamp = Timestamp::from_unix_nanos(later);
            let earlier = Timestamp::from_unix_nanos(earlier);

            assert_eq!(timestamp.checked_duration_since(earlier), Some(expected));
        }
    }

    #[test]
    fn equal_timestamps_have_zero_elapsed_duration() {
        let cases = [i64::MIN, -1, 0, 1, i64::MAX];

        for value in cases {
            let timestamp = Timestamp::from_unix_nanos(value);

            assert_eq!(
                timestamp.checked_duration_since(timestamp),
                Some(Duration::ZERO),
            );
        }
    }
}

#[test]
fn nanoseconds_round_trip_without_precision_loss() {
    // 2^53 + 1 is the first positive integer that f64 cannot represent
    // exactly.
    let beyond_f64_precision = (1_i64 << 53) + 1;
    let cases = [
        i64::MIN,
        i64::MIN + 1,
        -beyond_f64_precision,
        -1,
        0,
        1,
        beyond_f64_precision,
        i64::MAX - 1,
        i64::MAX,
    ];

    for value in cases {
        let timestamp = Timestamp::from_unix_nanos(value);

        assert_eq!(timestamp.as_unix_nanos(), value);
    }
}

#[test]
fn constants_identify_epoch_and_range_limits() {
    assert_eq!(Timestamp::UNIX_EPOCH.as_unix_nanos(), 0);
    assert_eq!(Timestamp::MIN.as_unix_nanos(), i64::MIN);
    assert_eq!(Timestamp::MAX.as_unix_nanos(), i64::MAX);
}
