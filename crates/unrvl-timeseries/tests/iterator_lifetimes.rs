use std::time::Duration;

use unrvl_timeseries::resample::{BucketLabel, FixedBuckets, SessionAssignment};
use unrvl_timeseries::{SeriesView, Session, SessionSchedule, TimeInterval, Timestamp};

#[test]
fn fixed_resampling_iterators_outlive_the_assignment() {
    let start = Timestamp::UNIX_EPOCH;
    let end = Timestamp::from_unix_secs(120).unwrap();
    let bounds = TimeInterval::new(start, end).unwrap();
    let buckets = FixedBuckets::new(start, Duration::from_secs(60), bounds).unwrap();
    let timestamps = [start];
    let values = [100.0];
    let series = SeriesView::new(&timestamps, &values).unwrap();

    let (groups, first, last, lows, highs, means, sums) = {
        let assigned = buckets.assign(series).unwrap();

        (
            assigned.iter(),
            assigned.first(BucketLabel::Start),
            assigned.last(BucketLabel::End),
            assigned.min(BucketLabel::End),
            assigned.max(BucketLabel::End),
            assigned.mean(BucketLabel::End),
            assigned.sum(BucketLabel::End),
        )
    };

    let counts = [
        groups.count(),
        first.count(),
        last.count(),
        lows.count(),
        highs.count(),
        means.count(),
        sums.count(),
    ];

    assert_eq!(counts, [2; 7]);
}

#[test]
fn session_resampling_iterators_outlive_the_assignment() {
    let start = Timestamp::UNIX_EPOCH;
    let boundary = Timestamp::from_unix_secs(60).unwrap();
    let end = Timestamp::from_unix_secs(120).unwrap();
    let coverage = TimeInterval::new(start, end).unwrap();
    let first_open = TimeInterval::new(start, boundary).unwrap();
    let second_open = TimeInterval::new(boundary, end).unwrap();
    let first = Session::new("first", vec![first_open]).unwrap();
    let second = Session::new("second", vec![second_open]).unwrap();
    let schedule = SessionSchedule::new(coverage, vec![first, second]).unwrap();

    let timestamps = [start];
    let values = [100.0];
    let series = SeriesView::new(&timestamps, &values).unwrap();

    let (groups, first, last, lows, highs, means, sums) = {
        let assigned = SessionAssignment::new(series, &schedule, coverage).unwrap();

        (
            assigned.iter(),
            assigned.first(BucketLabel::Start),
            assigned.last(BucketLabel::End),
            assigned.min(BucketLabel::End),
            assigned.max(BucketLabel::End),
            assigned.mean(BucketLabel::End),
            assigned.sum(BucketLabel::End),
        )
    };

    let counts = [
        groups.count(),
        first.count(),
        last.count(),
        lows.count(),
        highs.count(),
        means.count(),
        sums.count(),
    ];

    assert_eq!(counts, [2; 7]);
}
