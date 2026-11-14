mod time {
    use std::time::Duration;

    use unrvl_timeseries::{SeriesView, TimeInterval, Timestamp, TimestampError};

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
    fn time_intervals_are_available_externally() {
        let start = Timestamp::UNIX_EPOCH;
        let end = Timestamp::from_unix_nanos(2);
        let interval = TimeInterval::new(start, end).unwrap();

        let _ = interval.start();
        let _ = interval.end();
        let _ = interval.contains(start);
        let _ = interval.duration();

        let timestamps = [start, end];
        let values = [1.0, 2.0];
        let series = SeriesView::new(&timestamps, &values).unwrap();
        let _ = series.between(interval).unwrap();
    }
}

mod series {
    use unrvl_timeseries::{AvailabilityView, Series, SeriesView, Timestamp};

    #[test]
    fn series_views_are_available_externally() {
        let timestamps = [0, 1, 3].map(Timestamp::from_unix_nanos);
        let values = [-2.0, 0.0, 5.0];
        let series = SeriesView::new(&timestamps, &values).unwrap();

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
        let series = Series::new(timestamps, values).unwrap();

        let _ = series.timestamps();
        let _ = series.values();
        let _ = series.len();
        let _ = series.is_empty();
        let _ = series.view();
    }

    #[test]
    fn availability_views_are_available_externally() {
        let timestamps = [Timestamp::UNIX_EPOCH];
        let series = SeriesView::new(&timestamps, &[100.0]).unwrap();
        let availability = AvailabilityView::new(series, &timestamps).unwrap();

        let _ = availability.series();
        let _ = availability.available_at();
        let _ = availability.len();
        let _ = availability.is_empty();
        let _ = availability.between(..).unwrap();
    }
}

mod session {
    use unrvl_timeseries::{Session, SessionSchedule, TimeInterval, Timestamp};

    #[test]
    fn sessions_are_available_externally() {
        let start = Timestamp::UNIX_EPOCH;
        let end = Timestamp::from_unix_nanos(2);
        let opening = TimeInterval::new(start, end).unwrap();
        let session = Session::new("regular", vec![opening]).unwrap();

        let _ = session.id();
        let _ = session.intervals();
        let _ = session.first_open();
        let _ = session.final_close();
        let _ = session.is_open_at(start);
    }

    #[test]
    fn session_schedules_are_available_externally() {
        let start = Timestamp::UNIX_EPOCH;
        let end = Timestamp::from_unix_nanos(2);
        let coverage = TimeInterval::new(start, end).unwrap();
        let session = Session::new("regular", vec![coverage]).unwrap();
        let schedule = SessionSchedule::new(coverage, vec![session]).unwrap();

        let _ = schedule.coverage();
        let _ = schedule.sessions();
        let _ = schedule.close_labels().next();
        let _ = schedule.session_at(start);
        let _ = schedule.previous_session(end);
        let _ = schedule.next_session(start);
        let _ = schedule.status_at(start);
        let _ = schedule.open_duration(coverage);
    }
}

mod cadence {
    use std::time::Duration;

    use unrvl_timeseries::cadence::{FixedGrid, GridCoverage, SessionCloseCoverage};
    use unrvl_timeseries::{SeriesView, Session, SessionSchedule, TimeInterval, Timestamp};

    #[test]
    fn fixed_sampling_grids_are_available_externally() {
        let grid = FixedGrid::new(
            Timestamp::UNIX_EPOCH,
            Duration::from_secs(3600),
            Duration::from_secs(60),
        )
        .unwrap();

        let _ = grid.origin();
        let _ = grid.spacing();
        let _ = grid.tolerance();
    }

    #[test]
    fn grid_coverage_is_available_externally() {
        let timestamps = [Timestamp::UNIX_EPOCH];
        let series = SeriesView::new(&timestamps, &[100.0]).unwrap();
        let grid = FixedGrid::new(timestamps[0], Duration::from_secs(60), Duration::ZERO).unwrap();
        let coverage = GridCoverage::new(series, grid, timestamps[0]..=timestamps[0]).unwrap();

        let _ = coverage.series();
        let _ = coverage.grid();
        let _ = coverage.bounds();
        let _ = coverage.observed_count();
        let _ = coverage.expected_count();
        let _ = coverage.missing_count();
        let _ = coverage.is_complete();
        let _ = coverage.gaps();
        let _ = coverage.iter().next();
    }

    #[test]
    fn session_close_coverage_is_available_externally() {
        let open = Timestamp::UNIX_EPOCH;
        let close = Timestamp::from_unix_nanos(1);
        let interval = TimeInterval::new(open, close).unwrap();
        let session = Session::new("regular", vec![interval]).unwrap();
        let schedule = SessionSchedule::new(interval, vec![session]).unwrap();
        let timestamps = [close];
        let series = SeriesView::new(&timestamps, &[100.0]).unwrap();
        let coverage = SessionCloseCoverage::new(series, &schedule, ..).unwrap();

        let _ = coverage.series();
        let _ = coverage.schedule();
        let _ = coverage.bounds();
        let _ = coverage.observed_count();
        let _ = coverage.expected_count();
        let _ = coverage.missing_count();
        let _ = coverage.is_complete();
        let _ = coverage.gaps();
        let _ = coverage.missing().next();
        let _ = coverage.iter().next();
    }
}

mod align {
    use std::time::Duration;

    use unrvl_timeseries::cadence::{FixedGrid, GridCoverage};
    use unrvl_timeseries::{AvailabilityView, SeriesView, Timestamp, align};

    #[test]
    fn exact_alignment_is_available_externally() {
        let timestamps = [Timestamp::UNIX_EPOCH];
        let left = SeriesView::new(&timestamps, &[1.0]).unwrap();
        let right = SeriesView::new(&timestamps, &[2.0]).unwrap();

        let aligned = align::inner(left, right);
        let _ = aligned.iter().next();
    }

    #[test]
    fn backward_as_of_alignment_is_available_externally() {
        let timestamps = [Timestamp::UNIX_EPOCH];
        let query_times = [Timestamp::from_unix_nanos(1)];
        let left = SeriesView::new(&query_times, &[1.0]).unwrap();
        let right = SeriesView::new(&timestamps, &[2.0]).unwrap();

        let aligned = align::as_of_backward(left, right, Duration::ZERO);
        let row = aligned.iter().next().unwrap();
        let align::AsOfOutcome::TooOld { right_index, age } = row.outcome else {
            panic!("expected a stale observation");
        };

        let _ = (row.left_index, right_index, age);
    }

    #[test]
    fn availability_aware_alignment_is_available_externally() {
        let timestamps = [Timestamp::UNIX_EPOCH];
        let query_times = [Timestamp::from_unix_nanos(1)];
        let left = SeriesView::new(&query_times, &[1.0]).unwrap();
        let right = SeriesView::new(&timestamps, &[2.0]).unwrap();
        let right = AvailabilityView::new(right, &timestamps).unwrap();

        let aligned = align::as_of_available(left, right, Duration::ZERO);
        let row = aligned.iter().next().unwrap();
        let align::AvailableAsOfOutcome::TooOld { right_index, age } = row.outcome else {
            panic!("expected a stale observation");
        };

        let _ = (row.left_index, right_index, age);
    }

    #[test]
    fn slot_alignment_is_available_externally() {
        let timestamps = [Timestamp::UNIX_EPOCH];
        let left = SeriesView::new(&timestamps, &[100.0]).unwrap();
        let right = SeriesView::new(&timestamps, &[101.0]).unwrap();
        let grid = FixedGrid::new(timestamps[0], Duration::from_secs(60), Duration::ZERO).unwrap();

        let aligned = {
            let left = GridCoverage::new(left, grid, ..).unwrap();
            let right = GridCoverage::new(right, grid, ..).unwrap();
            align::inner_slots(&left, &right).unwrap()
        };

        let _ = aligned.iter().next();
    }
}

mod resample {
    use std::time::Duration;

    use unrvl_timeseries::resample::{
        AggregateOutcome, BucketLabel, FixedBuckets, SessionAggregate, SessionAssignment,
        SessionSelection,
    };
    use unrvl_timeseries::{SeriesView, Session, SessionSchedule, TimeInterval, Timestamp};

    #[test]
    fn fixed_bucket_assignment_is_available_externally() {
        let start = Timestamp::UNIX_EPOCH;
        let end = Timestamp::from_unix_secs(120).unwrap();
        let bounds = TimeInterval::new(start, end).unwrap();
        let buckets = FixedBuckets::new(start, Duration::from_secs(60), bounds).unwrap();
        let timestamps = [start];
        let series = SeriesView::new(&timestamps, &[100.0]).unwrap();
        let assigned = buckets.assign(series).unwrap();
        let bucket = assigned.iter().next().unwrap();

        let _ = buckets.origin();
        let _ = buckets.width();
        let _ = buckets.bounds();
        let _ = buckets.bucket_count();
        let _ = assigned.series();
        let _ = assigned.buckets();
        let _ = bucket.interval;
        let _ = bucket.selected_interval;
        let _ = bucket.source_range;
        let _ = bucket.observations.values();
        let _ = bucket.is_partial();
        let _ = bucket.label(BucketLabel::Start);
        let _ = bucket.label(BucketLabel::End);
    }

    #[test]
    fn bucket_observation_selectors_are_available_externally() {
        let start = Timestamp::UNIX_EPOCH;
        let end = Timestamp::from_unix_secs(120).unwrap();
        let bounds = TimeInterval::new(start, end).unwrap();
        let buckets = FixedBuckets::new(start, Duration::from_secs(60), bounds).unwrap();
        let timestamps = [start];
        let series = SeriesView::new(&timestamps, &[100.0]).unwrap();
        let assigned = buckets.assign(series).unwrap();

        let _ = assigned.first(BucketLabel::Start).next();
        let _ = assigned.min(BucketLabel::End).next();
        let _ = assigned.max(BucketLabel::End).next();

        let row = assigned.last(BucketLabel::End).next().unwrap();
        let observation = row.observation.unwrap();

        let _ = row.timestamp;
        let _ = row.interval;
        let _ = row.selected_interval;
        let _ = row.source_range;
        let _ = row.observed_count();
        let _ = row.is_partial();
        let _ = observation.index;
        let _ = observation.timestamp;
        let _ = observation.value;
    }

    #[test]
    fn bucket_aggregates_are_available_externally() {
        let start = Timestamp::UNIX_EPOCH;
        let end = Timestamp::from_unix_secs(120).unwrap();
        let bounds = TimeInterval::new(start, end).unwrap();
        let buckets = FixedBuckets::new(start, Duration::from_secs(60), bounds).unwrap();
        let timestamps = [start];
        let series = SeriesView::new(&timestamps, &[100.0]).unwrap();
        let assigned = buckets.assign(series).unwrap();

        let _ = assigned.sum(BucketLabel::Start).next();

        let row = assigned.mean(BucketLabel::End).next().unwrap();
        let outcome = row.outcome;

        let _ = row.timestamp;
        let _ = row.interval;
        let _ = row.selected_interval;
        let _ = row.source_range;
        let _ = row.observed_count();
        let _ = row.is_partial();

        match outcome {
            AggregateOutcome::Value(value)
            | AggregateOutcome::NumericalFailure { result: value } => {
                let _ = value;
            }
            AggregateOutcome::Empty => {}
        }
    }

    #[test]
    fn session_assignment_and_resampling_are_available_externally() {
        let open = Timestamp::UNIX_EPOCH;
        let close = Timestamp::from_unix_secs(3600).unwrap();
        let bounds = TimeInterval::new(open, close).unwrap();
        let session = Session::new("regular", vec![bounds]).unwrap();
        let schedule = SessionSchedule::new(bounds, vec![session]).unwrap();
        let timestamps = [open];
        let series = SeriesView::new(&timestamps, &[100.0]).unwrap();
        let assigned = SessionAssignment::new(series, &schedule, bounds).unwrap();
        let bucket = assigned.iter().next().unwrap();

        let _ = assigned.series();
        let _ = assigned.schedule();
        let _ = assigned.bounds();
        let _ = assigned.session_count();
        let _ = bucket.session_index;
        let _ = bucket.session.id();
        let _ = bucket.interval;
        let _ = bucket.selected_interval;
        let _ = bucket.source_range;
        let _ = bucket.observations.values();
        let _ = bucket.label(BucketLabel::End);
        let _ = bucket.is_partial();

        let _ = assigned.first(BucketLabel::Start).next();
        let _ = assigned.min(BucketLabel::End).next();
        let _ = assigned.max(BucketLabel::End).next();

        let selected: SessionSelection<'_> = assigned.last(BucketLabel::End).next().unwrap();
        let _ = selected.session.id();
        let _ = selected.session_index;
        let _ = selected.observation;
        let _ = selected.observed_count();
        let _ = selected.is_partial();

        let _ = assigned.sum(BucketLabel::Start).next();

        let calculated: SessionAggregate<'_> = assigned.mean(BucketLabel::End).next().unwrap();
        let _ = calculated.session.id();
        let _ = calculated.session_index;
        let _ = calculated.outcome;
        let _ = calculated.observed_count();
        let _ = calculated.is_partial();
    }
}
