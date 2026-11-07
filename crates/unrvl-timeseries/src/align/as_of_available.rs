use std::time::Duration;

use crate::{AvailabilityView, SeriesView, Timestamp};

/// Outcome of an as-of lookup that accounts for observation availability.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AvailableAsOfOutcome {
    /// The latest available observation is within the age limit.
    Matched {
        /// Index into the supplied right view.
        right_index: usize,

        /// Observation timestamp from the right series.
        right_timestamp: Timestamp,

        /// Value of the right observation.
        right_value: f64,

        /// When the right observation became available.
        available_at: Timestamp,

        /// Elapsed time from the right observation to the left observation.
        age: Duration,

        /// Elapsed time from the right observation to its availability.
        availability_delay: Duration,
    },

    /// The right view has no observation at or before the left timestamp.
    NoPriorObservation,

    /// There are preceding observations, but none is available yet.
    NotYetAvailable,

    /// The latest available observation is older than the age limit.
    ///
    /// Newer observations may exist but are not yet available.
    TooOld,
}

/// A left observation and the outcome of its availability-aware as-of lookup.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AvailableAsOfRow {
    /// Index into the supplied left view.
    pub left_index: usize,

    /// Timestamp of the left observation, used as the query time.
    pub left_timestamp: Timestamp,

    /// Value of the left observation.
    pub left_value: f64,

    /// Outcome of the lookup in the right view.
    pub outcome: AvailableAsOfOutcome,
}

/// Availability-aware backward alignment with one row per left observation.
///
/// Borrows the source data and stores the lookup outcome for each left index.
#[derive(Debug, Clone)]
pub struct AvailableAsOfAlignment<'left, 'right> {
    left: SeriesView<'left>,
    right: AvailabilityView<'right>,
    entries: Vec<Entry>,
    no_prior: usize,
    not_yet_available: usize,
    too_old: usize,
}

impl AvailableAsOfAlignment<'_, '_> {
    /// Returns the number of left observations, including unmatched rows.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns whether the left view is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the number of left observations with a match.
    #[must_use]
    pub const fn matched_count(&self) -> usize {
        self.len() - self.no_prior - self.not_yet_available - self.too_old
    }

    /// Returns the number of left observations with no preceding observation.
    #[must_use]
    pub const fn no_prior_count(&self) -> usize {
        self.no_prior
    }

    /// Returns the number of left observations with preceding observations
    /// that are all still unavailable.
    #[must_use]
    pub const fn not_yet_available_count(&self) -> usize {
        self.not_yet_available
    }

    /// Returns the number of left observations whose available candidate was
    /// too old.
    #[must_use]
    pub const fn too_old_count(&self) -> usize {
        self.too_old
    }

    /// Yields the rows in left timestamp order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = AvailableAsOfRow> + '_ {
        let right = self.right.series();

        self.entries
            .iter()
            .enumerate()
            .map(move |(left_index, entry)| {
                let outcome = match *entry {
                    Entry::Matched { right_index, age } => {
                        let right_timestamp = right.timestamps()[right_index];
                        let available_at = self.right.available_at()[right_index];
                        let availability_delay = Duration::from_nanos(
                            available_at
                                .as_unix_nanos()
                                .abs_diff(right_timestamp.as_unix_nanos()),
                        );

                        AvailableAsOfOutcome::Matched {
                            right_index,
                            right_timestamp,
                            right_value: right.values()[right_index],
                            available_at,
                            age,
                            availability_delay,
                        }
                    }
                    Entry::NoPriorObservation => AvailableAsOfOutcome::NoPriorObservation,
                    Entry::NotYetAvailable => AvailableAsOfOutcome::NotYetAvailable,
                    Entry::TooOld => AvailableAsOfOutcome::TooOld,
                };

                AvailableAsOfRow {
                    left_index,
                    left_timestamp: self.left.timestamps()[left_index],
                    left_value: self.left.values()[left_index],
                    outcome,
                }
            })
    }
}

#[derive(Debug, Clone, Copy)]
enum Entry {
    Matched { right_index: usize, age: Duration },
    NoPriorObservation,
    NotYetAvailable,
    TooOld,
}

/// Matches each left observation to the latest right observation already
/// available at its timestamp, up to an inclusive `max_age`.
///
/// Both observation and availability times must be at or before the query.
/// Among eligible observations, chooses the latest observation time. An older
/// observation arriving later does not replace a newer eligible observation.
///
/// Age is elapsed time from observation to query, including any availability
/// delay. [`Duration::ZERO`] permits a match only when observation,
/// availability, and query times are equal. A right observation can match
/// several left observations. Indices are relative to the supplied views.
///
/// Every left observation produces a row. [`AvailableAsOfOutcome`]
/// distinguishes matches, missing preceding observations, observations not yet
/// available, and available candidates that are too old.
///
/// Sorts a temporary availability index without changing the supplied data.
/// Uses `O(m log m + n)` time and `O(n + m)` additional memory for `n` left
/// and `m` right observations.
///
/// # Examples
///
/// ```
/// # use std::time::Duration;
/// # use unrvl_timeseries::{AvailabilityView, SeriesView, Timestamp, align};
/// # use unrvl_timeseries::align::AvailableAsOfOutcome;
/// let query_times = [
///     Timestamp::from_unix_secs(10 * 3600 + 3 * 60)?,
///     Timestamp::from_unix_secs(10 * 3600 + 8 * 60)?,
/// ];
/// let observed_at = [Timestamp::from_unix_secs(10 * 3600)?];
/// let available_at = [Timestamp::from_unix_secs(10 * 3600 + 7 * 60)?];
/// let trades = SeriesView::new(&query_times, &[101.0, 102.0])?;
/// let prices = SeriesView::new(&observed_at, &[100.0])?;
/// let prices = AvailabilityView::new(prices, &available_at)?;
///
/// // The 10:00 price is unavailable at 10:03 and eligible at 10:08.
/// let aligned = align::as_of_available(trades, prices, Duration::from_secs(10 * 60));
/// let mut rows = aligned.iter();
/// assert_eq!(
///     rows.next().unwrap().outcome,
///     AvailableAsOfOutcome::NotYetAvailable
/// );
/// assert!(matches!(
///     rows.next().unwrap().outcome,
///     AvailableAsOfOutcome::Matched { .. }
/// ));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[must_use]
pub fn as_of_available<'left, 'right>(
    left: SeriesView<'left>,
    right: AvailabilityView<'right>,
    max_age: Duration,
) -> AvailableAsOfAlignment<'left, 'right> {
    let right_times = right.series().timestamps();
    let available_at = right.available_at();

    let mut by_availability: Vec<_> = (0..right.len()).collect();
    by_availability.sort_unstable_by_key(|&index| available_at[index]);

    let mut next_available = 0;
    let mut latest_index = None;

    let mut entries = Vec::with_capacity(left.len());

    let mut no_prior = 0;
    let mut not_yet_available = 0;
    let mut too_old = 0;

    for &left_time in left.timestamps() {
        while let Some(&right_index) = by_availability.get(next_available) {
            if available_at[right_index] > left_time {
                break;
            }

            // Source indices increase with observation time. A late older
            // observation must not displace a newer available observation.
            if latest_index.is_none_or(|index| index < right_index) {
                latest_index = Some(right_index);
            }

            next_available += 1;
        }

        let Some(right_index) = latest_index else {
            if right_times.first().is_some_and(|&time| time <= left_time) {
                not_yet_available += 1;
                entries.push(Entry::NotYetAvailable);
            } else {
                no_prior += 1;
                entries.push(Entry::NoPriorObservation);
            }

            continue;
        };

        // AvailabilityView guarantees observation <= availability, and the
        // sweep guarantees availability <= query. The unsigned difference
        // preserves the exact age across the full signed timestamp span.
        let age = Duration::from_nanos(
            left_time
                .as_unix_nanos()
                .abs_diff(right_times[right_index].as_unix_nanos()),
        );

        if age <= max_age {
            entries.push(Entry::Matched { right_index, age });
        } else {
            too_old += 1;
            entries.push(Entry::TooOld);
        }
    }

    AvailableAsOfAlignment {
        left,
        right,
        entries,
        no_prior,
        not_yet_available,
        too_old,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_left_rows_and_distinguishes_unmatched_reasons() {
        let left_times = [-1, 0, 2, 3, 4, 5, 6].map(Timestamp::from_unix_nanos);
        let left_values = [-10.0, 0.0, 20.0, 30.0, 40.0, 50.0, 60.0];

        let right_times = [0, 2, 4].map(Timestamp::from_unix_nanos);
        let right_values = [100.0, 102.0, 104.0];

        let available_at = [2, 5, 6].map(Timestamp::from_unix_nanos);
        let left = SeriesView::new(&left_times, &left_values).unwrap();
        let prices = SeriesView::new(&right_times, &right_values).unwrap();
        let right = AvailabilityView::new(prices, &available_at).unwrap();

        let aligned = as_of_available(left, right, Duration::from_nanos(3));

        let matched = |right_index, age_nanos, delay_nanos| AvailableAsOfOutcome::Matched {
            right_index,
            right_timestamp: right_times[right_index],
            right_value: right_values[right_index],
            available_at: available_at[right_index],
            age: Duration::from_nanos(age_nanos),
            availability_delay: Duration::from_nanos(delay_nanos),
        };

        let expected = [
            AvailableAsOfOutcome::NoPriorObservation,
            AvailableAsOfOutcome::NotYetAvailable,
            matched(0, 2, 2),
            matched(0, 3, 2),
            AvailableAsOfOutcome::TooOld,
            matched(1, 3, 3),
            matched(2, 2, 2),
        ];

        assert_eq!(aligned.len(), expected.len());
        assert_eq!(aligned.iter().len(), expected.len());
        assert_eq!(aligned.matched_count(), 4);
        assert_eq!(aligned.no_prior_count(), 1);
        assert_eq!(aligned.not_yet_available_count(), 1);
        assert_eq!(aligned.too_old_count(), 1);

        for (index, (row, outcome)) in aligned.iter().zip(expected).enumerate() {
            assert_eq!(
                row,
                AvailableAsOfRow {
                    left_index: index,
                    left_timestamp: left_times[index],
                    left_value: left_values[index],
                    outcome,
                },
            );
        }
    }

    #[test]
    fn late_older_observations_do_not_replace_newer_available_ones() {
        let left_times = [4, 5, 6, 7, 8].map(Timestamp::from_unix_nanos);
        let right_times = [0, 2, 4, 6].map(Timestamp::from_unix_nanos);

        let available_at = [7, 5, 5, 6].map(Timestamp::from_unix_nanos);
        let left = SeriesView::new(&left_times, &[1.0; 5]).unwrap();
        let prices = SeriesView::new(&right_times, &[100.0, 102.0, 104.0, 106.0]).unwrap();
        let right = AvailabilityView::new(prices, &available_at).unwrap();

        let aligned = as_of_available(left, right, Duration::MAX);
        let indices: Vec<_> = aligned
            .iter()
            .map(|row| match row.outcome {
                AvailableAsOfOutcome::Matched { right_index, .. } => Some(right_index),
                AvailableAsOfOutcome::NotYetAvailable => None,
                outcome => panic!("unexpected outcome: {outcome:?}"),
            })
            .collect();

        // At 5 ns, two observations arrive together; the newer one wins.
        // The oldest observation arriving at 7 ns does not change the match.
        assert_eq!(indices, [None, Some(2), Some(3), Some(3), Some(3)]);
    }

    #[test]
    fn handles_empty_views() {
        let empty = SeriesView::new(&[], &[]).unwrap();
        let single = SeriesView::new(&[Timestamp::UNIX_EPOCH], &[1.0]).unwrap();
        let available = AvailabilityView::new(single, &[Timestamp::UNIX_EPOCH]).unwrap();

        let aligned = as_of_available(empty, available, Duration::ZERO);
        assert!(aligned.is_empty());
        assert_eq!(aligned.iter().next(), None);

        let available = AvailabilityView::new(empty, &[]).unwrap();
        let aligned = as_of_available(single, available, Duration::MAX);
        assert_eq!(aligned.len(), 1);
        assert_eq!(aligned.no_prior_count(), 1);
        assert_eq!(
            aligned.iter().next().unwrap().outcome,
            AvailableAsOfOutcome::NoPriorObservation,
        );
    }

    #[test]
    fn zero_age_requires_equal_observation_availability_and_query_times() {
        let left_times = [0, 1, 2].map(Timestamp::from_unix_nanos);
        let right_times = [0, 2].map(Timestamp::from_unix_nanos);

        let available_at = [1, 2].map(Timestamp::from_unix_nanos);
        let left = SeriesView::new(&left_times, &[1.0; 3]).unwrap();
        let prices = SeriesView::new(&right_times, &[100.0, 102.0]).unwrap();
        let right = AvailabilityView::new(prices, &available_at).unwrap();

        let aligned = as_of_available(left, right, Duration::ZERO);
        let mut rows = aligned.iter();

        assert_eq!(
            rows.next().unwrap().outcome,
            AvailableAsOfOutcome::NotYetAvailable,
        );
        assert_eq!(rows.next().unwrap().outcome, AvailableAsOfOutcome::TooOld);
        assert_eq!(
            rows.next().unwrap().outcome,
            AvailableAsOfOutcome::Matched {
                right_index: 1,
                right_timestamp: right_times[1],
                right_value: 102.0,
                available_at: available_at[1],
                age: Duration::ZERO,
                availability_delay: Duration::ZERO,
            },
        );
    }

    #[test]
    fn computes_age_and_delay_exactly_across_the_full_timestamp_span() {
        let left = SeriesView::new(&[Timestamp::MAX], &[1.0]).unwrap();
        let prices = SeriesView::new(&[Timestamp::MIN], &[100.0]).unwrap();
        let right = AvailabilityView::new(prices, &[Timestamp::MAX]).unwrap();
        let span = Duration::from_nanos(u64::MAX);

        let aligned = as_of_available(left, right, span);
        assert_eq!(
            aligned.iter().next().unwrap().outcome,
            AvailableAsOfOutcome::Matched {
                right_index: 0,
                right_timestamp: Timestamp::MIN,
                right_value: 100.0,
                available_at: Timestamp::MAX,
                age: span,
                availability_delay: span,
            },
        );

        let just_below_span = Duration::from_nanos(u64::MAX - 1);
        let aligned = as_of_available(left, right, just_below_span);
        assert_eq!(
            aligned.iter().next().unwrap().outcome,
            AvailableAsOfOutcome::TooOld,
        );
    }

    #[test]
    fn indices_are_relative_to_the_supplied_views() {
        let left_times = [0, 3, 4].map(Timestamp::from_unix_nanos);
        let right_times = [-1, 0, 2].map(Timestamp::from_unix_nanos);

        let available_at = [-1, 0, 3].map(Timestamp::from_unix_nanos);
        let left = SeriesView::new(&left_times, &[10.0, 30.0, 40.0]).unwrap();
        let prices = SeriesView::new(&right_times, &[99.0, 100.0, 102.0]).unwrap();

        let left = left.between(left_times[1]..).unwrap();
        let prices = prices.between(Timestamp::UNIX_EPOCH..).unwrap();
        let right = AvailabilityView::new(prices, &available_at[1..]).unwrap();

        let aligned = as_of_available(left, right, Duration::MAX);
        let row = aligned.iter().next().unwrap();

        assert_eq!(row.left_index, 0);
        assert_eq!(row.left_timestamp, left_times[1]);
        assert_eq!(row.left_value, 30.0);
        assert!(matches!(
            row.outcome,
            AvailableAsOfOutcome::Matched { right_index: 1, right_value: 102.0, .. },
        ));
    }

    #[test]
    fn agrees_with_a_direct_search_for_each_query() {
        let left_times: Vec<_> = (-1..=10).map(Timestamp::from_unix_nanos).collect();
        let left_values = vec![1.0; left_times.len()];
        let right_times = [0, 2, 4, 6].map(Timestamp::from_unix_nanos);
        let left = SeriesView::new(&left_times, &left_values).unwrap();
        let prices = SeriesView::new(&right_times, &[100.0, 102.0, 104.0, 106.0]).unwrap();
        let cases = [
            ([0, 2, 4, 6], Duration::ZERO),
            ([7, 5, 5, 6], Duration::from_nanos(2)),
            ([7, 5, 5, 6], Duration::MAX),
            ([7, 7, 7, 7], Duration::from_nanos(3)),
            ([1, 3, 6, 9], Duration::from_nanos(2)),
        ];

        for (availability_nanos, max_age) in cases {
            let available_at = availability_nanos.map(Timestamp::from_unix_nanos);
            let right = AvailabilityView::new(prices, &available_at).unwrap();
            let aligned = as_of_available(left, right, max_age);
            let actual: Vec<_> = aligned.iter().map(|row| row.outcome).collect();
            let expected: Vec<_> = left_times
                .iter()
                .map(|&query| reference_outcome(query, right, max_age))
                .collect();

            assert_eq!(
                actual, expected,
                "availability={availability_nanos:?}, max_age={max_age:?}"
            );
        }
    }

    fn reference_outcome(
        query: Timestamp,
        right: AvailabilityView<'_>,
        max_age: Duration,
    ) -> AvailableAsOfOutcome {
        let series = right.series();
        let times = series.timestamps();
        let available_at = right.available_at();

        // Search observation order directly, without sorting or a shared
        // cursor.
        let candidate = (0..right.len())
            .rev()
            .find(|&index| times[index] <= query && available_at[index] <= query);

        let Some(index) = candidate else {
            return if times.iter().any(|&time| time <= query) {
                AvailableAsOfOutcome::NotYetAvailable
            } else {
                AvailableAsOfOutcome::NoPriorObservation
            };
        };

        let age = query.checked_duration_since(times[index]).unwrap();
        if age > max_age {
            return AvailableAsOfOutcome::TooOld;
        }

        AvailableAsOfOutcome::Matched {
            right_index: index,
            right_timestamp: times[index],
            right_value: series.values()[index],
            available_at: available_at[index],
            age,
            availability_delay: available_at[index]
                .checked_duration_since(times[index])
                .unwrap(),
        }
    }
}
