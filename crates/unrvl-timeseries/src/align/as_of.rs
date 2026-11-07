use std::time::Duration;

use crate::{SeriesView, Timestamp};

/// Outcome of looking for a right observation at or before a left timestamp.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AsOfOutcome {
    /// The latest eligible right observation is within the age limit.
    Matched {
        /// Index into the supplied right view.
        right_index: usize,

        /// Timestamp of the right observation.
        right_timestamp: Timestamp,

        /// Value of the right observation.
        right_value: f64,

        /// Elapsed time from the right observation to the left observation.
        age: Duration,
    },

    /// The right view has no observation at or before the left timestamp.
    NoPriorObservation,

    /// The latest right observation at or before the left timestamp is older
    /// than the age limit.
    TooOld,
}

/// A left observation and the outcome of its as-of lookup.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AsOfRow {
    /// Index into the supplied left view.
    pub left_index: usize,

    /// Timestamp of the left observation.
    pub left_timestamp: Timestamp,

    /// Value of the left observation.
    pub left_value: f64,

    /// Outcome of the lookup in the right view.
    pub outcome: AsOfOutcome,
}

/// Backward as-of alignment with one result row per left observation.
///
/// Borrows the source data and stores the lookup outcome for each left index.
#[derive(Debug, Clone)]
pub struct AsOfAlignment<'left, 'right> {
    left: SeriesView<'left>,
    right: SeriesView<'right>,
    entries: Vec<Entry>,
    no_prior: usize,
    too_old: usize,
}

impl AsOfAlignment<'_, '_> {
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
        self.len() - self.no_prior - self.too_old
    }

    /// Returns the number of left observations with no prior candidate.
    #[must_use]
    pub const fn no_prior_count(&self) -> usize {
        self.no_prior
    }

    /// Returns the number of left observations whose candidate was too old.
    #[must_use]
    pub const fn too_old_count(&self) -> usize {
        self.too_old
    }

    /// Yields the rows in left timestamp order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = AsOfRow> + '_ {
        self.entries.iter().enumerate().map(|(left_index, entry)| {
            let outcome = match *entry {
                Entry::Matched { right_index, age } => AsOfOutcome::Matched {
                    right_index,
                    right_timestamp: self.right.timestamps()[right_index],
                    right_value: self.right.values()[right_index],
                    age,
                },
                Entry::NoPriorObservation => AsOfOutcome::NoPriorObservation,
                Entry::TooOld => AsOfOutcome::TooOld,
            };

            AsOfRow {
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
    TooOld,
}

/// Matches each left observation to the latest right observation at or before
/// its timestamp, up to an inclusive `max_age`.
///
/// Age is elapsed time between the timestamps. [`Duration::ZERO`] allows only
/// exact matches. A right observation can match several left observations.
/// Indices are relative to the supplied views.
///
/// Uses `O(n + m)` time and `O(n)` additional memory for `n` left and `m`
/// right observations.
///
/// # Examples
///
/// ```
/// # use std::time::Duration;
/// # use unrvl_timeseries::{SeriesView, Timestamp, align};
/// let trade_times = [
///     Timestamp::from_unix_secs(10 * 3600 + 3 * 60)?,
///     Timestamp::from_unix_secs(10 * 3600 + 6 * 60)?,
/// ];
/// let quote_times = [Timestamp::from_unix_secs(10 * 3600 + 60)?];
/// let trades = SeriesView::new(&trade_times, &[101.0, 102.0])?;
/// let quotes = SeriesView::new(&quote_times, &[100.0])?;
///
/// let aligned = align::as_of_backward(trades, quotes, Duration::from_secs(2 * 60));
///
/// // The 10:03 trade matches the 10:01 quote. At 10:06, the quote is stale.
/// assert_eq!(aligned.matched_count(), 1);
/// assert_eq!(aligned.too_old_count(), 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[must_use]
pub fn as_of_backward<'left, 'right>(
    left: SeriesView<'left>,
    right: SeriesView<'right>,
    max_age: Duration,
) -> AsOfAlignment<'left, 'right> {
    let right_times = right.timestamps();
    let mut next_right = 0;
    let mut entries = Vec::with_capacity(left.len());
    let mut no_prior = 0;
    let mut too_old = 0;

    for &left_time in left.timestamps() {
        while right_times
            .get(next_right)
            .is_some_and(|&time| time <= left_time)
        {
            next_right += 1;
        }

        let Some(right_index) = next_right.checked_sub(1) else {
            no_prior += 1;
            entries.push(Entry::NoPriorObservation);
            continue;
        };

        // The candidate is at or before left_time. The unsigned difference
        // gives the exact age even across the full signed timestamp range.
        let right_time = right_times[right_index];
        let age = Duration::from_nanos(
            left_time
                .as_unix_nanos()
                .abs_diff(right_time.as_unix_nanos()),
        );

        if age <= max_age {
            entries.push(Entry::Matched { right_index, age });
        } else {
            too_old += 1;
            entries.push(Entry::TooOld);
        }
    }

    AsOfAlignment { left, right, entries, no_prior, too_old }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_left_rows_and_matches_the_latest_right_observation_within_age() {
        let left_times = [-1, 2, 4, 5, 7, 9].map(Timestamp::from_unix_nanos);
        let left_values = [10.0, 20.0, 40.0, 50.0, 70.0, 90.0];

        let right_times = [0, 2, 7, 10].map(Timestamp::from_unix_nanos);
        let right_values = [100.0, 200.0, 700.0, 1000.0];

        let left = SeriesView::new(&left_times, &left_values).unwrap();
        let right = SeriesView::new(&right_times, &right_values).unwrap();

        let aligned = as_of_backward(left, right, Duration::from_nanos(2));

        let matched = |right_index, nanos| AsOfOutcome::Matched {
            right_index,
            right_timestamp: right_times[right_index],
            right_value: right_values[right_index],
            age: Duration::from_nanos(nanos),
        };

        let expected = [
            AsOfOutcome::NoPriorObservation,
            matched(1, 0),
            matched(1, 2),
            AsOfOutcome::TooOld,
            matched(2, 0),
            matched(2, 2),
        ];

        assert_eq!(aligned.len(), expected.len());
        assert_eq!(aligned.iter().len(), expected.len());
        assert_eq!(aligned.matched_count(), 4);
        assert_eq!(aligned.no_prior_count(), 1);
        assert_eq!(aligned.too_old_count(), 1);

        for (index, (row, outcome)) in aligned.iter().zip(expected).enumerate() {
            assert_eq!(
                row,
                AsOfRow {
                    left_index: index,
                    left_timestamp: left_times[index],
                    left_value: left_values[index],
                    outcome,
                }
            );
        }
    }

    #[test]
    fn handles_empty_views() {
        let empty = SeriesView::new(&[], &[]).unwrap();
        let single = SeriesView::new(&[Timestamp::UNIX_EPOCH], &[1.0]).unwrap();

        let aligned = as_of_backward(empty, single, Duration::ZERO);
        assert!(aligned.is_empty());
        assert_eq!(aligned.iter().next(), None);

        let aligned = as_of_backward(single, empty, Duration::MAX);
        assert_eq!(aligned.len(), 1);
        assert_eq!(aligned.no_prior_count(), 1);
        assert_eq!(
            aligned.iter().next().unwrap().outcome,
            AsOfOutcome::NoPriorObservation
        );
    }

    #[test]
    fn zero_age_accepts_an_exact_match_but_rejects_one_nanosecond() {
        let left_times = [0, 1].map(Timestamp::from_unix_nanos);
        let left = SeriesView::new(&left_times, &[1.0, 2.0]).unwrap();
        let right = SeriesView::new(&[Timestamp::UNIX_EPOCH], &[3.0]).unwrap();

        let aligned = as_of_backward(left, right, Duration::ZERO);
        let mut rows = aligned.iter();

        assert!(matches!(
            rows.next().unwrap().outcome,
            AsOfOutcome::Matched { age: Duration::ZERO, .. }
        ));
        assert_eq!(rows.next().unwrap().outcome, AsOfOutcome::TooOld);
    }

    #[test]
    fn computes_age_exactly_across_the_full_timestamp_span() {
        let left = SeriesView::new(&[Timestamp::MAX], &[1.0]).unwrap();
        let right = SeriesView::new(&[Timestamp::MIN], &[2.0]).unwrap();
        let span = Duration::from_nanos(u64::MAX);

        let aligned = as_of_backward(left, right, span);
        assert_eq!(
            aligned.iter().next().unwrap().outcome,
            AsOfOutcome::Matched {
                right_index: 0,
                right_timestamp: Timestamp::MIN,
                right_value: 2.0,
                age: span,
            }
        );

        let just_below_span = Duration::from_nanos(u64::MAX - 1);
        let aligned = as_of_backward(left, right, just_below_span);
        assert_eq!(aligned.iter().next().unwrap().outcome, AsOfOutcome::TooOld);
    }
}
