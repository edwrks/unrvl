use std::cmp::Ordering;

use crate::Timestamp;

/// Yields equal-timestamp index pairs from strictly increasing slices.
pub(super) fn matching_indices<'a>(
    left: &'a [Timestamp],
    right: &'a [Timestamp],
) -> impl Iterator<Item = (usize, usize)> + 'a {
    let mut left_index = 0;
    let mut right_index = 0;

    std::iter::from_fn(move || {
        while let (Some(left_time), Some(right_time)) =
            (left.get(left_index), right.get(right_index))
        {
            match left_time.cmp(right_time) {
                Ordering::Less => left_index += 1,
                Ordering::Greater => right_index += 1,
                Ordering::Equal => {
                    let pair = (left_index, right_index);

                    left_index += 1;
                    right_index += 1;

                    return Some(pair);
                }
            }
        }

        None
    })
}
