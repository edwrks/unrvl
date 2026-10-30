use unrvl_numerics::convert::usize_to_f64;

/// Returns one-based average ranks in the original observation order.
///
/// The caller must validate that all values are finite and the sample length
/// is at most `2^53`.
#[expect(
    clippy::float_cmp,
    reason = "ties use exact equality, including signed zeros"
)]
pub fn average_ranks(xs: &[f64]) -> Vec<f64> {
    // Sort observation indices by value, preserving their original positions.
    let mut indices: Vec<_> = (0..xs.len()).collect();
    indices.sort_unstable_by(|&a, &b| xs[a].total_cmp(&xs[b]));

    let mut ranks = vec![0.0; xs.len()];
    let mut start = 0;

    // Process each group of equal values in sorted order.
    while start < indices.len() {
        // Find the exclusive end of the current tied group. A group of one
        // observation is handled the same way as ties.
        let mut end = start + 1;
        while end < indices.len() && xs[indices[end]] == xs[indices[start]] {
            end += 1;
        }

        // The group occupies one-based ranks start + 1 through end. Assign
        // every member the midpoint of those ranks.
        let rank = usize_to_f64(start + 1).midpoint(usize_to_f64(end));

        // Write ranks back to their original observation positions.
        for &index in &indices[start..end] {
            ranks[index] = rank;
        }

        // Advance to the next group.
        start = end;
    }

    ranks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_ties_in_the_original_observation_order() {
        // Sorted values are [1, 2, 2, 4, 4]. The twos share ranks 2 and 3;
        // the fours share ranks 4 and 5. Return ranks in the original order.
        let xs = [4.0, 1.0, 2.0, 2.0, 4.0];
        let expected = [4.5, 1.0, 2.5, 2.5, 4.5];

        assert_eq!(average_ranks(&xs), expected);
    }
}
