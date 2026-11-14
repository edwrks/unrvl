/// Which full bucket boundary supplies its output timestamp.
///
/// Applies to both fixed and session buckets. For a session, the boundaries
/// are its first opening and final close.
///
/// Labels use the full interval even when analysis bounds clip the bucket.
/// A label can lie outside the selected analysis interval. It does not indicate
/// when an observation occurred or when data or a calculated result became
/// available.
///
/// See [`BucketSelection`](super::BucketSelection) for an example
/// distinguishing the label, full interval, selected interval, and original
/// observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BucketLabel {
    /// Inclusive start of the full bucket, or the session's first opening.
    Start,

    /// Exclusive end of the full bucket, or the session's final close.
    End,
}
