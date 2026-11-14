//! Grouping observations into time intervals for resampling.

mod aggregate;
mod fixed;
mod label;
mod selection;
mod session;

pub use aggregate::{AggregateOutcome, BucketAggregate, SessionAggregate};
pub use fixed::{Bucket, BucketAssignment, BucketError, FixedBuckets};
pub use label::BucketLabel;
pub use selection::{BucketObservation, BucketSelection, SessionSelection};
pub use session::{SessionAssignment, SessionAssignmentError, SessionBucket};
