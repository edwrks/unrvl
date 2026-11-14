//! Pairing observations from two time series.

mod as_of;
mod as_of_available;
mod inner;
mod merge;
mod slots;

pub use as_of::{AsOfAlignment, AsOfOutcome, AsOfRow, as_of_backward};
pub use as_of_available::{
    AvailableAsOfAlignment, AvailableAsOfOutcome, AvailableAsOfRow, as_of_available,
};
pub use inner::{AlignedPair, InnerAlignment, inner};
pub use slots::{SlotAlignment, SlotAlignmentError, SlotPair, inner_slots};
