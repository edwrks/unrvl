//! Fixed-grid and session-close sampling expectations and observation coverage.

mod coverage;
mod grid;
mod session_close;

pub use coverage::{CadenceError, GridCoverage, GridObservation, SlotGap};
pub use grid::{FixedGrid, GridError};
pub use session_close::{
    SessionClose, SessionCloseCoverage, SessionCloseError, SessionCloseObservation,
};
