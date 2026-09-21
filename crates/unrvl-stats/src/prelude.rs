//! Extension traits for method-style statistical operations.
//!
//! Importing this prelude makes the crate's statistical methods available on
//! slices, arrays, and vectors of `f64` through slice coercion.
//!
//! ```
//! use unrvl_stats::prelude::*;
//!
//! let xs = [1.0, 2.0, 3.0];
//!
//! assert_eq!(xs.mean(), 2.0);
//! ```

// assert_eq!(xs.std_dev(), 1.0);
// assert_eq!(xs.median(), 2.0);

// pub use crate::association::AssociationExt;
// pub use crate::dispersion::DispersionExt;
pub use crate::moments::MomentsExt;
// pub use crate::quantile::QuantileExt;
// pub use crate::transform::TransformsExt;
