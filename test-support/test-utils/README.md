# Test Utilities

Shared testing utilities used across the workspace.

This crate contains reusable helpers that do not belong to any production crate,
including:

- approximate floating-point comparison utilities;
- assertion helpers; and
- property-testing strategies for numerical inputs.

Keeping these utilities in a dedicated crate allows tests across the workspace
to use the same comparison policies and input-generation strategies without
duplicating implementation details.

## Modules

- `approx`: floating-point tolerances and approximate equality helpers.
- `strategies`: reusable property-testing strategies for numerical values and
  samples.

This crate is intended for tests and development tooling only.

