# Statistical Test Data

Fixture requirements and integrity checks for the workspace's reference data.
See the [R generator guide](../scripts/r/README.md) for regeneration and
statistical definitions.

## Layout

- [fixtures/r/](fixtures/r/): hand-authored observations grouped by family.
- [r/](r/): committed R reference CSVs and environment versions.
- [nist/](nist/): published univariate and linear-regression datasets,
  certified values, and integrity manifests.

## R Fixtures

Every fixture must produce a finite, usable reference for every statistic in
its family. Generation rejects malformed inputs and non-finite outputs.

Univariate families use exactly `value`; paired association and regression use
exactly `x,y`. For regression, `x` is the predictor and `y` the response
(`y ~ x`). Matrix fixtures have one named column per variable and one row per
aligned observation. All records must be rectangular, with nonempty, unique
column names; no observations or pairs are silently deleted.

Use small, well-conditioned samples covering ordinary decimals, ties, uneven
tails, minimum sample sizes, and exact relationships where applicable. Copies
across families are intentional so each corpus can evolve independently.

Key restrictions follow from the statistics generated together:

- Moments require at least four observations and nonzero spread for adjusted
  shape statistics.
- Dispersion requires at least two observations and a nonzero mean for finite
  sample/population standard deviations and signed CVs.
- Association and transform fixtures require at least two observations and
  nonconstant series for correlations and z-scores. Linearly dependent matrix
  columns are valid; matrix invertibility is not required.
- Constant and singleton samples are valid empirical fixtures. Their query and
  probability lists live in [empirical.R](../scripts/r/empirical.R).

Undefined cases, degenerate fits, invalid divisors, and extreme scales or
offsets belong in Rust unit/property tests or appropriate NIST validation.

## NIST Integrity

NIST data is externally published and is not input to the R generator. From the
repository root, verify the bundled files with:

```sh
shasum -a 256 --check test-support/reference-data/data/nist/univariate/SHA256SUMS
shasum -a 256 --check test-support/reference-data/data/nist/linear-regression/SHA256SUMS
```

Use the [shared test utilities](../../test-utils/README.md) for approximate
comparisons, with tolerances appropriate to the statistic and numerical scale.
