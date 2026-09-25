# Statistical Test Data

The workspace uses three complementary testing layers:

| Layer                    | Responsibility                                            |
| ------------------------ | --------------------------------------------------------- |
| NIST datasets            | Independently certified numerical accuracy                |
| R conformance fixtures   | Statistical definitions on small, well-conditioned inputs |
| Rust unit/property tests | Undefined cases, numerical stress, and invariants         |

## Layout

```text
data/
├── fixtures/r/
│   ├── moments/
│   ├── dispersion/
│   ├── quantiles/
│   ├── association/
│   └── regression/
├── nist/
│   ├── linear-regression/
│   └── univariate/
└── r/
    ├── moments.csv
    ├── dispersion.csv
    ├── quantiles.csv
    ├── association.csv
    ├── regression.csv
    └── versions.csv
```

## R Fixtures

**Inclusion in a fixture family means every statistic generated for that family
must have a finite, usable R reference.**

Univariate families use exactly `value`; bivariate families use exactly `x,y`,
with `x` the predictor and `y` the response for regression (`y ~ x`). Files
contain small hand-authored observations. Deliberate copies across families keep
each corpus independent and easy to inspect.

| Case                                    | Purpose                                                             |
| --------------------------------------- | ------------------------------------------------------------------- |
| `baseline`                              | Ordinary decimal observations and interpolation                     |
| `asymmetric-tail`                       | Uneven tails and, for dispersion, a negative mean/CV                |
| `ties`                                  | Repeated integer observations and quantile boundaries               |
| `minimum-shape-sample`                  | Four observations and small-sample corrections                      |
| `symmetric`                             | Zero skewness; included only in moments, where a zero mean is valid |
| `perfect-positive` / `perfect-negative` | Exact linear relationships and correlation signs                    |
| `zero-covariance`                       | Varying series with zero association                                |

Constants, singular predictors, zero-mean CV cases, and extreme scales/offsets
belong in Rust tests or appropriate NIST validation. Do not add them to a family
where any reference would be undefined or numerically unreliable.

The [generator guide](../scripts/r/README.md) defines all statistics and the
single Docker command. The generator reads fixtures without modifying them and
writes finite, deterministic reference tables under `r/`.

## NIST Integrity

NIST data is externally published and is not input to the R generator. From the
repository root, verify the bundled files with:

```sh
shasum -a 256 --check test-support/reference-data/data/nist/univariate/SHA256SUMS
shasum -a 256 --check test-support/reference-data/data/nist/linear-regression/SHA256SUMS
```

Use approximate comparisons with the workspace's shared test utilities. Choose
tolerances appropriate to each statistic and numerical scale.

