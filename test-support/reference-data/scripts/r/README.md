# R Statistical Conformance Suite

From the repository root, regenerate all reference CSVs with:

```sh
docker compose -f test-support/reference-data/scripts/r/docker/compose.yaml run --build --rm generate
```

Docker with Compose is the only local prerequisite. Building the image downloads
pinned dependencies; generation runs with networking disabled. Fixtures and
source are mounted read-only, and output is written to `data/r/`.

## Statistical Definitions

Each family script defines its R calculations. Exact CSV headers live in the
[committed tables](../../data/r/) and are checked by the
[typed loaders](../../src/r/).

- [moments.R](moments.R): arithmetic mean; variance with divisors `n - 1`, `n`,
  and `n - 2`; and Joanes-Gill Type 1/2/3 skewness and kurtosis through
  `e1071::skewness` and `e1071::kurtosis`. Kurtosis includes **excess** and
  **Pearson** references; Pearson adds three to each corresponding excess
  estimate. Sample variance uses `stats::var(x)`; population and custom-divisor
  references use centered sums directly: `mean((x - mean(x))^2)` and
  `sum((x - mean(x))^2) / (length(x) - 2)`.
- [dispersion.R](dispersion.R): range, Type 7 IQR, **raw** MAD using
  `stats::mad(x, constant = 1)`, and sample/population standard deviations and
  CVs. Sample standard deviation uses `stats::sd(x)`; population standard
  deviation uses `sqrt(mean((x - mean(x))^2))`. Both CVs divide by the signed
  mean and are ratios, not percentages.
- [quantiles.R](quantiles.R): `stats::quantile(x, probs, type = 7)` at the
  probabilities declared in the script.
- [empirical.R](empirical.R): `stats::ecdf(x)(query)` includes ties;
  `mean(x > query)` counts strict survival directly. Inverse ECDF references use
  `stats::quantile(x, probs, type = 1)`, selecting observations without
  interpolation; probability zero gives the minimum and one gives the maximum.
  Type 1 uses the pinned R version's default rounding tolerance (`fuzz`).
- [association.R](association.R): `stats::cov` uses the sample divisor `n - 1`.
  Paired population covariance uses `mean((x - mean(x)) * (y - mean(y)))`.
  `stats::cor` supplies Pearson, Spearman with average ranks, and Kendall tau-b
  via its corresponding `method` arguments. Matrices use sample covariance,
  population covariance from `crossprod(centered) / nrow(x)` with centered
  columns, and Pearson correlation.
- [regression.R](regression.R): default `stats::lm(y ~ x)` includes an
  intercept. Coefficients and R² come from the fit and its summary; observation
  references use `stats::predict` and `stats::residuals`. The slope is also the
  beta reference.
- [transform.R](transform.R): `x - mean(x)`, sample z-scores from
  `as.numeric(scale(x))`, population z-scores from
  `centered / sqrt(mean(centered^2))`, and `rank(x, ties.method = "average")`.
  Statistical ranks are one-based and may be fractional; positive and negative
  zero tie.

## Input and Output Contracts

The [data guide](../../data/README.md) defines fixture formats and inclusion
rules. Empirical query/probability lists live in `empirical.R`; they must match
fixture case names and be finite, nonempty, and strictly increasing.
Probabilities must lie in `[0, 1]`.

Outputs use fixture filenames without extensions as `case` labels. Cases are
sorted, with rows within each case ordered as follows:

- Quantile tables: ascending probability; empirical CDF/survival: ascending
  query.
- Regression observations and transforms: zero-based `observation` indices in
  original fixture order, distinct from statistical ranks.
- Association matrices: all `p * p` entries in row-major order, including the
  diagonal and both triangles. Zero-based `row,column` indices follow variable
  order in the fixture header.

Numeric references are finite and serialized with 17 significant digits for Rust
`f64` comparisons. Query and probability labels keep their declared decimal
spelling. CSVs use UTF-8, LF endings, unquoted fields, stable headers, and no
implicit row numbers. [common.R](common.R) handles validation and serialization.

## Reproducibility

The [Dockerfile](docker/Dockerfile) pins the R image by version and digest and
sets a dated CRAN snapshot. [renv.lock](renv.lock) locks R and package versions,
including e1071. [Compose](docker/compose.yaml) fixes `linux/amd64`, using
emulation on ARM hosts.

The generator checks installed versions against the lockfile before writing
references. Actual R and package versions are recorded in
[versions.csv](../../data/r/versions.csv).
