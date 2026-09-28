# R Statistical Conformance Suite

From the repository root, regenerate all reference CSVs with:

```sh
docker compose -f test-support/reference-data/scripts/r/docker/compose.yaml run --build --rm generate
```

Docker with Compose is the only local prerequisite.

Generation runs without network access. Fixtures and source are mounted
read-only, and output is written to `data/r/`.

## Families and Definitions

Inputs live under `data/fixtures/r/<family>/*.csv`.

Moments, dispersion, quantiles, and empirical fixtures use a `value` column.
Association and regression use exactly `x,y`. For regression, `x` is the
predictor and `y` is the response.

| Family / output           | Columns after `case`                       |
| ------------------------- | ------------------------------------------ |
| `moments.csv`             | `n,mean,variance,skewness,excess_kurtosis` |
| `dispersion.csv`          | `n,range,iqr,mad,std_dev,cv`               |
| `quantiles.csv`           | `probability,value`                        |
| `empirical.csv`           | `query,cdf,sf`                             |
| `empirical-quantiles.csv` | `probability,value`                        |
| `association.csv`         | `n,covariance,correlation`                 |
| `regression.csv`          | `n,intercept,slope,r_squared`              |

- Variance, standard deviation, and covariance use the sample denominator
  `n - 1`.
- Skewness and excess kurtosis use `e1071` Type 2; moments require at least four
  observations.
- IQR and `quantiles.csv` use Type 7. Quantile probabilities are 0, .01, .05,
  .25, .5, .75, .95, .99, and 1.
- The empirical `cdf` column uses `stats::ecdf(x)(query)`: the fraction of
  observations less than or equal to the query, including all ties. Values
  are in `[0, 1]`; the ECDF is a step function without interpolation.
- The empirical `sf` column uses `mean(x > query)`, counting strict exceedances
  directly rather than subtracting the ECDF from one.
- Inverse ECDF references in `empirical-quantiles.csv` use
  Hyndman–Fan Type 1 quantiles: `stats::quantile(x, probs, type = 1)`.
  Values are observations selected without interpolation. Probability zero
  returns the sample minimum; probability one returns the maximum. This does
  not change the Type 7 references in `quantiles.csv`.
- MAD is **raw** median absolute deviation from the median:
  `stats::mad(x, center = stats::median(x), constant = 1)`. R's default scaling
  is intentionally disabled.
- CV is the **signed ratio** `stats::sd(x) / mean(x)`, not a percentage or an
  absolute value. Dispersion fixtures have non-zero means; the negative-mean
  asymmetric fixture tests this convention.
- Correlation is Pearson correlation.
- Regression uses unmodified `stats::lm(y ~ x)`. Coefficients and R² come
  directly from the fit and its summary. The slope also serves as the beta
  reference.

`versions.csv` has `component,version` columns and records the actual R version
and all locked package versions. Every numeric reference is finite, formatted
with 17 significant digits, and directly comparable with Rust `f64`. Query and
probability labels retain their declared decimal spelling. Files use UTF-8, LF
endings, unquoted fields, stable headers, no row numbers, and sorted cases (then
ascending probability for quantile tables or query for `empirical.csv`).

## Empirical Queries and Probabilities

`empirical.R` declares query and probability lists for each observation fixture.
Case names must match exactly, and each list must be finite, nonempty, and
strictly increasing. Probabilities must lie in `[0, 1]`. There are no separate
input CSVs for queries or probabilities.

The five cases cover ordinary decimal observations, a constant sample, a
singleton, repeated observations, and two distinct observations. They produce:

- 33 ECDF and survival rows, with queries below and above the sample range, at
  observations, and between observations.
- 68 inverse ECDF rows, covering probability endpoints, interior probabilities,
  and values below, at, and above the cumulative jumps. For the ties fixture,
  the jump probabilities `2/9`, `4/9`, and `7/9` use decimal approximations.

Hyndman–Fan Type 1 references use the pinned R version's default rounding
tolerance (`fuzz`). The tables provide references for `Ecdf::cdf`, `Ecdf::sf`,
and `Ecdf::inverse_cdf`.

## Reproducibility

- R **4.6.1**, from a Rocker image pinned by version and digest.
- Platform **linux/amd64**, using Docker emulation on ARM hosts.
- `CRAN_SNAPSHOT=2026-07-01`, using
  <https://packagemanager.posit.co/cran/2026-07-01>.
- `E1071_VERSION=1.7-17`.
- Exact R package dependencies in **renv.lock**.
