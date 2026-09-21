# R Statistical Conformance Suite

From the repository root, regenerate all six reference CSVs with:

```sh
docker compose -f test-support/reference-data/scripts/r/docker/compose.yaml run --build --rm generate
```

Docker with Compose is the only local prerequisite.

Generation runs without network access. Fixtures and source are mounted
read-only, and output is written to `data/r/`.

## Families and Definitions

Inputs live under `data/fixtures/r/<family>/*.csv`.

Moments, dispersion, and quantiles use a `value` column; association and
regression use exactly `x,y`. For regression, `x` is the predictor and `y` is
the response.

| Family / output   | Columns after `case`                       |
| ----------------- | ------------------------------------------ |
| `moments.csv`     | `n,mean,variance,skewness,excess_kurtosis` |
| `dispersion.csv`  | `n,range,iqr,mad,std_dev,cv`               |
| `quantiles.csv`   | `probability,value`                        |
| `association.csv` | `n,covariance,correlation`                 |
| `regression.csv`  | `n,intercept,slope,r_squared`              |

- Variance, standard deviation, and covariance use the sample denominator
  `n - 1`.
- Skewness and excess kurtosis use `e1071` Type 2; moments require at least four
  observations.
- IQR and quantiles use Type 7. Quantile probabilities are 0, .01, .05, .25, .5,
  .75, .95, .99, and 1.
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
with 17 significant digits, and directly comparable with Rust `f64`. Files use
UTF-8, LF endings, unquoted fields, stable headers, no row numbers, and sorted
cases (then ascending probability for quantiles).

## Reproducibility

- R **4.6.1**, from a Rocker image pinned by version and digest.
- Platform **linux/amd64**, using Docker emulation on ARM hosts.
- `CRAN_SNAPSHOT=2026-07-01`, using
  <https://packagemanager.posit.co/cran/2026-07-01>.
- `E1071_VERSION=1.7-17`.
- Exact R package dependencies in **renv.lock**.

