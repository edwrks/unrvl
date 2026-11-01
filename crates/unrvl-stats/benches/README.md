# Statistics benchmarks

Run one category at a time in Cargo's benchmark profile:

```sh
cargo bench -p unrvl-stats --bench moments
cargo bench -p unrvl-stats --bench dispersion
cargo bench -p unrvl-stats --bench quantiles
cargo bench -p unrvl-stats --bench empirical
cargo bench -p unrvl-stats --bench association
cargo bench -p unrvl-stats --bench regression
cargo bench -p unrvl-stats --bench transform
```

Filter a group, or exercise all benchmarks once without timing measurements:

```sh
cargo bench -p unrvl-stats --bench association -- 'association/matrix'
cargo bench -p unrvl-stats -- --test
```

Criterion records results under `target/criterion`. Full measurement runs can
take several minutes per category.

## Workloads

| Target        | Operations                                                                                  |
| ------------- | ------------------------------------------------------------------------------------------- |
| `moments`     | Mean, variance, skewness, kurtosis; individual and separate-call workloads                  |
| `dispersion`  | Range, IQR, MAD, standard deviation, CV; individual and separate-call workloads             |
| `quantiles`   | Median, a single Type 7 quantile, and separate versus batched quantile queries              |
| `empirical`   | ECDF construction, single CDF/survival/inverse-CDF queries, and varied CDF/survival batches |
| `association` | Covariance, Pearson, Spearman, Kendall tau-b, and covariance/correlation matrices           |
| `regression`  | Simple linear fit, residuals, and scalar prediction                                         |
| `transform`   | Demeaning, sample z-scores, and one-based average ranks                                     |

## Input data

Seeded inputs use uniform random observations in `[-5, 5)` (`random`) and the
same draws shifted by `1e12` (`large_offset`). Paired and matrix inputs draw
successive series from the same RNG. See [support/data.rs](support/data.rs).

Most scalar and transform workloads use 32, 1,024, 32,768, and 1,048,576
observations. Kendall adds intermediate sizes; matrices vary observation and
variable counts separately. Exact workloads and sizes live in each target.

Variance, standard deviation, CV, covariance (scalar and matrix), and z-scores
use `Normalization::Sample`. Skewness and kurtosis use `ShapeEstimator::Type2`,
with `Kurtosis::Excess` for kurtosis.
These choices also apply to the separate-call workloads.

## Timing and throughput

Each iteration measures the complete public call, including its internal
validation, copying, allocation, and sorting where applicable. Returned vectors,
matrices, and ECDFs are also dropped inside the timed loop. Multi-call workloads
include all calls and result construction in that iteration.

Input generation happens before timing. ECDF queries use an already constructed
distribution; regression prediction and residual benchmarks use an already
fitted model. Varied ECDF batches return fixed-size arrays.

Criterion's elements-per-second figure uses these units:

| Workload                                             | Elements per iteration                 |
| ---------------------------------------------------- | -------------------------------------- |
| Scalar statistics, transforms, and ECDF construction | `n` input values                       |
| Paired association, regression fit, and residuals    | `n` aligned pairs                      |
| Matrix association                                   | `n * p` input values across `p` series |
| Single ECDF query                                    | One query                              |
| Varied ECDF batch                                    | Number of queries in the batch         |

Throughput counts logical input size or query count, rather than arithmetic
operations or repeated passes, including in multi-call workloads. Compare
elapsed time for the same workload and shape; throughput units differ across
categories. Matrix work grows as `O(n * p²)` even though its input size is
`n * p`.
