# Statistics benchmarks

Run one category at a time in Cargo's benchmark profile:

```sh
cargo bench -p unrvl-stats --bench moments
cargo bench -p unrvl-stats --bench dispersion
cargo bench -p unrvl-stats --bench quantiles
cargo bench -p unrvl-stats --bench empirical
cargo bench -p unrvl-stats --bench association
```

Filter a benchmark group when only one workload is needed:

```sh
cargo bench -p unrvl-stats --bench association -- 'association/matrix'
cargo bench -p unrvl-stats --bench empirical -- '/queries/'
```

Criterion records results under `target/criterion`. A full measurement run can
take several minutes per category. To exercise every benchmark once without
collecting performance measurements:

```sh
cargo bench -p unrvl-stats -- --test
```

## Workloads

| Target        | Operations                                                                                    |
| ------------- | --------------------------------------------------------------------------------------------- |
| `moments`     | Mean, variance, skewness, excess kurtosis, and four separate calls                            |
| `dispersion`  | Range, IQR, MAD, standard deviation, CV, and five separate calls                              |
| `quantiles`   | Median, a single Type 7 quantile, seven separate queries, and the same seven batched queries  |
| `empirical`   | ECDF construction, then CDF, survival, and inverse-CDF queries on an already constructed ECDF |
| `association` | Covariance, Pearson, Spearman, Kendall tau-b, covariance matrices, and correlation matrices   |

## Input data

| Case           | Values                                  |
| -------------- | --------------------------------------- |
| `random`       | Uniform random observations in [-5, 5)  |
| `large_offset` | The same observations shifted by `1e12` |

All benchmark targets use these two cases. Each input dataset starts a seeded
`StdRng`. Paired inputs and matrix series draw successive samples from that RNG,
giving different series without cloning or shuffling. The offset cases use the
same underlying draws for comparison.

Input generation and ECDF construction for query benchmarks happen outside the
timed loops.

ECDF queries include repeated single queries and batches of nine varied CDF or
survival queries covering existing observations, intermediate values, and both
tails. Batch outputs use fixed-size arrays.

Shared input definitions live in `support/data.rs`. The subdirectory keeps the
helper separate from the standalone benchmark targets discovered by Cargo.

Most scalar workloads use 32, 1,024, 32,768, and 1,048,576 observations. Kendall
also includes 256 and 4,096 to show how sorting and inversion counting scale
through intermediate sample sizes. Matrices vary observation count and variable
count separately.

## Timing and throughput

Each iteration measures the complete public call, including its internal
validation, copying, allocation, and sorting where applicable. Returned vectors,
matrices, and ECDFs are also dropped inside the timed loop. Multi-call workloads
include all calls and result construction in that iteration.

Criterion's elements-per-second figure uses these units:

| Workload                                          | Elements per iteration                                    |
| ------------------------------------------------- | --------------------------------------------------------- |
| Scalar statistics, including multi-call workloads | `n` input values                                          |
| Paired association                                | `n` paired observation positions, each containing x and y |
| Matrix association                                | `n * p` input values across `p` series                    |
| ECDF construction                                 | `n` input values                                          |
| Single ECDF query                                 | One query                                                 |
| Varied ECDF query batch                           | Nine queries                                              |

Throughput counts logical input size or query count, rather than arithmetic
operations or repeated passes. Compare elapsed time for the same workload and
shape; throughput figures across different categories have different meanings.
Matrix work grows as `O(n * p²)` even though its input size is `n * p`.
