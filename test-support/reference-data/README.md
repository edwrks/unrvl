# Reference Data

Test reference datasets and loaders for the workspace's statistical tests.

- **NIST:** published datasets and certified numerical reference values.
- **R conformance:** small, hand-authored fixture families and committed
  reference values for statistical definitions.
- **Rust unit/property tests:** undefined cases, floating-point stress, and
  invariants.

Reference data stays separate from production crates. See the
[data guide](data/README.md) for fixture families and NIST integrity checks.

## Regenerate R References

From the repository root:

```sh
docker compose -f test-support/reference-data/scripts/r/docker/compose.yaml run --build --rm generate
```

This writes six tables to `data/r/` using a pinned R/CRAN/e1071 environment and
`renv.lock`. See the [R generator guide](scripts/r/README.md) for schemas,
statistical conventions, reproducibility, and source organization.

Every fixture in an R family must produce finite, usable values for every
statistic in that family. Invalid cases fail generation.

