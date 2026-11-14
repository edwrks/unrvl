# ==============================================================================
# Meta and Defaults
# ==============================================================================

# Display available recipes.
[group("meta")]
default:
    @just --list

# ==============================================================================
# Formatting
# ==============================================================================

# Formats all Rust code.
[group("format")]
fmt:
    @echo "Formatting Rust code"
    cargo fmt --all

# Checks formatting without changing files.
[group("format")]
fmt-check:
    @echo "Checking Rust formatting"
    cargo fmt --all -- --check

# ==============================================================================
# Checks
# ==============================================================================

# Checks the project compiles without building final binaries.
[group("check")]
check:
    @echo "Checking Rust project"
    cargo check --locked --workspace --all-targets --all-features

# Runs Clippy with warnings treated as errors.
[group("check")]
clippy:
    @echo "Running Clippy"
    cargo clippy --locked --workspace --all-targets --all-features -- -D warnings

# ==============================================================================
# Tests
# ==============================================================================

# Runs all tests.
[group("test")]
test:
    @echo "Running tests"
    cargo test --locked --workspace --all-targets --all-features

# Runs doc tests.
[group("test")]
test-doc:
    @echo "Running doc tests"
    cargo test --locked --workspace --doc --all-features

# ==============================================================================
# Coverage
# ==============================================================================

# Runs test coverage.
[group("coverage")]
coverage:
    @echo "Running coverage"
    cargo llvm-cov --locked --workspace --all-features

[group("coverage")]
coverage-html:
    @echo "Generating coverage report"
    cargo llvm-cov --locked --workspace --all-features --html --open

# ==============================================================================
# Documentation
# ==============================================================================

# Builds docs and fails on rustdoc warnings.
[group("docs")]
doc:
    @echo "Building docs"
    RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --all-features --no-deps

# Opens docs locally.
[group("docs")]
doc-open:
    cargo doc --locked --workspace --all-features --no-deps --open

# ==============================================================================
# Quality
# ==============================================================================

# Runs the standard local quality gate.
[group("quality")]
qc: fmt-check check clippy test test-doc doc

# ==============================================================================
# Development
# ==============================================================================

# Runs a package's default binary.
[group("dev")]
run app:
    cargo run -p {{ app }}

# Runs a binary from a specific package.
[group("dev")]
run-bin app bin:
    cargo run -p {{ app }} --bin {{ bin }}

# Removes build artifacts.
[group("dev")]
clean:
    cargo clean

# ==============================================================================
# Maintenance
# ==============================================================================

# Updates Cargo.lock without changing Cargo.toml requirements.
[group("maintenance")]
update-lockfile:
    cargo update
