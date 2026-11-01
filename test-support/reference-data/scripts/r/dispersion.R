generate_dispersion <- function() {
  rows <- list()

  for (path in fixture_files("dispersion")) {
    case <- case_name(path)
    x <- read_fixture(path, "value", minimum_n = 2L)$value
    sample_std_dev <- stats::sd(x)
    population_std_dev <- sqrt(base::mean((x - base::mean(x))^2))

    rows[[case]] <- checked_reference(data.frame(
      case = case,
      n = length(x),
      range = max(x) - min(x),

      # Type 7 is the quantile convention used by the conformance suite.
      iqr = stats::IQR(x, type = 7),

      # Rust exposes raw MAD. Disable R's normal-consistency scaling.
      mad = stats::mad(x, center = stats::median(x), constant = 1),

      std_dev = sample_std_dev,
      population_std_dev = population_std_dev,

      # Signed CV: a negative mean produces a negative ratio, not a percentage.
      cv = sample_std_dev / base::mean(x),
      population_cv = population_std_dev / base::mean(x)
    ), path)
  }

  write_reference(do.call(rbind, rows), "dispersion.csv")
}
