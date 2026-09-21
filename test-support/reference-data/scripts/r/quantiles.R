generate_quantiles <- function() {
  probabilities <- c(0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1)
  rows <- list()
  for (path in fixture_files("quantiles")) {
    case <- case_name(path)
    x <- read_fixture(path, "value")$value
    rows[[case]] <- checked_reference(data.frame(
      case = case,
      probability = probabilities,
      # Explicit Type 7 fixes the interpolation convention.
      value = stats::quantile(x, probs = probabilities, names = FALSE, type = 7)
    ), path)
  }
  write_reference(do.call(rbind, rows), "quantiles.csv")
}
