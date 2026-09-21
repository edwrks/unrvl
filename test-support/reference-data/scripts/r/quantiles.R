generate_quantiles <- function() {
  probability_labels <- c(
    "0", "0.01", "0.05", "0.25", "0.5", "0.75", "0.95", "0.99", "1"
  )

  probabilities <- as.numeric(probability_labels)

  rows <- list()

  for (path in fixture_files("quantiles")) {
    case <- case_name(path)
    x <- read_fixture(path, "value")$value

    row <- checked_reference(data.frame(
      case = case,
      probability = probabilities,
      # Explicit Type 7 fixes the interpolation convention.
      value = stats::quantile(x, probs = probabilities, names = FALSE, type = 7)
    ), path)

    # Validate probabilities as numbers, then preserve their declared decimal
    # labels when writing the CSV.
    row$probability <- probability_labels
    rows[[case]] <- row
  }

  write_reference(do.call(rbind, rows), "quantiles.csv")
}
