generate_moments <- function() {
  rows <- list()

  for (path in fixture_files("moments")) {
    case <- case_name(path)
    x <- read_fixture(path, "value", minimum_n = 4L)$value
    rows[[case]] <- checked_reference(data.frame(
      case = case,
      n = length(x),
      mean = base::mean(x),
      variance = stats::var(x),
      # Type 2 selects the adjusted sample skewness and excess kurtosis.
      skewness = e1071::skewness(x, type = 2),
      excess_kurtosis = e1071::kurtosis(x, type = 2)
    ), path)
  }

  write_reference(do.call(rbind, rows), "moments.csv")
}
