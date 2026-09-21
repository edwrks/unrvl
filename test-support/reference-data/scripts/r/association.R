generate_association <- function() {
  rows <- list()
  for (path in fixture_files("association")) {
    case <- case_name(path)
    fixture <- read_fixture(path, c("x", "y"), minimum_n = 2L)
    rows[[case]] <- checked_reference(data.frame(
      case = case,
      n = nrow(fixture),
      covariance = stats::cov(fixture$x, fixture$y),
      correlation = stats::cor(fixture$x, fixture$y)
    ), path)
  }
  write_reference(do.call(rbind, rows), "association.csv")
}
