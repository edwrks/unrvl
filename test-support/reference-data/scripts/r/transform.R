generate_transform <- function() {
  rows <- list()

  for (path in fixture_files("transform")) {
    case <- case_name(path)
    x <- read_fixture(path, "value", minimum_n = 2L)$value

    # Keep all three transformations aligned with the original observations.
    rows[[case]] <- checked_reference(data.frame(
      case = case,
      observation = seq_along(x) - 1L,
      demeaned = x - base::mean(x),

      # R's defaults center and divide by the sample standard deviation.
      z_score = as.numeric(base::scale(x)),

      # Statistical ranks start at one; observation indices above start at zero.
      rank = base::rank(x, ties.method = "average")
    ), path)
  }

  write_reference(do.call(rbind, rows), "transform.csv")
}
