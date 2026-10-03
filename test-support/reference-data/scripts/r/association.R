generate_association <- function() {
  rows <- list()
  for (path in fixture_files("association")) {
    case <- case_name(path)
    fixture <- read_fixture(path, c("x", "y"), minimum_n = 2L)

    rows[[case]] <- checked_reference(data.frame(
      case = case,
      n = nrow(fixture),
      covariance = stats::cov(fixture$x, fixture$y),
      pearson = stats::cor(fixture$x, fixture$y, method = "pearson"),
      spearman = stats::cor(fixture$x, fixture$y, method = "spearman"),
      kendall = stats::cor(fixture$x, fixture$y, method = "kendall")
    ), path)
  }

  write_reference(do.call(rbind, rows), "association.csv")
}

generate_association_matrix <- function() {
  rows <- list()

  for (path in fixture_files("association-matrix")) {
    case <- case_name(path)
    fixture <- read_matrix_fixture(path, minimum_n = 2L)
    x <- as.matrix(fixture)

    covariance <- stats::cov(x)
    correlation <- stats::cor(x)
    p <- ncol(x)

    # Explicit row-major ordering; CSV indices are zero-based.
    indices <- data.frame(
      row = rep(seq_len(p), each = p),
      column = rep(seq_len(p), times = p)
    )
    coordinates <- as.matrix(indices)

    rows[[case]] <- checked_reference(data.frame(
      case = case,
      n = nrow(x),
      row = indices$row - 1L,
      column = indices$column - 1L,
      covariance = covariance[coordinates],
      correlation = correlation[coordinates]
    ), path)
  }

  write_reference(do.call(rbind, rows), "association-matrix.csv")
}
