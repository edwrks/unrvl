generate_regression <- function() {
  fit_rows <- list()
  observation_rows <- list()

  for (path in fixture_files("regression")) {
    case <- case_name(path)
    fixture <- read_fixture(path, c("x", "y"), minimum_n = 2L)

    # Fit y as a straight line in x, including an intercept, using R's defaults.
    fit <- stats::lm(y ~ x, data = fixture)
    coefficients <- stats::coef(fit)

    # One row per fixture describes the fitted line and its overall fit.
    fit_rows[[case]] <- checked_reference(data.frame(
      case = case,
      n = nrow(fixture),
      intercept = coefficients[["(Intercept)"]],
      slope = coefficients[["x"]],
      r_squared = summary(fit)$r.squared
    ), path)

    # One row per observation keeps its prediction and residual together.
    # Subtract one from R's row positions to match Rust's zero-based indices.
    observation_rows[[case]] <- checked_reference(data.frame(
      case = case,
      observation = seq_len(nrow(fixture)) - 1L,
      prediction = as.numeric(stats::predict(fit, newdata = fixture)),
      residual = as.numeric(stats::residuals(fit))
    ), path)
  }

  # Stack the cases into one table of fits and one table of observations.
  write_reference(do.call(rbind, fit_rows), "regression.csv")
  write_reference(do.call(rbind, observation_rows), "regression-observations.csv")
}
