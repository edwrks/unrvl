generate_regression <- function() {
  rows <- list()
  for (path in fixture_files("regression")) {
    case <- case_name(path)
    fixture <- read_fixture(path, c("x", "y"), minimum_n = 2L)
    # Use lm's defaults for every fixture, without centering or fallbacks.
    fit <- stats::lm(y ~ x, data = fixture)
    coefficients <- stats::coef(fit)
    rows[[case]] <- checked_reference(data.frame(
      case = case,
      n = nrow(fixture),
      intercept = coefficients[["(Intercept)"]],
      # x is the predictor and y the response; the x coefficient is slope/beta.
      slope = coefficients[["x"]],
      r_squared = summary(fit)$r.squared
    ), path)
  }

  write_reference(do.call(rbind, rows), "regression.csv")
}
