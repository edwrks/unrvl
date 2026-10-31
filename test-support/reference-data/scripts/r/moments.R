generate_moments <- function() {
  rows <- list()

  for (path in fixture_files("moments")) {
    case <- case_name(path)
    x <- read_fixture(path, "value", minimum_n = 4L)$value
    deviations <- x - base::mean(x)

    # Both kurtosis conventions use these Joanes-Gill estimates.
    excess_kurtosis_type_1 <- e1071::kurtosis(x, type = 1)
    excess_kurtosis_type_2 <- e1071::kurtosis(x, type = 2)
    excess_kurtosis_type_3 <- e1071::kurtosis(x, type = 3)

    rows[[case]] <- checked_reference(data.frame(
      case = case,
      n = length(x),
      mean = base::mean(x),

      # Variance denominators are n - 1, n, and n - 2.
      variance = stats::var(x),
      population_variance = base::mean(deviations^2),
      variance_ddof_2 = base::sum(deviations^2) / (length(x) - 2L),

      # Joanes-Gill skewness estimators.
      skewness_type_1 = e1071::skewness(x, type = 1),
      skewness_type_2 = e1071::skewness(x, type = 2),
      skewness_type_3 = e1071::skewness(x, type = 3),

      # Excess kurtosis.
      excess_kurtosis_type_1 = excess_kurtosis_type_1,
      excess_kurtosis_type_2 = excess_kurtosis_type_2,
      excess_kurtosis_type_3 = excess_kurtosis_type_3,

      # Pearson kurtosis shifts each estimator by three.
      pearson_kurtosis_type_1 = excess_kurtosis_type_1 + 3,
      pearson_kurtosis_type_2 = excess_kurtosis_type_2 + 3,
      pearson_kurtosis_type_3 = excess_kurtosis_type_3 + 3
    ), path)
  }

  write_reference(do.call(rbind, rows), "moments.csv")
}
