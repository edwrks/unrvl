generate_empirical <- function() {
  query_labels_by_case <- list(
    baseline = c(
      "-0.03", "-0.02", "-0.01", "0", "0.005", "0.01",
      "0.02", "0.03", "0.035", "0.04", "0.05"
    ),
    constant = c("-3", "-2", "-1"),
    singleton = c("2", "2.5", "3"),
    ties = c("-2", "-1", "-0.5", "0", "1", "2", "3", "4", "5"),
    `two-point` = c("-1", "0", "2.5", "5", "7.5", "10", "11")
  )

  endpoint_probability_labels <- c(
    "0", "0.01", "0.05", "0.25", "0.5", "0.75", "0.95", "0.99", "1"
  )

  probability_labels_by_case <- list(
    baseline = c(
      "0", "0.01", "0.05", "0.19", "0.2", "0.21", "0.25",
      "0.39", "0.4", "0.41", "0.5", "0.59", "0.6", "0.61",
      "0.75", "0.79", "0.8", "0.81", "0.95", "0.99", "1"
    ),
    constant = endpoint_probability_labels,
    singleton = endpoint_probability_labels,
    ties = c(
      "0", "0.01", "0.05", "0.22", "0.2222222222222222", "0.23",
      "0.25", "0.44", "0.4444444444444444", "0.45", "0.5", "0.75",
      "0.77", "0.7777777777777778", "0.78", "0.95", "0.99", "1"
    ),
    `two-point` = c(
      "0", "0.01", "0.05", "0.25", "0.49", "0.5", "0.51",
      "0.75", "0.95", "0.99", "1"
    )
  )

  paths <- fixture_files("empirical")
  cases <- vapply(paths, case_name, character(1), USE.NAMES = FALSE)

  if (!identical(cases, sort(names(query_labels_by_case), method = "radix"))) {
    stop("Empirical fixtures and query definitions must have matching case names")
  }

  if (!identical(cases, sort(names(probability_labels_by_case), method = "radix"))) {
    stop("Empirical fixtures and probability definitions must have matching case names")
  }

  evaluation_rows <- list()
  quantile_rows <- list()

  for (path in paths) {
    case <- case_name(path)
    x <- read_fixture(path, "value")$value
    query_labels <- query_labels_by_case[[case]]
    queries <- as.numeric(query_labels)

    if (!is.numeric(queries) || length(queries) == 0L ||
        !all(is.finite(queries)) || is.unsorted(queries, strictly = TRUE)) {
      stop("Queries must be finite, nonempty, and strictly increasing for ", case)
    }

    probability_labels <- probability_labels_by_case[[case]]
    probabilities <- as.numeric(probability_labels)

    if (length(probabilities) == 0L || !all(is.finite(probabilities)) ||
        any(probabilities < 0 | probabilities > 1) ||
        is.unsorted(probabilities, strictly = TRUE)) {
      stop("Probabilities must be nonempty, finite, strictly increasing, and in [0, 1] for ", case)
    }

    evaluation <- checked_reference(data.frame(
      case = case,
      query = queries,
      cdf = stats::ecdf(x)(queries),
      sf = vapply(queries, function(query) mean(x > query), numeric(1))
    ), path)

    quantile <- checked_reference(data.frame(
      case = case,
      probability = probabilities,
      value = stats::quantile(x, probs = probabilities, names = FALSE, type = 1)
    ), path)

    # Validate inputs as numbers, then preserve their declared decimal labels
    # when writing the CSV files.
    evaluation$query <- query_labels
    quantile$probability <- probability_labels
    evaluation_rows[[case]] <- evaluation
    quantile_rows[[case]] <- quantile
  }

  write_reference(do.call(rbind, evaluation_rows), "empirical.csv")
  write_reference(do.call(rbind, quantile_rows), "empirical-quantiles.csv")
}
