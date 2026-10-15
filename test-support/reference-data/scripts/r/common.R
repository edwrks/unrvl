# Check the environment before writing any reference files.
check_environment <- function() {
  lock <- renv::lockfile_read("renv.lock")

  stopifnot(as.character(getRversion()) == lock$R$Version)
  stopifnot(getOption("repos")[["CRAN"]] == lock$R$Repositories[["CRAN"]])
  stopifnot(lock$Packages$e1071$Version == "1.7-17")

  for (package in names(lock$Packages)) {
    actual <- utils::packageDescription(package, fields = "Version")
    if (!identical(actual, lock$Packages[[package]]$Version)) {
      stop("Package version differs from renv.lock: ", package)
    }
  }
}

fixture_files <- function(family) {
  files <- list.files(
    file.path("../../data/fixtures/r", family),
    pattern = "\\.csv$",
    full.names = TRUE
  )

  if (length(files) == 0L) {
    stop("No fixtures found for ", family)
  }

  sort(files, method = "radix")
}

case_name <- function(path) {
  tools::file_path_sans_ext(basename(path))
}

read_fixture <- function(path, columns, minimum_n = 1L) {
  values <- read_numeric_fixture(path, minimum_n)

  if (!identical(names(values), columns)) {
    stop("Invalid columns in ", path, "; expected: ", paste(columns, collapse = ", "))
  }

  values
}

read_matrix_fixture <- function(path, minimum_n = 2L) {
  read_numeric_fixture(path, minimum_n)
}

read_numeric_fixture <- function(path, minimum_n) {
  # Reject ragged records before read.csv can infer row names or fill fields.
  fields <- utils::count.fields(
    path, sep = ",", quote = "\"", comment.char = "", blank.lines.skip = FALSE
  )

  if (length(fields) == 0L || anyNA(fields) || fields[1L] < 1L || any(fields != fields[1L])) {
    stop("Invalid fixture ", path, ": expected a rectangular CSV table")
  }

  values <- tryCatch(
    utils::read.csv(
      path,
      colClasses = "numeric",
      check.names = FALSE,
      row.names = NULL,
      fill = FALSE,
      blank.lines.skip = FALSE
    ),
    error = function(error) {
      stop("Invalid fixture ", path, ": ", conditionMessage(error))
    }
  )

  columns <- names(values)
  if (length(columns) == 0L || any(!nzchar(trimws(columns))) || anyDuplicated(columns) > 0L) {
    stop("Invalid columns in ", path, ": names must be nonempty and unique")
  }

  if (nrow(values) < minimum_n) {
    stop("Invalid fixture ", path, ": need at least ", minimum_n, " observations")
  }

  for (column in columns) {
    if (!all(is.finite(values[[column]]))) {
      stop("Invalid fixture ", path, ": ", column, " must contain finite numbers")
    }
  }

  values
}

checked_reference <- function(row, path) {
  row <- tryCatch(row, error = function(error) {
      stop("Cannot compute reference for ", path, ": ", conditionMessage(error))
    }
  )

  for (statistic in setdiff(names(row), "case")) {
    if (!is.numeric(row[[statistic]]) || !all(is.finite(row[[statistic]]))) {
      stop("Non-finite reference for ", path, ": ", statistic)
    }
  }

  row
}

write_reference <- function(rows, filename) {
  # Explicit 17 significant digits round-trip an R double as a Rust f64.
  formatted <- lapply(rows, function(column) {
    if (is.numeric(column)) sprintf("%.17g", column) else column
  })

  utils::write.table(
    as.data.frame(formatted, check.names = FALSE),
    file = file.path("../../data/r", filename),
    sep = ",",
    quote = FALSE,
    row.names = FALSE,
    eol = "\n",
    fileEncoding = "UTF-8"
  )
}
