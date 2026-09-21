# Run from scripts/r (the Docker working directory). No randomness or clock
# values enter the output, and fixture enumeration uses bytewise ordering.
invisible(Sys.setlocale("LC_ALL", "C"))
source("common.R")
source("moments.R")
source("dispersion.R")
source("quantiles.R")
source("association.R")
source("regression.R")

check_environment()
dir.create("../../data/r", recursive = TRUE, showWarnings = FALSE)
generate_moments()
generate_dispersion()
generate_quantiles()
generate_association()
generate_regression()

packages <- sort(names(renv::lockfile_read("renv.lock")$Packages), method = "radix")
write_reference(data.frame(
  component = c("R", packages),
  version = c(as.character(getRversion()), vapply(packages, function(package) {
    utils::packageDescription(package, fields = "Version")
  }, character(1)))
), "versions.csv")
message("Generated R conformance references in data/r/.")
