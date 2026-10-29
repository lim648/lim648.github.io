#!/usr/bin/env Rscript

run_or_stop <- function(command, arguments) {
  status <- system2(command, arguments)
  if (!identical(status, 0L)) {
    stop("Command failed: ", command, " ", paste(arguments, collapse = " "))
  }
}

rscript <- file.path(R.home("bin"), "Rscript")
run_or_stop(rscript, shQuote(file.path("R Code", "generate_synthetic_data.R")))
run_or_stop(rscript, shQuote(file.path("R Code", "TSMA_Tutorial_Code.R")))
run_or_stop(
  "quarto",
  c("render", shQuote("TSMA_Tutorial.qmd"), "--output", shQuote("TSMA_Tutorial.html"))
)

cat("Tutorial data, results, and HTML were regenerated successfully.\n")
