#!/usr/bin/env Rscript
#
# Reference fit for cross-validating the `censtrunc` Python package against R.
#
# Reads a CSV with a column `y` and one or more regressor columns, then fits:
#   1. a two-sided Tobit via AER::tobit  (censored regression);
#   2. a left-truncated regression via truncreg::truncreg.
# Coefficients and scale are printed to stdout as JSON for the Python test to
# parse.
#
# Usage:
#   Rscript fit_tobit.R <csv_path> <left> <right>
#
# Requires the R packages: AER, truncreg, jsonlite.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("usage: Rscript fit_tobit.R <csv_path> <left> <right>")
}
csv_path <- args[1]
left  <- as.numeric(args[2])
right <- as.numeric(args[3])

suppressMessages({
  library(AER)
  library(truncreg)
  library(jsonlite)
})

d <- read.csv(csv_path)
xnames <- setdiff(names(d), "y")
form <- as.formula(paste("y ~", paste(xnames, collapse = " + ")))

# 1. Two-sided Tobit (censored regression).
tob <- tobit(form, left = left, right = right, data = d)
tobit_out <- list(coef = as.list(coef(tob)), scale = unname(tob$scale))

# 2. Left-truncated regression (truncreg supports a single truncation point).
dt <- d[d$y > left, ]
tr <- truncreg(form, data = dt, point = left, direction = "left")
trc <- coef(tr)
truncreg_out <- list(coef = as.list(trc))  # includes 'sigma' as a named element

cat(toJSON(list(tobit = tobit_out, truncreg = truncreg_out),
           auto_unbox = TRUE, digits = 10))
