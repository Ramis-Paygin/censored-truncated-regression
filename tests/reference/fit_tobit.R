#!/usr/bin/env Rscript
#
# Reference fit for cross-validating the `censtrunc` Python package against R.
#
# Reads a CSV with a column `y` and one or more regressor columns, then fits:
#   1. a two-sided Tobit via survival::survreg (Gaussian, interval2 censoring);
#   2. a left-truncated regression via truncreg::truncreg.
# Coefficients and scale are printed to stdout as JSON for the Python test.
#
# survival::survreg with dist="gaussian" *is* the maximum-likelihood Tobit
# estimator (it is exactly what AER::tobit wraps); using it directly avoids
# AER's heavy dependency chain (car -> lme4 -> nloptr).
#
# Usage:
#   Rscript fit_tobit.R <csv_path> <left> <right>
#
# Requires the R packages: survival (ships with R), truncreg, jsonlite.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("usage: Rscript fit_tobit.R <csv_path> <left> <right>")
}
csv_path <- args[1]
left  <- as.numeric(args[2])
right <- as.numeric(args[3])

suppressMessages({
  library(survival)
  library(truncreg)
  library(jsonlite)
})

d <- read.csv(csv_path)
xnames <- setdiff(names(d), "y")
rhs <- paste(xnames, collapse = " + ")

# --- 1. Two-sided Tobit via interval2 censoring -------------------------------
# interval2 encoding: lower=NA  -> left-censored (event <= upper);
#                     upper=NA  -> right-censored (event >= lower);
#                     lower=upper -> exact observation.
y <- d$y
is_left  <- y <= left + 1e-9
is_right <- y >= right - 1e-9
is_exact <- !is_left & !is_right

lower <- rep(NA_real_, length(y))
upper <- rep(NA_real_, length(y))
lower[is_exact] <- y[is_exact]; upper[is_exact] <- y[is_exact]
lower[is_left]  <- NA;          upper[is_left]  <- left
lower[is_right] <- right;       upper[is_right] <- NA

surv_obj <- Surv(lower, upper, type = "interval2")
tob <- survreg(as.formula(paste("surv_obj ~", rhs)), data = d, dist = "gaussian")
tobit_out <- list(coef = as.list(coef(tob)), scale = unname(tob$scale))

# --- 2. Left-truncated regression --------------------------------------------
dt <- d[d$y > left, ]
tr <- truncreg(as.formula(paste("y ~", rhs)), data = dt,
               point = left, direction = "left")
truncreg_out <- list(coef = as.list(coef(tr)))  # includes 'sigma'

cat(toJSON(list(tobit = tobit_out, truncreg = truncreg_out),
           auto_unbox = TRUE, digits = 10))
