#!/usr/bin/env Rscript
# Reference fit for the Heckman selection model via the canonical R package
# `sampleSelection` (Toomet & Henningsen, JSS 27(7), 2008). This is the
# standard Heckit reference in the R ecosystem: it implements both the
# two-step (Heckman 1979) and joint-MLE estimators on the very log-likelihood
# we use in `censtrunc/heckit.py::_neg_loglik_heckit`, so the two estimates
# must agree to optimiser tolerance.
#
# Called as:
#     Rscript fit_heckit.R /path/to/data.csv
# CSV columns: y (NA when unselected), s (0/1), x1, x2, z1, z2.

suppressMessages({
  library(sampleSelection)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
csv_path <- args[1]
df <- read.csv(csv_path)
df$s <- as.integer(df$s)

selection_formula <- s ~ x1 + x2 + z1 + z2
outcome_formula   <- y ~ x1 + x2

# The design-matrix column names for the two equations, used to label the
# coefficient vectors below regardless of the estimator's internal layout.
sel_names <- colnames(model.matrix(selection_formula, data = df))
out_names <- colnames(model.matrix(outcome_formula,
                                   data = df[!is.na(df$y), ]))

# Two-step Heckman.
m_2s <- selection(
  selection = selection_formula,
  outcome   = outcome_formula,
  data      = df,
  method    = "2step"
)

# Joint MLE.
m_ml <- selection(
  selection = selection_formula,
  outcome   = outcome_formula,
  data      = df,
  method    = "ml"
)

# `sampleSelection` keeps coefficients in a single long vector indexed by
# `param$index$betaS` (selection equation) and `param$index$betaO` (outcome
# equation). The two-step estimator reads them through `coef()`; the MLE
# stores them on `$estimate`.
get_all <- function(m) if (m$method == "2step") coef(m) else m$estimate

sel_coefs <- function(m) {
  vec <- get_all(m)[m$param$index$betaS]
  names(vec) <- sel_names
  as.list(vec)
}

out_coefs <- function(m) {
  vec <- get_all(m)[m$param$index$betaO]
  names(vec) <- out_names
  as.list(vec)
}

extract_2s <- function(m) {
  list(
    selection = sel_coefs(m),
    outcome   = out_coefs(m),
    sigma     = unname(m$sigma),
    rho       = unname(m$rho)
  )
}

extract_ml <- function(m) {
  est <- m$estimate
  list(
    selection = sel_coefs(m),
    outcome   = out_coefs(m),
    sigma     = unname(est["sigma"]),
    rho       = unname(est["rho"]),
    logLik    = as.numeric(logLik(m))
  )
}

cat(toJSON(
  list(twostep = extract_2s(m_2s), mle = extract_ml(m_ml)),
  digits = 10, auto_unbox = TRUE
))
