#!/usr/bin/env Rscript
# Independent Heckit reference fit using base R only.
#
# We deliberately avoid `sampleSelection` (its dependency `nloptr` does not
# compile on many setups, including Apple Silicon). All formulas below are
# textbook Heckman (1979) -- two-step via probit + augmented OLS, plus joint
# MLE via `optim`. The same likelihood is in `censtrunc/heckit.py::
# _neg_loglik_heckit`, so this script provides an independent numerical check
# of our implementation: two optimisers (R's `optim` vs scipy's L-BFGS-B) on
# the same data must converge to the same point.
#
# Called as:
#     Rscript fit_heckit.R /path/to/data.csv
# CSV columns: y (NA when unselected), s (0/1), x1, x2, z1, z2.

suppressMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
csv_path <- args[1]
df <- read.csv(csv_path)
df$s <- as.integer(df$s)

# ----------------------------------------------------------------------
# Two-step Heckman estimator
# ----------------------------------------------------------------------

# 1. Probit of s on Z.
probit <- glm(s ~ x1 + x2 + z1 + z2, data = df, family = binomial(link = "probit"))
gamma_hat <- coef(probit)

# 2. Inverse Mills ratio for the selected subsample.
df_sel <- subset(df, s == 1)
Zg <- as.numeric(predict(probit, newdata = df_sel, type = "link"))
lambda_hat <- dnorm(Zg) / pnorm(Zg)

# 3. Augmented OLS of y on (X, lambda).
ols <- lm(y ~ x1 + x2 + lambda_hat, data = df_sel)
beta_aug <- coef(ols)
beta_hat <- beta_aug[c("(Intercept)", "x1", "x2")]
sigma_eu <- unname(beta_aug["lambda_hat"])

# Heckman-corrected residual variance.
e <- residuals(ols)
n_sel <- nrow(df_sel)
delta <- lambda_hat * (lambda_hat + Zg)
sigma_sq <- sum(e^2) / n_sel + sigma_eu^2 * mean(delta)
sigma_hat <- sqrt(sigma_sq)
rho_hat <- sigma_eu / sigma_hat

twostep <- list(
  selection = as.list(gamma_hat),
  outcome   = as.list(beta_hat),
  sigma     = unname(sigma_hat),
  rho       = unname(rho_hat),
  sigma_eu  = sigma_eu
)

# ----------------------------------------------------------------------
# Joint MLE  --  same likelihood as censtrunc/heckit.py
# ----------------------------------------------------------------------

neg_loglik <- function(theta, X, y, Z, s) {
  sigma_e <- theta[1]
  rho     <- theta[2]
  beta    <- theta[3:(2 + ncol(X))]
  gamma   <- theta[(3 + ncol(X)):length(theta)]
  if (sigma_e <= 0 || abs(rho) >= 0.999) return(1e16)

  Zg <- as.numeric(Z %*% gamma)
  ll <- 0
  i0 <- s == 0
  if (any(i0)) ll <- ll + sum(pnorm(-Zg[i0], log.p = TRUE))
  i1 <- s == 1
  if (any(i1)) {
    e <- y[i1] - as.numeric(X[i1, ] %*% beta)
    z_inner <- (Zg[i1] + (rho / sigma_e) * e) / sqrt(1 - rho^2)
    ll <- ll + sum(pnorm(z_inner, log.p = TRUE))
    ll <- ll - 0.5 * sum(i1) * log(2 * pi * sigma_e^2)
    ll <- ll - 0.5 * sum(e^2) / sigma_e^2
  }
  -ll
}

X_mat <- model.matrix(~ x1 + x2, data = df)
Z_mat <- model.matrix(~ x1 + x2 + z1 + z2, data = df)
y_full <- df$y
y_full[is.na(y_full)] <- 0  # the s = 0 rows do not enter the y-part of LL.

# Warm start from the two-step.
theta0 <- c(sigma_hat, max(min(rho_hat, 0.9), -0.9), beta_hat, gamma_hat)
res <- optim(
  theta0, neg_loglik,
  X = X_mat, y = y_full, Z = Z_mat, s = df$s,
  method = "L-BFGS-B",
  lower = c(1e-6, -0.999, rep(-Inf, length(theta0) - 2)),
  upper = c(Inf,   0.999, rep( Inf, length(theta0) - 2)),
  control = list(maxit = 500, factr = 1e7)
)
theta_mle <- res$par
k_x <- ncol(X_mat); k_z <- ncol(Z_mat)
beta_mle  <- theta_mle[3:(2 + k_x)]; names(beta_mle)  <- colnames(X_mat)
gamma_mle <- theta_mle[(3 + k_x):(2 + k_x + k_z)]; names(gamma_mle) <- colnames(Z_mat)

mle <- list(
  selection = as.list(gamma_mle),
  outcome   = as.list(beta_mle),
  sigma     = unname(theta_mle[1]),
  rho       = unname(theta_mle[2]),
  logLik    = -res$value,
  converged = (res$convergence == 0)
)

cat(toJSON(list(twostep = twostep, mle = mle), digits = 10, auto_unbox = TRUE))
