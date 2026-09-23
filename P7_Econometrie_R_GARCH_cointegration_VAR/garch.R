#!/usr/bin/env Rscript
# ============================================================
# garch.R
# =======
# Modele GARCH(1,1), simule ET estime PAR MAXIMUM DE VRAISEMBLANCE
# implemente a la main (optim() sur la log-vraisemblance gaussienne),
# plutot qu'avec le package `rugarch` (indisponible sans acces a
# CRAN dans cet environnement -- voir README).
#
#   r_t        = sigma_t * z_t,          z_t ~ N(0,1) iid
#   sigma_t^2  = omega + alpha*r_{t-1}^2 + beta*sigma_{t-1}^2
#
# La simulation part de parametres CONNUS, ce qui permet de valider
# l'estimateur en comparant les parametres retrouves aux parametres
# "vrais" -- meme logique de validation que sur tous les projets
# precedents (Monte Carlo, EDP, TSRV, Black-Litterman).
# ============================================================

script_args <- commandArgs(trailingOnly = FALSE)
file_arg <- script_args[grep("^--file=", script_args)]
if (length(file_arg) > 0) {
  script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg[1])))
} else {
  script_dir <- getwd()
}
outputs_dir <- normalizePath(file.path(script_dir, "outputs"), winslash = "/", mustWork = FALSE)

set.seed(42)

# ------------------------------------------------------------
# 1) Simulation d'un processus GARCH(1,1)
# ------------------------------------------------------------
simulate_garch11 <- function(n, omega, alpha, beta, burn_in = 500) {
  n_total <- n + burn_in
  sigma2 <- numeric(n_total)
  r <- numeric(n_total)
  sigma2[1] <- omega / (1 - alpha - beta)   # variance inconditionnelle (point de depart naturel)
  r[1] <- sqrt(sigma2[1]) * rnorm(1)

  for (t in 2:n_total) {
    sigma2[t] <- omega + alpha * r[t - 1]^2 + beta * sigma2[t - 1]
    r[t] <- sqrt(sigma2[t]) * rnorm(1)
  }

  list(r = r[(burn_in + 1):n_total], sigma2 = sigma2[(burn_in + 1):n_total])
}

# ------------------------------------------------------------
# 2) Log-vraisemblance GARCH(1,1) (gaussienne) et estimation
# ------------------------------------------------------------
garch11_negloglik <- function(params, r) {
  omega <- params[1]; alpha <- params[2]; beta <- params[3]
  n <- length(r)

  # Penalite douce si la contrainte de stationnarite (alpha+beta<1)
  # est violee, plutot qu'une contrainte dure : garde optim() stable
  # meme si un pas d'optimisation explore temporairement cette zone.
  if (omega <= 0 || alpha < 0 || beta < 0 || (alpha + beta) >= 0.999) {
    return(1e10)
  }

  sigma2 <- numeric(n)
  sigma2[1] <- var(r)
  for (t in 2:n) {
    sigma2[t] <- omega + alpha * r[t - 1]^2 + beta * sigma2[t - 1]
  }

  loglik <- -0.5 * sum(log(2 * pi) + log(sigma2) + r^2 / sigma2)
  -loglik   # optim() minimise par defaut -> on retourne l'oppose
}

estimate_garch11 <- function(r) {
  start <- c(omega = 0.1 * var(r), alpha = 0.05, beta = 0.85)
  fit <- optim(start, garch11_negloglik, r = r, method = "L-BFGS-B",
               lower = c(1e-8, 0, 0), upper = c(Inf, 0.999, 0.999),
               control = list(maxit = 500))
  list(omega = fit$par[1], alpha = fit$par[2], beta = fit$par[3],
       loglik = -fit$value, convergence = fit$convergence)
}

filtered_sigma2 <- function(r, params) {
  n <- length(r)
  sigma2 <- numeric(n)
  sigma2[1] <- var(r)
  for (t in 2:n) {
    sigma2[t] <- params$omega + params$alpha * r[t - 1]^2 + params$beta * sigma2[t - 1]
  }
  sigma2
}

# ------------------------------------------------------------
# 3) Simulation + estimation + validation
# ------------------------------------------------------------
cat("=======================================================\n")
cat(" GARCH(1,1) -- simulation et estimation par MLE (optim)\n")
cat("=======================================================\n\n")

true_params <- list(omega = 0.05, alpha = 0.10, beta = 0.85)
cat(sprintf("Parametres vrais : omega=%.4f  alpha=%.3f  beta=%.3f  (persistance alpha+beta=%.3f)\n",
            true_params$omega, true_params$alpha, true_params$beta,
            true_params$alpha + true_params$beta))

sim <- simulate_garch11(n = 3000, omega = true_params$omega,
                          alpha = true_params$alpha, beta = true_params$beta)

fit <- estimate_garch11(sim$r)
cat(sprintf("Parametres estimes : omega=%.4f  alpha=%.3f  beta=%.3f  (persistance=%.3f)\n",
            fit$omega, fit$alpha, fit$beta, fit$alpha + fit$beta))
cat(sprintf("Convergence optim() : %s (0 = succes)\n", fit$convergence))
cat(sprintf("Ecart relatif sur alpha+beta (persistance de la volatilite) : %.2f %%\n\n",
            100 * abs((fit$alpha + fit$beta) - (true_params$alpha + true_params$beta)) /
              (true_params$alpha + true_params$beta)))

sigma2_hat <- filtered_sigma2(sim$r, fit)

# ------------------------------------------------------------
# 4) Diagnostics : clustering de volatilite (ACF des rendements au carre)
# ------------------------------------------------------------
# Signature empirique classique de la volatilite conditionnelle :
# les rendements eux-memes ne sont quasi pas autocorreles, mais leurs
# CARRES le sont fortement -- c'est precisement ce qu'un GARCH capture
# et qu'un bruit blanc simple ne capture pas.

acf_returns <- acf(sim$r, plot = FALSE, lag.max = 20)
acf_returns_sq <- acf(sim$r^2, plot = FALSE, lag.max = 20)

# ------------------------------------------------------------
# 5) Graphiques (base R)
# ------------------------------------------------------------
dir.create(outputs_dir, showWarnings = FALSE, recursive = TRUE)
png(file.path(outputs_dir, "garch_diagnostics.png"), width = 1150, height = 900, res = 120)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

plot(sim$r, type = "l", col = "steelblue", lwd = 0.7,
     xlab = "t", ylab = "Rendement simule", main = "Rendements simules (GARCH(1,1))")

plot(sqrt(sim$sigma2), type = "l", col = "black", lwd = 1.3,
     xlab = "t", ylab = "Volatilite conditionnelle",
     main = "Volatilite : vraie (noir) vs filtree/estimee (rouge)")
lines(sqrt(sigma2_hat), col = "firebrick", lwd = 1, lty = 2)
legend("topright", legend = c("Vraie (simulation)", "Estimee (MLE)"),
       col = c("black", "firebrick"), lty = c(1, 2), bty = "n", cex = 0.8)

barplot(acf_returns$acf[-1], names.arg = 1:20, col = "gray70",
        xlab = "Retard", ylab = "ACF", main = "ACF des rendements (faible)")

barplot(acf_returns_sq$acf[-1], names.arg = 1:20, col = "darkorange",
        xlab = "Retard", ylab = "ACF", main = "ACF des rendements au carre (clustering)")

dev.off()
cat(sprintf("Graphique exporte -> %s\n", file.path(outputs_dir, "garch_diagnostics.png")))
