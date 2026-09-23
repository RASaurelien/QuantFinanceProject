#!/usr/bin/env Rscript
# ============================================================
# var_model.R
# ===========
# Modele VAR(1) bivarie -- estimation par OLS equation par equation
# (forme matricielle explicite, pas le package `vars`, indisponible
# sans acces a CRAN -- voir README) et fonctions de reponse
# impulsionnelle (IRF) calculees a la main par recursion sur la
# matrice compagnon.
#
#   x_t = A %*% x_{t-1} + eps_t,   eps_t ~ N(0, Sigma)
#
# Interpretation economique du systeme simule : x1 = croissance du
# PIB (desaisonnalisee), x2 = taux d'interet court terme. La matrice
# A encode une dynamique plausible : un choc de croissance pousse le
# taux a la hausse avec un delai (reaction de politique monetaire),
# et un choc de taux freine la croissance avec un delai (canal du
# credit) -- un cas d'usage VAR macro-financier classique.
# ============================================================

script_args <- commandArgs(trailingOnly = FALSE)
file_arg <- script_args[grep("^--file=", script_args)]
if (length(file_arg) > 0) {
  script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg[1])))
} else {
  script_dir <- getwd()
}
outputs_dir <- normalizePath(file.path(script_dir, "outputs"), winslash = "/", mustWork = FALSE)

set.seed(123)

# ------------------------------------------------------------
# 1) Simulation d'un VAR(1) bivarie a partir d'une matrice A CONNUE
# ------------------------------------------------------------
A_true <- matrix(c(0.5, 0.3,      # ligne 1 : x1_t = 0.5*x1_{t-1} + 0.3*x2_{t-1} + eps1
                     -0.2, 0.6),   # ligne 2 : x2_t = -0.2*x1_{t-1} + 0.6*x2_{t-1} + eps2
                   nrow = 2, byrow = TRUE)
Sigma_true <- matrix(c(1.0, 0.3,
                         0.3, 0.5), nrow = 2)

simulate_var1 <- function(n, A, Sigma, burn_in = 200) {
  n_total <- n + burn_in
  L <- chol(Sigma)   # pour generer des residus correles: eps = z %*% L, z ~ N(0,I)
  X <- matrix(0, n_total, 2)
  for (t in 2:n_total) {
    eps <- rnorm(2) %*% L
    X[t, ] <- A %*% X[t - 1, ] + as.numeric(eps)
  }
  X[(burn_in + 1):n_total, ]
}

X <- simulate_var1(600, A_true, Sigma_true)
colnames(X) <- c("croissance_pib", "taux_interet")

# ------------------------------------------------------------
# 2) Estimation OLS du VAR(1), forme matricielle explicite
# ------------------------------------------------------------
# Pour un VAR(1) : chaque equation x_i,t = a_i' x_{t-1} + eps_i,t est
# une regression lineaire standard. Forme compacte matricielle :
#     A_hat = (X_lag' X_lag)^-1 X_lag' X_now
n_obs <- nrow(X)
X_now <- X[2:n_obs, ]
X_lag <- X[1:(n_obs - 1), ]

A_hat <- t(solve(t(X_lag) %*% X_lag, t(X_lag) %*% X_now))   # (2x2), estimateur OLS multivarie
residuals_var <- X_now - X_lag %*% t(A_hat)
Sigma_hat <- cov(residuals_var)

cat("=======================================================\n")
cat(" VAR(1) bivarie -- estimation OLS et reponses impulsionnelles\n")
cat("=======================================================\n\n")
cat("Matrice A vraie :\n"); print(round(A_true, 3))
cat("\nMatrice A estimee :\n"); print(round(A_hat, 3))
cat(sprintf("\nErreur quadratique moyenne d'estimation sur A : %.5f\n\n",
            mean((A_hat - A_true)^2)))

cat("Matrice de covariance des residus (Sigma) estimee :\n"); print(round(Sigma_hat, 3))
cat("\n")

# ------------------------------------------------------------
# 3) Fonctions de reponse impulsionnelle (IRF), orthogonalisees
#    par decomposition de Cholesky (identification recursive standard)
# ------------------------------------------------------------
# Reponse a l'horizon h a un choc structurel unitaire a t=0 :
#     IRF(h) = A^h %*% P
# ou P est le facteur de Cholesky de Sigma (P P' = Sigma), qui
# "orthogonalise" les chocs (un choc structurel a la fois, dans un
# ordre causal suppose -- ici : la croissance reagit avant le taux).
compute_irf <- function(A, Sigma, horizon = 20) {
  P <- t(chol(Sigma))   # Cholesky : Sigma = P %*% t(P), P triangulaire inferieure
  n_vars <- nrow(A)
  irf <- array(0, dim = c(horizon + 1, n_vars, n_vars))   # [horizon, variable_reponse, variable_choc]
  A_power <- diag(n_vars)
  for (h in 0:horizon) {
    irf[h + 1, , ] <- A_power %*% P
    A_power <- A_power %*% A
  }
  irf
}

irf_true <- compute_irf(A_true, Sigma_true)
irf_hat <- compute_irf(A_hat, Sigma_hat)

# ------------------------------------------------------------
# 4) Graphiques (base R) : IRF vraies vs estimees, 4 combinaisons
#    (reponse de chaque variable a un choc sur chaque variable)
# ------------------------------------------------------------
dir.create(outputs_dir, showWarnings = FALSE, recursive = TRUE)
png(file.path(outputs_dir, "var_irf.png"), width = 1150, height = 900, res = 120)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

var_names <- c("Croissance PIB", "Taux d'interet")
titles <- list(
  c(1, 1, "Choc croissance -> Croissance"),
  c(1, 2, "Choc croissance -> Taux"),
  c(2, 1, "Choc taux -> Croissance"),
  c(2, 2, "Choc taux -> Taux")
)

for (t in titles) {
  resp_var <- as.integer(t[1]); shock_var <- as.integer(t[2]); title <- t[3]
  y_true <- irf_true[, resp_var, shock_var]
  y_hat <- irf_hat[, resp_var, shock_var]
  plot(0:20, y_true, type = "l", col = "black", lwd = 2,
       xlab = "Horizon (periodes)", ylab = "Reponse", main = title,
       ylim = range(c(y_true, y_hat)))
  lines(0:20, y_hat, col = "firebrick", lwd = 1.5, lty = 2)
  abline(h = 0, col = "gray70", lty = 3)
  legend("topright", legend = c("IRF vraie", "IRF estimee"), col = c("black", "firebrick"),
         lty = c(1, 2), bty = "n", cex = 0.75)
}

dev.off()
cat(sprintf("Graphique exporte -> %s\n", file.path(outputs_dir, "var_irf.png")))
