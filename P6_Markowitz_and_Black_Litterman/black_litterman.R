#!/usr/bin/env Rscript
# ============================================================
# black_litterman.R
# ==================
# Calibre le modèle de Black-Litterman en partant des estimations
# Markowitz calculées côté Python (mu.csv, cov.csv, market_weights.csv)
# et produit :
#   - les rendements d'équilibre implicites (reverse optimization)
#   - les rendements postérieurs, mise à jour bayésienne à partir de
#     deux vues d'investisseur explicites
#   - le portefeuille optimal (tangence) sous les rendements postérieurs
#   - un graphique en base R comparant rendements a priori / a posteriori
#
# Ce script utilise UNIQUEMENT les fonctions de base R (algèbre
# matricielle, graphiques) : aucune dépendance externe (quadprog,
# ggplot2, ...) n'est nécessaire, ce qui le rend portable sans accès
# réseau à CRAN.
# ============================================================

args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) >= 1) args[1] else "../data"
out_dir  <- if (length(args) >= 2) args[2] else "../outputs"

mu_df     <- read.csv(file.path(data_dir, "mu.csv"), row.names = 1)
cov_df    <- read.csv(file.path(data_dir, "cov.csv"), row.names = 1)
weights_df <- read.csv(file.path(data_dir, "market_weights.csv"), row.names = 1)

assets <- rownames(mu_df)
# rendements espérés annualisés (Markowitz, a priori "naïf")
mu     <- setNames(mu_df[, 1], assets)
# matrice de covariance annualisée
Sigma  <- as.matrix(cov_df[assets, assets])
# poids de marché (équilibre)
w_mkt  <- setNames(weights_df[assets, 1], assets)

rf <- 0.02   # taux sans risque annualisé

cat("=======================================================\n")
cat(" Black-Litterman? mise a jour bayesienne des rendements\n")
cat("=======================================================\n\n")
cat("Actifs :", paste(assets, collapse = ", "), "\n\n")

# ------------------------------------------------------------
# Etape 1 : rendements d'equilibre implicites (reverse optimization)
# ------------------------------------------------------------
# Plutot que de partir de rendements estimes historiquement (bruites,
# peu fiables? le probleme classique de Markowitz "garbage in,
# garbage out"), Black-Litterman part des rendements que le MARCHE
# implique DEJA via ses poids observes (capitalisation), en inversant
# la formule d'optimisation moyenne-variance :
# Pi = delta * Sigma %*% w_mkt
# ou delta est le coefficient d'aversion au risque implicite du marche.
ret_mkt_excess <- as.numeric(t(w_mkt) %*% (mu - rf))
var_mkt <- as.numeric(t(w_mkt) %*% Sigma %*% w_mkt)
delta <- ret_mkt_excess / var_mkt

# rendements d'equilibre, EN EXCES du taux sans risque
Pi_excess <- delta * (Sigma %*% w_mkt)
Pi_total  <- as.numeric(Pi_excess) + rf

cat(sprintf("Coef aversion au risque implicite (delta) : %.3f\n\n", delta))
cat("Rendements d'equilibre implicites (Pi) :\n")
for (i in seq_along(assets)) {
  cat(sprintf("  %-25s %6.2f%%\n", assets[i], Pi_total[i] * 100))
}
cat("\n")

# ------------------------------------------------------------
# Etape 2 : vues de l'investisseur (P, Q) et incertitude (Omega)
# ------------------------------------------------------------
# Deux vues concretes, formulees comme un investisseur macro le ferait :
#   Vue 1 (relative) : les actions emergentes (EEM) surperformeront les
#                        actions US (SPY) de 3 points par an.
#   Vue 2 (absolue)   : l'or (GLD) delivrera un rendement de 7% par an.
P <- matrix(0, nrow = 2, ncol = length(assets), dimnames = list(c("vue1", "vue2"), assets))
P["vue1", "EEM"] <-  1
P["vue1", "SPY"] <- -1
P["vue2", "GLD"] <-  1
# vue1 : +3% de surperformance EEM vs SPY | vue2 : GLD a 7% absolu
Q <- c(0.03, 0.07)

# facteur d'echelle standard (incertitude sur Pi lui-meme, He & Litterman 1999)
tau <- 0.05

# Incertitude des vues (Omega) : proportionnelle a la variance a priori
# du portefeuille de vue lui-meme, ponderee par tau -- formule standard
# (He & Litterman, 1999) qui evite d'avoir a fixer une "confiance"
# subjective par vue.
Omega <- diag(diag(tau * P %*% Sigma %*% t(P)), nrow = nrow(P))

cat("Vues de l'investisseur :\n")
cat(sprintf("  Vue 1 : EEM - SPY = %+.1f%% (surperformance relative)\n", Q[1] * 100))
cat(sprintf("  Vue 2 : GLD = %.1f%% (rendement absolu)\n", Q[2] * 100))
cat(sprintf("  Incertitude (Omega, diagonale) : %s\n\n", paste(round(diag(Omega), 5), collapse = ", ")))

# ------------------------------------------------------------
# Etape 3 : mise a jour bayesienne (coeur de Black-Litterman)
# ------------------------------------------------------------
# Le prior est Gaussien : rendements ~ N(Pi, tau*Sigma)
# La vraisemblance des vues est Gaussienne : P*mu ~ N(Q, Omega)
# Le posterieur (Gaussien-Gaussien, conjugue) a une forme fermee :
#     M          = [ (tau*Sigma)^-1 + P' Omega^-1 P ]^-1
#     mu_post    = M %*% [ (tau*Sigma)^-1 %*% Pi + P' Omega^-1 %*% Q ]
#     Sigma_post = Sigma + M
tauSigma_inv <- solve(tau * Sigma)
Omega_inv <- solve(Omega)

M <- solve(tauSigma_inv + t(P) %*% Omega_inv %*% P)
mu_post_excess <- M %*% (tauSigma_inv %*% Pi_excess + t(P) %*% Omega_inv %*% Q)
Sigma_post <- Sigma + M

mu_post_total <- as.numeric(mu_post_excess) + rf

cat("Rendements posterieurs (apres mise a jour bayesienne) :\n")
for (i in seq_along(assets)) {
  delta_bps <- (mu_post_total[i] - Pi_total[i]) * 1e4
  cat(sprintf("  %-25s %6.2f%%   (deplacement : %+.0f bps)\n", assets[i], mu_post_total[i] * 100, delta_bps))
}
cat("\n")

# ------------------------------------------------------------
# Etape 4 : portefeuille de tangence sous les rendements posterieurs
# ------------------------------------------------------------
# Meme formule fermee que cote Python (theoreme des deux fonds), mais
# appliquee a Sigma_post et mu_post -- pas de solveur QP necessaire.
Sigma_post_inv <- solve(Sigma_post)
raw_weights <- Sigma_post_inv %*% mu_post_excess
w_bl <- raw_weights / sum(raw_weights)
w_bl <- setNames(as.numeric(w_bl), assets)

ret_bl <- as.numeric(t(w_bl) %*% (mu_post_total))
vol_bl <- as.numeric(sqrt(t(w_bl) %*% Sigma_post %*% w_bl))
sharpe_bl <- (ret_bl - rf) / vol_bl

cat("Portefeuille optimal Black-Litterman (tangence) :\n")
for (i in seq_along(assets)) {
  cat(sprintf("  %-25s %6.2f%%\n", assets[i], w_bl[i] * 100))
}
cat(sprintf("\n  Rendement attendu : %.2f%%  |  Volatilite : %.2f%%  |  Sharpe : %.3f\n\n",
            ret_bl * 100, vol_bl * 100, sharpe_bl))

# ------------------------------------------------------------
# Export des resultats pour le graphique final cote Python
# ------------------------------------------------------------
dir.create(data_dir, showWarnings = FALSE, recursive = TRUE)
results <- data.frame(
  asset = assets,
  mu_prior = Pi_total,
  mu_posterior = mu_post_total,
  w_market = as.numeric(w_mkt),
  w_bl = w_bl
)
write.csv(results, file.path(data_dir, "bl_results.csv"), row.names = FALSE)
cat(sprintf("Resultats exportes -> %s\n", file.path(data_dir, "bl_results.csv")))

# ------------------------------------------------------------
# Graphique (base R) : rendements a priori vs a posteriori par actif
# ------------------------------------------------------------
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
png(file.path(out_dir, "bl_returns_comparison.png"), width = 1000, height = 650, res = 120)

bar_data <- rbind(Pi_total * 100, mu_post_total * 100)
rownames(bar_data) <- c("Equilibre (a priori)", "Black-Litterman (a posteriori)")
colnames(bar_data) <- assets

bp <- barplot(bar_data, beside = TRUE, col = c("gray70", "navy"),
              ylim = c(0, max(bar_data) * 1.3),
              ylab = "Rendement attendu annualise (%)",
              main = "Rendements d'equilibre vs Black-Litterman (vues investisseur)",
              las = 2, cex.names = 0.85)
legend("topright", legend = rownames(bar_data), fill = c("gray70", "navy"), bty = "n")
box()

dev.off()
cat(sprintf("Graphique exporte -> %s\n", file.path(out_dir, "bl_returns_comparison.png")))
