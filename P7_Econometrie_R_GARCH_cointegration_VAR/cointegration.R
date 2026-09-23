#!/usr/bin/env Rscript
# ============================================================
# cointegration.R
# ================
# Test de cointegration d'Engle-Granger (methode en deux etapes),
# applique a une paire simulee COINTEGREE (meme tendance stochastique
# commune) ET a une paire NON cointegree (deux marches aleatoires
# independantes), comme controle negatif -- pour verifier que le test
# distingue vraiment les deux cas, pas juste "produire un chiffre".
#
# Etape 1 : regression de long terme  P1_t = c + beta*P2_t + u_t
# Etape 2 : test de racine unitaire (ADF) sur le residu u_t
#           -> si u_t est stationnaire, P1 et P2 sont cointegres
#              (leur ecart de long terme ne diverge jamais indefiniment)
# ============================================================

script_args <- commandArgs(trailingOnly = FALSE)
file_arg <- script_args[grep("^--file=", script_args)]
if (length(file_arg) > 0) {
  script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg[1])))
} else {
  script_dir <- getwd()
}
source(file.path(script_dir, "utils.R"))

set.seed(7)

n <- 1000

# ------------------------------------------------------------
# 1) Paire COINTEGREE : tendance stochastique commune + bruits
#    stationnaires idiosyncratiques (chacun I(1) individuellement,
#    mais leur combinaison lineaire est stationnaire)
# ------------------------------------------------------------
common_trend <- cumsum(rnorm(n, mean = 0, sd = 0.5))     # marche aleatoire commune
beta_true <- 1.5

price1_coint <- 50 + common_trend + rnorm(n, sd = 0.8)     # bruit idiosyncratique stationnaire
price2_coint <- 30 + common_trend / beta_true + rnorm(n, sd = 0.6)

# ------------------------------------------------------------
# 2) Paire NON cointegree : deux marches aleatoires INDEPENDANTES
#    (controle negatif -- ne doivent PAS apparaitre cointegrees)
# ------------------------------------------------------------
price1_indep <- 50 + cumsum(rnorm(n, sd = 0.5))
price2_indep <- 30 + cumsum(rnorm(n, sd = 0.5))

# ------------------------------------------------------------
# 3) Procedure d'Engle-Granger, appliquee aux deux paires
# ------------------------------------------------------------
engle_granger <- function(p1, p2, label) {
  reg <- lm(p1 ~ p2)
  spread <- residuals(reg)
  adf_res <- adf_test(spread, lags = 2)

  cat(sprintf("\n--- %s ---\n", label))
  cat(sprintf("  Regression long terme : p1 = %.3f + %.3f * p2\n", coef(reg)[1], coef(reg)[2]))
  print_adf_result("ADF sur le residu (spread)", adf_res)

  list(spread = spread, adf = adf_res, beta_hat = coef(reg)[2])
}

cat("=======================================================\n")
cat(" Cointegration -- methode d'Engle-Granger (2 etapes)\n")
cat("=======================================================\n")

result_coint <- engle_granger(price1_coint, price2_coint,
                               sprintf("Paire COINTEGREE (beta vrai = %.2f)", beta_true))
result_indep <- engle_granger(price1_indep, price2_indep,
                               "Paire NON cointegree (controle negatif)")

cat(sprintf("\nResume : paire cointegree -> %s   |   paire independante -> %s\n",
            if (result_coint$adf$stationary_5pct) "cointegration detectee (correct)" else "PAS detectee (inattendu)",
            if (!result_indep$adf$stationary_5pct) "pas de cointegration (correct)" else "cointegration detectee A TORT"))

# ------------------------------------------------------------
# 4) Signal de pairs trading : z-score du spread cointegre
# ------------------------------------------------------------
z_score <- (result_coint$spread - mean(result_coint$spread)) / sd(result_coint$spread)

# ------------------------------------------------------------
# 5) Graphiques (base R)
# ------------------------------------------------------------
outputs_dir <- normalizePath(file.path(script_dir, "outputs"), winslash = "/", mustWork = FALSE)
dir.create(outputs_dir, showWarnings = FALSE, recursive = TRUE)
png(file.path(outputs_dir, "cointegration.png"), width = 1150, height = 900, res = 120)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

plot(price1_coint, type = "l", col = "navy", ylim = range(c(price1_coint, price2_coint * beta_true)),
     xlab = "t", ylab = "Prix", main = "Paire cointegree : prix (co-mouvement visible)")
lines(price2_coint * beta_true, col = "firebrick")
legend("topleft", legend = c("Actif 1", "Actif 2 (mis a l'echelle)"), col = c("navy", "firebrick"),
       lty = 1, bty = "n", cex = 0.8)

plot(result_coint$spread, type = "l", col = "darkgreen",
     xlab = "t", ylab = "Spread (residu)", main = "Spread cointegre (stationnaire)")
abline(h = 0, lty = 2)

plot(price1_indep, type = "l", col = "navy", ylim = range(c(price1_indep, price2_indep)),
     xlab = "t", ylab = "Prix", main = "Paire NON cointegree (controle negatif)")
lines(price2_indep, col = "firebrick")
legend("topleft", legend = c("Actif 1", "Actif 2"), col = c("navy", "firebrick"), lty = 1, bty = "n", cex = 0.8)

plot(z_score, type = "l", col = "black", xlab = "t", ylab = "Z-score du spread",
     main = "Signal pairs trading (paire cointegree)")
abline(h = c(-2, 0, 2), col = c("firebrick", "gray50", "firebrick"), lty = 2)
legend("topleft", legend = c("Seuil entree/sortie (+/-2 sigma)"), col = "firebrick", lty = 2, bty = "n", cex = 0.8)

dev.off()
cat(sprintf("\nGraphique exporte -> %s\n", file.path(outputs_dir, "cointegration.png")))
