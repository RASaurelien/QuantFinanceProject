#!/usr/bin/env Rscript
# ============================================================
# utils.R
# =======
# Fonctions partagées par les trois scripts d'économétrie, en
# particulier le test de racine unitaire de Dickey-Fuller augmenté
# (ADF), implémenté à la main plutôt qu'importé du package `urca`
# (indisponible : pas d'accès à CRAN dans cet environnement -- voir
# README). L'implémenter soi-même a aussi une vraie valeur pédagogique :
# ça oblige à comprendre exactement ce que teste le test, pas juste à
# lire une p-value produite par une boîte noire.
# ============================================================

#' Test de Dickey-Fuller augmenté (ADF), specification "constante, sans tendance"
#'
#' Regression testee :  Delta y_t = alpha + beta * y_{t-1} + sum_i gamma_i * Delta y_{t-i} + eps_t
#' H0 : beta = 0 (racine unitaire, serie NON stationnaire)
#' H1 : beta < 0 (serie stationnaire)
#'
#' Les valeurs critiques asymptotiques (grand echantillon, avec
#' constante sans tendance) sont celles de MacKinnon (1996), largement
#' reprises dans la litterature (Hamilton, 1994 ; Enders, 2014).
adf_test <- function(y, lags = 1) {
  n <- length(y)
  dy <- diff(y)
  y_lag <- y[-n]                          # y_{t-1}, aligne sur dy

  # Construction des termes augmentes (differences retardees), pour
  # blanchir l'autocorrelation residuelle avant le test.
  df <- data.frame(dy = dy[(lags + 1):length(dy)], y_lag = y_lag[(lags + 1):length(y_lag)])
  for (i in 1:lags) {
    df[[paste0("dy_lag", i)]] <- dy[(lags + 1 - i):(length(dy) - i)]
  }

  fit <- lm(dy ~ ., data = df)
  beta_hat <- coef(fit)["y_lag"]
  se_beta <- summary(fit)$coefficients["y_lag", "Std. Error"]
  t_stat <- beta_hat / se_beta

  # Valeurs critiques asymptotiques MacKinnon (1996), cas "constante, sans tendance"
  critical <- c("1%" = -3.43, "5%" = -2.86, "10%" = -2.57)

  list(
    t_stat = as.numeric(t_stat),
    critical = critical,
    stationary_5pct = as.numeric(t_stat) < critical["5%"],
    n_obs = nrow(df)
  )
}

#' Petit format d'impression coherent pour tous les scripts.
print_adf_result <- function(label, res) {
  cat(sprintf("  %-40s t-stat = %7.3f   (seuil 5%% = %.2f)   -> %s\n",
              label, res$t_stat, res$critical["5%"],
              if (res$stationary_5pct) "STATIONNAIRE (H0 rejetee)" else "NON stationnaire (H0 non rejetee)"))
}
