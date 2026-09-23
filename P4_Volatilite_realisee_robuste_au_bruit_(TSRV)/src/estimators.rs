//! estimators.rs
//! =============
//! Deux estimateurs de la variance intégrée à partir de log-prix
//! observés (bruités) :
//!
//! - `realized_variance` : l'estimateur "naïf" (somme des rendements
//!   au carré, à la fréquence la plus fine disponible). C'est
//!   l'estimateur classique en l'absence de bruit... mais en présence
//!   de bruit de microstructure, il diverge : plus on échantillonne
//!   finement, plus le bruit domine, et RV_naive -> IV_vraie + 2*n*E[epsilon^2].
//!   C'est le phénomène du "signature plot" bien connu en finance
//!   haute fréquence (voir main.rs, experiment 1).
//!
//! - `tsrv` : l'estimateur Two-Scale Realized Volatility de Zhang,
//!   Mykland & Aït-Sahalia (2005), qui combine une échelle lente
//!   (sous-échantillonnage, peu sensible au bruit mais peu de données)
//!   et une échelle rapide (toutes les données, biaisée par le bruit)
//!   pour annuler le biais tout en restant efficace. Convergence en
//!   n^(-1/6) au lieu de diverger.

/// Variance réalisée "tout tick" : somme des carrés des rendements
/// log successifs, sur la grille la plus fine disponible.
pub fn realized_variance(log_prices: &[f64]) -> f64 {
    log_prices
        .windows(2)
        .map(|w| {
            let r = w[1] - w[0];
            r * r
        })
        .sum()
}

/// Moyenne des variances réalisées calculées sur K sous-grilles non-
/// chevauchantes (grille "lente"). C'est le coeur de la technique
/// "two-scale" : sous-échantillonner réduit le nombre de retours mais
/// aussi le poids relatif du bruit dans chaque retour individuel.
fn subsampled_rv_average(log_prices: &[f64], k: usize) -> f64 {
    let n_obs = log_prices.len();
    let mut total = 0.0_f64;
    let mut n_grids = 0usize;

    for offset in 0..k {
        let sub: Vec<f64> = log_prices
            .iter()
            .copied()
            .skip(offset)
            .step_by(k)
            .collect();
        if sub.len() < 2 {
            continue;
        }
        total += realized_variance(&sub);
        n_grids += 1;
    }

    debug_assert!(n_grids > 0, "K trop grand : aucune sous-grille exploitable");
    let _ = n_obs;
    total / n_grids as f64
}

/// Estimateur Two-Scale Realized Volatility (Zhang, Mykland & Aït-
/// Sahalia, 2005), avec la correction petit-échantillon standard.
///
/// TSRV = (1 - nbar/n)^-1 * [ RV_avg,K - (nbar/n) * RV_all ]
///
/// où RV_avg,K est la moyenne des variances réalisées sur K sous-
/// grilles ("échelle lente"), RV_all est la variance réalisée sur
/// toute la grille ("échelle rapide", biaisée par le bruit), n est le
/// nombre de rendements sur la grille complète, et nbar la taille
/// moyenne d'une sous-grille.
///
/// Choix de K : la théorie recommande K ~ c * n^(2/3) pour un taux de
/// convergence optimal en n^(-1/6) (voir `optimal_k` ci-dessous).
pub fn tsrv(log_prices: &[f64], k: usize) -> f64 {
    let n_obs = log_prices.len();
    let n = (n_obs - 1) as f64; // nombre de rendements, grille complète

    let rv_avg_k = subsampled_rv_average(log_prices, k);
    let rv_all = realized_variance(log_prices);

    let nbar = (n_obs as f64 - k as f64 + 1.0) / k as f64;
    let adjustment = 1.0 - nbar / n;

    (rv_avg_k - (nbar / n) * rv_all) / adjustment
}

/// Taille de sous-échantillonnage K asymptotiquement optimale d'après
/// Zhang, Mykland & Aït-Sahalia (2005) : K* ~ c * n^(2/3). La constante
/// c dépend du ratio signal/bruit réel (inconnu en pratique) ; on
/// retient c=1 par simplicité, un choix standard dans la littérature
/// pédagogique sur le sujet.
pub fn optimal_k(n_ticks: usize) -> usize {
    let n = n_ticks as f64;
    (n.powf(2.0 / 3.0)).round().max(2.0) as usize
}
