//! simulate.rs
//! ===========
//! Simule des trajectoires de log-prix haute fréquence sous un modèle
//! GBM (volatilité constante), observées avec un bruit de microstructure
//! additif iid (modèle standard de Zhang, Mykland & Aït-Sahalia, 2005).
//!
//! Comme sigma est constant, la variance intégrée VRAIE sur [0,T] est
//! connue exactement : IV = sigma^2 * T. C'est ce qui permet de mesurer
//! précisément le biais de chaque estimateur (naïf vs TSRV) plutôt que
//! de se contenter d'un "ça a l'air raisonnable".

use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Simule n_ticks increments (donc n_ticks+1 observations) de log-prix
/// bruité. Retourne le vecteur de log-prix OBSERVÉS (prix vrai + bruit).
pub fn simulate_noisy_gbm<R: Rng>(
    rng: &mut R,
    n_ticks: usize,
    sigma: f64,
    t_horizon: f64,
    noise_std: f64,
) -> Vec<f64> {
    let dt = t_horizon / n_ticks as f64;
    let diffusion_step = sigma * dt.sqrt();

    let step_dist = Normal::new(0.0, diffusion_step).unwrap();
    let noise_dist = Normal::new(0.0, noise_std).unwrap();

    let mut true_log_price = 0.0_f64;
    let mut observed = Vec::with_capacity(n_ticks + 1);
    // t=0 : premier prix observé, déjà bruité (cohérent avec un carnet
    // d'ordres réel où même la première cotation a un bid-ask bounce).
    observed.push(true_log_price + noise_dist.sample(rng));

    for _ in 0..n_ticks {
        // Mouvement brownien géométrique : pas de dérive (mu=0), elle
        // n'affecte de toute façon pas la variance intégrée mesurée.
        true_log_price += step_dist.sample(rng);
        observed.push(true_log_price + noise_dist.sample(rng));
    }

    observed
}

/// Variance intégrée théorique exacte pour une trajectoire GBM à
/// volatilité constante sur [0, T]. Sert de référence "vérité terrain"
/// pour évaluer le biais des estimateurs.
pub fn true_integrated_variance(sigma: f64, t_horizon: f64) -> f64 {
    sigma * sigma * t_horizon
}
