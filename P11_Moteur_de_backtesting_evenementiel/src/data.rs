//! data.rs
//! =======
//! Génère une série de prix synthétique à changement de régime
//! markovien entre trois états : tendance haussière, tendance
//! baissière, range (retour à la moyenne). L'objectif est un banc
//! d'essai honnête : ni le momentum ni le mean-reversion ne doivent
//! dominer partout -- chaque stratégie doit gagner dans SON régime et
//! perdre (ou faire du surplace) dans l'autre, comme sur un vrai
//! marché ou aucune des deux familles de stratégies ne fonctionne
//! tout le temps.

use rand::RngExt;
use rand_distr::{Distribution, Normal};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Regime {
    TrendUp,
    TrendDown,
    Range,
}

pub fn simulate_prices(n_days: usize, seed: u64) -> Vec<f64> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut prices = Vec::with_capacity(n_days);
    let mut price = 100.0_f64;
    let mut anchor = price; // niveau d'ancrage pour le regime "range" (mis a jour a chaque bascule)
    let mut regime = Regime::TrendUp;
    let mut days_in_regime = 0usize;

    let normal = Normal::new(0.0, 1.0).unwrap();

    for _ in 0..n_days {
        // Bascule de regime : persistante (dure en moyenne quelques
        // dizaines de jours), tirage independant a chaque pas une fois
        // une duree minimale ecoulee, pour eviter des regimes de 1 jour.
        days_in_regime += 1;
        let switch_prob = if days_in_regime < 20 { 0.0 } else { 0.03 };
        if rng.random::<f64>() < switch_prob {
            regime = match rng.random_range(0..=2) {
                0 => Regime::TrendUp,
                1 => Regime::TrendDown,
                _ => Regime::Range,
            };
            anchor = price;
            days_in_regime = 0;
        }

        let z = normal.sample(&mut rng);
        let daily_return = match regime {
            Regime::TrendUp => 0.0009 + 0.011 * z,     // ~23%/an de derive, vol ~17%/an
            Regime::TrendDown => -0.0009 + 0.011 * z,
            Regime::Range => {
                // Processus Ornstein-Uhlenbeck discretise autour de `anchor` :
                // retour a la moyenne explicite, pas juste un bruit sans direction.
                let kappa = 0.05;
                (kappa * (anchor.ln() - price.ln())) + 0.009 * z
            }
        };

        price *= (1.0 + daily_return).max(0.01); // garde-fou : prix strictement positif
        prices.push(price);
    }

    prices
}
