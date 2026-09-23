//! generator.rs
//! ============
//! Génère un flux de ticks synthétique (bid/ask/tailles), pensé comme
//! un ITÉRATEUR plutôt qu'un vecteur pré-calculé : les ticks sont
//! produits à la demande, un par un, ce qui permet au pipeline en
//! aval de tourner en mémoire O(1) même pour des dizaines de millions
//! de ticks -- jamais besoin de tout charger en RAM d'un coup, comme
//! un vrai flux de marché.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

#[derive(Debug, Clone, Copy)]
pub struct Tick {
    pub index: u64,
    pub bid: f64,
    pub ask: f64,
    pub bid_size: f64,
    pub ask_size: f64,
}

pub struct TickGenerator {
    rng: StdRng,
    mid: f64,
    index: u64,
    n_ticks: u64,
    step_dist: Normal<f64>,
}

impl TickGenerator {
    pub fn new(n_ticks: u64, start_price: f64, tick_vol: f64, seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            mid: start_price,
            index: 0,
            n_ticks,
            step_dist: Normal::new(0.0, tick_vol).unwrap(),
        }
    }
}

impl Iterator for TickGenerator {
    type Item = Tick;

    fn next(&mut self) -> Option<Tick> {
        if self.index >= self.n_ticks {
            return None;
        }

        self.mid += self.step_dist.sample(&mut self.rng);
        self.mid = self.mid.max(1.0);

        // Spread stochastique (jamais negatif), taille du carnet aleatoire
        // des deux cotes -- assez pour calculer un microprix et un
        // imbalance non triviaux.
        let half_spread = 0.01 + 0.02 * rand::Rng::gen::<f64>(&mut self.rng);
        let bid_size = 100.0 + 400.0 * rand::Rng::gen::<f64>(&mut self.rng);
        let ask_size = 100.0 + 400.0 * rand::Rng::gen::<f64>(&mut self.rng);

        let tick = Tick {
            index: self.index,
            bid: self.mid - half_spread,
            ask: self.mid + half_spread,
            bid_size,
            ask_size,
        };
        self.index += 1;
        Some(tick)
    }
}
