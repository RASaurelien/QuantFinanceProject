//! features.rs
//! ===========
//! Calcule les features tick-by-tick en UNE SEULE PASSE, avec un état
//! de taille CONSTANTE (O(1)) quel que soit le nombre de ticks déjà
//! traités -- pas de fenêtre glissante stockée en mémoire (pas de
//! VecDeque de prix), uniquement des accumulateurs incrémentaux :
//!   - Welford (1962) pour la moyenne/variance en ligne des rendements
//!     (stable numériquement, pas de somme de carrés qui explose)
//!   - moyennes mobiles exponentielles (EMA) pour le microprix lissé
//!
//! C'est la différence entre un pipeline qui peut tourner sur un flux
//! de ticks infini (production) et un qui doit tout recharger en RAM
//! à chaque fenêtre (le piège classique d'un notebook qui "marche" sur
//! 100k lignes et explose en mémoire sur 100M).

use crate::generator::Tick;

pub struct FeatureRow {
    pub index: u64,
    pub mid: f64,
    pub spread: f64,
    pub microprice: f64,
    pub imbalance: f64,       // (bid_size - ask_size) / (bid_size + ask_size), dans [-1, 1]
    pub ema_microprice: f64,
    pub realized_vol: f64,    // ecart-type en ligne des rendements du microprix (Welford)
}

pub struct StreamingFeatureEngine {
    ema_alpha: f64,
    ema_microprice: f64,
    initialized: bool,

    // --- Accumulateurs Welford (moyenne/variance en ligne) ---
    welford_count: u64,
    welford_mean: f64,
    welford_m2: f64,
    prev_microprice: f64,
}

impl StreamingFeatureEngine {
    pub fn new(ema_alpha: f64) -> Self {
        Self {
            ema_alpha,
            ema_microprice: 0.0,
            initialized: false,
            welford_count: 0,
            welford_mean: 0.0,
            welford_m2: 0.0,
            prev_microprice: 0.0,
        }
    }

    /// Met a jour l'etat interne avec UN tick et retourne la ligne de
    /// features correspondante. Cout : O(1) en temps ET en memoire,
    /// independamment du nombre de ticks deja vus.
    pub fn update(&mut self, tick: &Tick) -> FeatureRow {
        let mid = 0.5 * (tick.bid + tick.ask);
        let spread = tick.ask - tick.bid;

        // Microprix : moyenne ponderee par la taille du COTE OPPOSE --
        // plus il y a de volume au bid, plus le "vrai" prix d'equilibre
        // est tire vers l'ask (pression acheteuse), et inversement.
        let total_size = tick.bid_size + tick.ask_size;
        let microprice = (tick.bid * tick.ask_size + tick.ask * tick.bid_size) / total_size;
        let imbalance = (tick.bid_size - tick.ask_size) / total_size;

        if !self.initialized {
            self.ema_microprice = microprice;
            self.prev_microprice = microprice;
            self.initialized = true;
        } else {
            self.ema_microprice = self.ema_alpha * microprice + (1.0 - self.ema_alpha) * self.ema_microprice;

            // Rendement du microprix depuis le tick precedent -> mise a
            // jour Welford (moyenne/variance en ligne, un seul passage,
            // stable numeriquement meme apres des millions de points).
            let ret = microprice / self.prev_microprice - 1.0;
            self.welford_count += 1;
            let delta = ret - self.welford_mean;
            self.welford_mean += delta / self.welford_count as f64;
            let delta2 = ret - self.welford_mean;
            self.welford_m2 += delta * delta2;

            self.prev_microprice = microprice;
        }

        let realized_vol = if self.welford_count > 1 {
            (self.welford_m2 / (self.welford_count - 1) as f64).sqrt()
        } else {
            0.0
        };

        FeatureRow {
            index: tick.index,
            mid,
            spread,
            microprice,
            imbalance,
            ema_microprice: self.ema_microprice,
            realized_vol,
        }
    }
}
