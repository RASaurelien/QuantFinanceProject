//! strategy.rs
//! ===========
//! Le trait `Strategy` ne voit QUE l'historique des prix jusqu'au jour
//! courant inclus -- aucune fuite d'information future n'est
//! possible par construction de l'architecture événementielle (la
//! stratégie est appelée une fois par jour, dans l'ordre chronologique,
//! et ne reçoit que `&history[..=today]`).

use crate::event::Direction;

/// Retourne la direction souhaitee (Long/Short/Flat) pour le jour
/// courant, ou None si aucun signal n'est genere (pas assez
/// d'historique, ou pas de changement d'avis).
pub trait Strategy {
    fn on_bar(&mut self, history: &[f64]) -> Option<Direction>;
    fn name(&self) -> &'static str;
}

/// --- Momentum : croisement de moyennes mobiles ---
/// Signal Long quand la moyenne courte croise au-dessus de la longue
/// (tendance haussiere naissante), Short dans le cas inverse. Classique
/// et robuste, mais par nature en retard (elle suit la tendance, elle
/// ne l'anticipe pas) -- doit bien performer en regime de tendance et
/// perdre en range (faux signaux repetes = "whipsaw").
pub struct MovingAverageCrossover {
    short_window: usize,
    long_window: usize,
    current_position: Direction,
}

impl MovingAverageCrossover {
    pub fn new(short_window: usize, long_window: usize) -> Self {
        Self { short_window, long_window, current_position: Direction::Flat }
    }
}

impl Strategy for MovingAverageCrossover {
    fn on_bar(&mut self, history: &[f64]) -> Option<Direction> {
        if history.len() < self.long_window + 1 {
            return None;
        }

        let short_ma = mean(&history[history.len() - self.short_window..]);
        let long_ma = mean(&history[history.len() - self.long_window..]);

        let desired = if short_ma > long_ma { Direction::Long } else { Direction::Short };

        if desired != self.current_position {
            self.current_position = desired;
            Some(desired)
        } else {
            None
        }
    }

    fn name(&self) -> &'static str { "Momentum (MA Crossover)" }
}

/// --- Mean-reversion : z-score sur fenetre glissante ---
/// Signal Long quand le prix est anormalement BAS par rapport a sa
/// moyenne recente (z < -z_entry), Short quand anormalement haut,
/// sortie de position quand le z-score revient pres de zero. Doit
/// bien performer en regime de range et perdre en tendance (elle
/// "achete la baisse" precisement quand une vraie tendance baissiere
/// continue de baisser).
pub struct MeanReversion {
    window: usize,
    z_entry: f64,
    z_exit: f64,
    current_position: Direction,
}

impl MeanReversion {
    pub fn new(window: usize, z_entry: f64, z_exit: f64) -> Self {
        Self { window, z_entry, z_exit, current_position: Direction::Flat }
    }
}

impl Strategy for MeanReversion {
    fn on_bar(&mut self, history: &[f64]) -> Option<Direction> {
        if history.len() < self.window + 1 {
            return None;
        }

        let window_data = &history[history.len() - self.window..];
        let mu = mean(window_data);
        let sigma = std_dev(window_data, mu);
        if sigma < 1e-8 {
            return None;
        }

        let price = *history.last().unwrap();
        let z = (price - mu) / sigma;

        let desired = match self.current_position {
            Direction::Flat => {
                if z < -self.z_entry { Direction::Long }
                else if z > self.z_entry { Direction::Short }
                else { Direction::Flat }
            }
            Direction::Long | Direction::Short => {
                if z.abs() < self.z_exit { Direction::Flat } else { self.current_position }
            }
        };

        if desired != self.current_position {
            self.current_position = desired;
            Some(desired)
        } else {
            None
        }
    }

    fn name(&self) -> &'static str { "Mean-Reversion (Z-score)" }
}

fn mean(data: &[f64]) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

fn std_dev(data: &[f64], mu: f64) -> f64 {
    let var = data.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / data.len() as f64;
    var.sqrt()
}
