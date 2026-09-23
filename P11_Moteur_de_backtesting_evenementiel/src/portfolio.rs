//! portfolio.rs
//! ============
//! Convertit un signal de stratégie (Long/Short/Flat, une INTENTION)
//! en ordre concret (une QUANTITÉ à acheter ou vendre), en fonction du
//! capital disponible -- puis met à jour cash/position à partir des
//! exécutions réelles (Fill). C'est le composant qui répond à la
//! question "combien ?", jamais posée par la stratégie elle-même
//! (qui ne répond qu'à "dans quel sens ?").
//!
//! Dimensionnement : fraction fixe du capital (95% de l'équité
//! courante), une convention simple mais standard pour un premier
//! moteur -- pas de Kelly, pas de vol-targeting (voir limites).

use crate::event::{Direction, Event};

pub struct Portfolio {
    pub cash: f64,
    pub position: f64,           // quantite detenue (positive=long, negative=short, 0=flat)
    pub equity_curve: Vec<f64>,   // valeur totale (cash + position au prix du jour) a chaque pas
    invest_fraction: f64,
    trade_log: Vec<(usize, f64, f64)>, // (index, delta_quantity, fill_price), pour le calcul des metriques
}

impl Portfolio {
    pub fn new(initial_cash: f64, invest_fraction: f64) -> Self {
        Self {
            cash: initial_cash,
            position: 0.0,
            equity_curve: Vec::new(),
            invest_fraction,
            trade_log: Vec::new(),
        }
    }

    pub fn current_equity(&self, price: f64) -> f64 {
        self.cash + self.position * price
    }

    /// Convertit un signal en ordre : calcule la position CIBLE (en
    /// quantite) compatible avec la direction desiree et le capital
    /// disponible, puis genere l'ordre du DELTA necessaire pour y
    /// arriver depuis la position actuelle.
    pub fn on_signal(&self, index: usize, direction: Direction, price: f64) -> Option<Event> {
        let equity = self.current_equity(price);
        let target_notional = self.invest_fraction * equity;

        let target_position = match direction {
            Direction::Long => target_notional / price,
            Direction::Short => -target_notional / price,
            Direction::Flat => 0.0,
        };

        let delta = target_position - self.position;
        if delta.abs() < 1e-6 {
            None
        } else {
            Some(Event::Order { index, delta_quantity: delta })
        }
    }

    pub fn on_fill(&mut self, delta_quantity: f64, fill_price: f64, commission: f64, index: usize) {
        self.cash -= delta_quantity * fill_price + commission;
        self.position += delta_quantity;
        self.trade_log.push((index, delta_quantity, fill_price));
    }

    pub fn record_equity(&mut self, price: f64) {
        self.equity_curve.push(self.current_equity(price));
    }

    pub fn trade_log(&self) -> &[(usize, f64, f64)] {
        &self.trade_log
    }
}
