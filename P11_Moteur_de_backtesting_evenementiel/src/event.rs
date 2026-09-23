//! event.rs
//! ========
//! Les quatre types d'événements qui circulent dans le moteur de
//! backtesting événementiel. C'est l'architecture standard (voir
//! Aronson, *Evidence-Based Technical Analysis* ; QuantStart, event-
//! driven backtesting) : chaque étape du cycle de vie d'une décision
//! de trading est un événement explicite et traçable, plutôt qu'un
//! calcul vectorisé "en bloc" qui masque l'ordre causal réel des
//! décisions (et qui, mal fait, introduit facilement du look-ahead
//! bias).
//!
//!   Market -> Strategy -> Signal -> Portfolio -> Order -> Execution -> Fill -> Portfolio
//!
//! Chaque flèche correspond à un composant qui ne voit QUE
//! l'information disponible à cet instant précis -- c'est ce qui
//! rend ce type de moteur fiable pour tester une stratégie sans se
//! mentir à soi-même.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Long,
    Short,
    Flat,
}

#[derive(Debug, Clone, Copy)]
pub struct Bar {
    pub index: usize,
    pub price: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum Event {
    Market(Bar),
    Signal { index: usize, direction: Direction },
    Order { index: usize, delta_quantity: f64 },
    Fill { index: usize, delta_quantity: f64, fill_price: f64, commission: f64 },
}
