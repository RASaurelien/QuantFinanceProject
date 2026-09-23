//! backtest.rs
//! ===========
//! La boucle événementielle elle-même : une vraie file d'événements
//! (VecDeque), traitée dans l'ordre FIFO à chaque pas de temps. Ce
//! n'est pas juste une suite d'appels de fonctions déguisée -- c'est
//! le patron "event-driven" au sens propre : chaque composant ne fait
//! que réagir à un événement et, éventuellement, en produire un
//! nouveau, sans jamais avoir de vision directe sur les autres
//! composants ni sur le futur.

use std::collections::VecDeque;

use crate::data::simulate_prices;
use crate::event::{Bar, Event};
use crate::execution::ExecutionHandler;
use crate::portfolio::Portfolio;
use crate::strategy::Strategy;

pub struct BacktestConfig {
    pub n_days: usize,
    pub seed: u64,
    pub initial_cash: f64,
    pub invest_fraction: f64,
    pub slippage_bps: f64,
    pub commission_bps: f64,
}

pub fn run_backtest(strategy: &mut dyn Strategy, config: &BacktestConfig) -> (Portfolio, Vec<f64>) {
    let prices = simulate_prices(config.n_days, config.seed);
    let mut portfolio = Portfolio::new(config.initial_cash, config.invest_fraction);
    let execution = ExecutionHandler::new(config.slippage_bps, config.commission_bps);

    let mut queue: VecDeque<Event> = VecDeque::new();

    for i in 0..prices.len() {
        let price = prices[i];
        queue.push_back(Event::Market(Bar { index: i, price }));

        // On vide la file a chaque pas de temps avant de passer au
        // jour suivant : Market -> (Signal) -> (Order) -> (Fill),
        // chaque etape ne connaissant que ce que l'evenement precedent
        // lui a transmis.
        while let Some(event) = queue.pop_front() {
            match event {
                Event::Market(bar) => {
                    let history = &prices[0..=bar.index];
                    if let Some(direction) = strategy.on_bar(history) {
                        queue.push_back(Event::Signal { index: bar.index, direction });
                    }
                }
                Event::Signal { index, direction } => {
                    if let Some(order_event) = portfolio.on_signal(index, direction, price) {
                        queue.push_back(order_event);
                    }
                }
                Event::Order { index, delta_quantity } => {
                    let fill_event = execution.execute(index, delta_quantity, price);
                    queue.push_back(fill_event);
                }
                Event::Fill { index, delta_quantity, fill_price, commission } => {
                    portfolio.on_fill(delta_quantity, fill_price, commission, index);
                }
            }
        }

        portfolio.record_equity(price);
    }

    (portfolio, prices)
}
