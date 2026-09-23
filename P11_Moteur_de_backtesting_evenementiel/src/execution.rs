//! execution.rs
//! ============
//! Simule l'exécution d'un ordre : dans la réalité, un ordre ne se
//! remplit jamais exactement au dernier prix coté (le carnet d'ordres
//! bouge, l'ordre a un impact) -- ce module modélise ça avec deux
//! frictions simples mais essentielles :
//!   - slippage : le prix d'exécution est décalé DÉFAVORABLEMENT par
//!     rapport au sens de l'ordre (acheter coûte un peu plus cher que
//!     le prix coté, vendre rapporte un peu moins).
//!   - commission : coût proportionnel au notionnel échangé.
//!
//! Ignorer ces frictions est l'erreur la plus commune (et la plus
//! dangereuse) en backtesting : une stratégie qui fait beaucoup
//! d'allers-retours peut sembler rentable sans coûts de transaction,
//! et perdante avec -- ce module force à en tenir compte dès le départ.

use crate::event::Event;

pub struct ExecutionHandler {
    slippage_bps: f64,
    commission_bps: f64,
}

impl ExecutionHandler {
    pub fn new(slippage_bps: f64, commission_bps: f64) -> Self {
        Self { slippage_bps, commission_bps }
    }
    
    // Le slippage joue TOUJOURS contre le trader : acheter (delta>0)
    // coute plus cher, vendre (delta<0) rapporte moins.
    pub fn execute(&self, index: usize, delta_quantity: f64, market_price: f64) -> Event {

        let slippage_factor = 1.0 + self.slippage_bps * 1e-4 * delta_quantity.signum();
        let fill_price = market_price * slippage_factor;

        let notional = delta_quantity.abs() * fill_price;
        let commission = notional * self.commission_bps * 1e-4;

        Event::Fill { index, delta_quantity, fill_price, commission }
    }
}
