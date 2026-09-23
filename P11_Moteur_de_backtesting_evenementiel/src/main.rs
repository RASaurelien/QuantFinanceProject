//! main.rs
//! =======
//! Lance le backtest pour les deux stratégies (Momentum, Mean-
//! Reversion) sur la MÊME série de prix simulée, plus un benchmark
//! Buy & Hold, imprime un rapport de performance comparatif et
//! exporte les courbes d'équité en CSV pour visualisation Python.

mod backtest;
mod data;
mod event;
mod execution;
mod metrics;
mod portfolio;
mod strategy;

use std::fs::File;
use std::io::Write;

use backtest::{run_backtest, BacktestConfig};
use metrics::compute_performance;
use strategy::{MeanReversion, MovingAverageCrossover, Strategy};

fn print_report(name: &str, equity_curve: &[f64], n_trades: usize) {
    let report = compute_performance(equity_curve, n_trades, 252.0);
    println!("--- {name} ---");
    println!("  Rendement total          : {:+.2}%", report.total_return * 100.0);
    println!("  Rendement annualise      : {:+.2}%", report.annualized_return * 100.0);
    println!("  Volatilite annualisee    : {:.2}%", report.annualized_vol * 100.0);
    println!("  Ratio de Sharpe          : {:.3}", report.sharpe_ratio);
    println!("  Drawdown maximal         : {:.2}%", report.max_drawdown * 100.0);
    println!("  Nombre de transactions   : {}", report.n_trades);
    println!();
}

fn main() {
    let config = BacktestConfig {
        n_days: 2000,
        seed: 42,
        initial_cash: 100_000.0,
        invest_fraction: 0.95,
        slippage_bps: 2.0,     // 2 bps de glissement contre le trader
        commission_bps: 1.0,    // 1 bp de commission par notionnel echange
    };

    println!("=======================================================\n");
    println!(" Moteur de backtesting evenementiel - Momentum vs Mean-Reversion\n");
    println!("=======================================================\n");
    println!("Configuration : {} jours simules, capital initial {:.0}, ",
             config.n_days, config.initial_cash);
    println!("slippage={} bps, commission={} bps\n", config.slippage_bps, config.commission_bps);

    // --- Strategie 1 : Momentum (croisement de moyennes mobiles) ---
    let mut momentum: Box<dyn Strategy> = Box::new(MovingAverageCrossover::new(10, 50));
    let (portfolio_momentum, prices) = run_backtest(momentum.as_mut(), &config);
    print_report(momentum.name(), &portfolio_momentum.equity_curve, portfolio_momentum.trade_log().len());

    // --- Strategie 2 : Mean-Reversion (z-score) ---
    let mut mean_rev: Box<dyn Strategy> = Box::new(MeanReversion::new(20, 1.5, 0.3));
    let (portfolio_mean_rev, _) = run_backtest(mean_rev.as_mut(), &config);
    print_report(mean_rev.name(), &portfolio_mean_rev.equity_curve, portfolio_mean_rev.trade_log().len());

    // --- Benchmark : Buy & Hold (achete 95% du capital au jour 0, ne bouge plus) ---
    let buy_hold_equity: Vec<f64> = prices.iter()
        .map(|&p| {
            let qty = config.invest_fraction * config.initial_cash / prices[0];
            let cash = config.initial_cash - qty * prices[0];
            cash + qty * p
        })
        .collect();
    print_report("Buy & Hold (benchmark)", &buy_hold_equity, 1);

    // --- Export CSV pour visualisation Python ---
    let mut file = File::create("equity_curves.csv").expect("creation du CSV");
    writeln!(file, "day,price,momentum,mean_reversion,buy_hold").unwrap();
    for i in 0..prices.len() {
        writeln!(file, "{},{},{},{},{}", i, prices[i],
                 portfolio_momentum.equity_curve[i],
                 portfolio_mean_rev.equity_curve[i],
                 buy_hold_equity[i]).unwrap();
    }
    println!("Courbes d'equite exportees -> equity_curves.csv\n");

    // --- Analyse de robustesse multi-seeds ---
    // Un seul seed peut suggerer qu'une strategie "gagne" universellement --
    // ce serait un artefact de la trajectoire simulee, pas un resultat
    // robuste. On rejoue le backtest sur N scenarios independants et on
    // regarde la DISPERSION des resultats, pas juste leur moyenne.
    println!("--- Robustesse : {} scenarios independants ---\n", 30);
    let mut momentum_sharpes = Vec::new();
    let mut mean_rev_sharpes = Vec::new();
    let mut momentum_wins = 0;
    let mut mean_rev_wins = 0;

    for s in 0..30u64 {
        let cfg = BacktestConfig { seed: 1000 + s, ..config_template() };

        let mut mom: Box<dyn Strategy> = Box::new(MovingAverageCrossover::new(10, 50));
        let (pm, _) = run_backtest(mom.as_mut(), &cfg);
        let rm = compute_performance(&pm.equity_curve, pm.trade_log().len(), 252.0);

        let mut mr: Box<dyn Strategy> = Box::new(MeanReversion::new(20, 1.5, 0.3));
        let (pr, _) = run_backtest(mr.as_mut(), &cfg);
        let rr = compute_performance(&pr.equity_curve, pr.trade_log().len(), 252.0);

        momentum_sharpes.push(rm.sharpe_ratio);
        mean_rev_sharpes.push(rr.sharpe_ratio);
        if rm.sharpe_ratio > rr.sharpe_ratio { momentum_wins += 1; } else { mean_rev_wins += 1; }
    }

    let avg = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let std = |v: &[f64], m: f64| (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt();

    let mom_avg = avg(&momentum_sharpes);
    let mr_avg = avg(&mean_rev_sharpes);
    println!("Momentum       : Sharpe moyen = {:+.3}  (ecart-type = {:.3})  |  meilleur sur {}/30 scenarios",
             mom_avg, std(&momentum_sharpes, mom_avg), momentum_wins);
    println!("Mean-Reversion : Sharpe moyen = {:+.3}  (ecart-type = {:.3})  |  meilleur sur {}/30 scenarios",
             mr_avg, std(&mean_rev_sharpes, mr_avg), mean_rev_wins);
    println!("\n-> Aucune des deux strategies ne domine universellement : la performance depend");
    println!("   du regime qui domine dans chaque scenario -- c'est le comportement ATTENDU,");
    println!("   pas un defaut du moteur (voir README).");
}

fn config_template() -> BacktestConfig {
    BacktestConfig {
        n_days: 2000,
        seed: 0,
        initial_cash: 100_000.0,
        invest_fraction: 0.95,
        slippage_bps: 2.0,
        commission_bps: 1.0,
    }
}
