//! metrics.rs
//! ==========
//! Métriques de performance standard calculées à partir de la courbe
//! d'équité et du journal des transactions.

pub struct PerformanceReport {
    pub total_return: f64,
    pub annualized_return: f64,
    pub annualized_vol: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub n_trades: usize,
}

pub fn compute_performance(equity_curve: &[f64], n_trades: usize, trading_days_per_year: f64) -> PerformanceReport {
    let n = equity_curve.len();
    let total_return = equity_curve[n - 1] / equity_curve[0] - 1.0;

    let daily_returns: Vec<f64> = equity_curve.windows(2)
        .map(|w| w[1] / w[0] - 1.0)
        .collect();

    let mean_daily = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;
    let var_daily = daily_returns.iter().map(|r| (r - mean_daily).powi(2)).sum::<f64>()
        / daily_returns.len() as f64;
    let std_daily = var_daily.sqrt();

    let annualized_return = (1.0 + mean_daily).powf(trading_days_per_year) - 1.0;
    let annualized_vol = std_daily * trading_days_per_year.sqrt();
    let sharpe_ratio = if annualized_vol > 1e-10 { annualized_return / annualized_vol } else { 0.0 };

    // Drawdown maximal : plus grande chute depuis un plus-haut glissant
    let mut peak = equity_curve[0];
    let mut max_drawdown = 0.0;
    for &e in equity_curve {
        if e > peak { peak = e; }
        let drawdown = (peak - e) / peak;
        if drawdown > max_drawdown { max_drawdown = drawdown; }
    }

    PerformanceReport {
        total_return,
        annualized_return,
        annualized_vol,
        sharpe_ratio,
        max_drawdown,
        n_trades,
    }
}
