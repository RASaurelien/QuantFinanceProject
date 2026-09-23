//! main.rs
//! =======
//! Deux expériences, chacune exportée en CSV pour être tracée côté
//! Python :
//!
//! 1. "Signature plot" : à bruit fixé, on fait varier la fréquence
//!    d'échantillonnage (nombre de ticks) et on regarde comment la RV
//!    naïve et le TSRV réagissent. Diagnostic classique en finance
//!    haute fréquence : la RV naïve doit "exploser" quand on
//!    échantillonne trop finement (le bruit domine), alors que le TSRV
//!    doit rester proche de la vraie variance intégrée.
//!
//! 2. Table de biais/RMSE : à fréquence fixée (réaliste, proche d'un
//!    échantillonnage 1-seconde sur une séance de 6h30), on fait
//!    varier l'intensité du bruit de microstructure et on mesure le
//!    biais et la RMSE de chaque estimateur sur plusieurs centaines de
//!    réplications Monte Carlo.

mod estimators;
mod montecarlo;
mod simulate;

use montecarlo::{run_parallel, summarize};
use simulate::true_integrated_variance;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

const SIGMA: f64 = 0.25; // vol annualisée du sous-jacent simulé
const T_HORIZON: f64 = 1.0 / 252.0; // une séance de bourse (en fraction d'année)
const N_REPS: usize = 300; // réplications Monte Carlo par point

fn main() {
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!("Threads disponibles : {n_threads}");

    let true_iv = true_integrated_variance(SIGMA, T_HORIZON);
    println!("Variance intégrée vraie (sigma={SIGMA}, T={T_HORIZON:.6}) : {true_iv:.8}\n");

    run_signature_plot_experiment(true_iv, n_threads);
    run_bias_rmse_experiment(true_iv, n_threads);
}

/// Expérience 1 : signature plot (RV naïve vs TSRV en fonction de la
/// fréquence d'échantillonnage, bruit fixé).
fn run_signature_plot_experiment(true_iv: f64, n_threads: usize) {
    println!("--- Expérience 1 : signature plot ---");
    let noise_std = 0.0005; // ~ bid-ask bounce réaliste en log-prix (quelques points de base)

    // Fréquences testées : de peu de ticks (échantillonnage grossier,
    // 5 min) à énormément de ticks (échantillonnage fin, ~1 seconde).
    let tick_counts: Vec<usize> = vec![
        20, 50, 100, 200, 390, 780, 1560, 3120, 6240, 11700, 23400,
    ];

    let t0 = Instant::now();
    let mut file = File::create("signature_plot.csv").expect("création du CSV");
    writeln!(file, "n_ticks,avg_naive_rv,avg_tsrv,true_iv").unwrap();

    for &n_ticks in &tick_counts {
        let k = estimators::optimal_k(n_ticks);
        let result = run_parallel(N_REPS, n_ticks, SIGMA, T_HORIZON, noise_std, k, 42, n_threads);
        let (mean_naive, _, _) = summarize(&result.naive_rv, true_iv);
        let (mean_tsrv, _, _) = summarize(&result.tsrv, true_iv);

        writeln!(file, "{n_ticks},{mean_naive},{mean_tsrv},{true_iv}").unwrap();
        println!(
            "  n_ticks={n_ticks:6}  K={k:4}  RV_naive={mean_naive:.8}  TSRV={mean_tsrv:.8}"
        );
    }

    println!(
        "Expérience 1 terminée en {:.1} ms -> signature_plot.csv\n",
        t0.elapsed().as_secs_f64() * 1000.0
    );
}

/// Expérience 2 : biais et RMSE des deux estimateurs en fonction de
/// l'intensité du bruit de microstructure, à fréquence réaliste fixée.
fn run_bias_rmse_experiment(true_iv: f64, n_threads: usize) {
    println!("--- Expérience 2 : biais / RMSE vs intensité du bruit ---");
    let n_ticks = 23_400; // ~ 1 observation par seconde sur 6h30 de séance
    let k = estimators::optimal_k(n_ticks);
    let noise_levels = vec![0.0, 0.0001, 0.0002, 0.0005, 0.001, 0.002];

    let t0 = Instant::now();
    let mut file = File::create("bias_rmse.csv").expect("création du CSV");
    writeln!(
        file,
        "noise_std,bias_naive,rmse_naive,bias_tsrv,rmse_tsrv"
    )
    .unwrap();

    for &noise_std in &noise_levels {
        let result = run_parallel(N_REPS, n_ticks, SIGMA, T_HORIZON, noise_std, k, 1234, n_threads);
        let (_, bias_naive, rmse_naive) = summarize(&result.naive_rv, true_iv);
        let (_, bias_tsrv, rmse_tsrv) = summarize(&result.tsrv, true_iv);

        writeln!(
            file,
            "{noise_std},{bias_naive},{rmse_naive},{bias_tsrv},{rmse_tsrv}"
        )
        .unwrap();
        println!(
            "  noise_std={noise_std:.4}  biais_naive={bias_naive:+.8}  RMSE_naive={rmse_naive:.8}  |  biais_TSRV={bias_tsrv:+.8}  RMSE_TSRV={rmse_tsrv:.8}"
        );
    }

    println!(
        "Expérience 2 terminée en {:.1} ms -> bias_rmse.csv",
        t0.elapsed().as_secs_f64() * 1000.0
    );
}
