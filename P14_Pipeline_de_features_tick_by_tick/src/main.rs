//! main.rs
//! =======
//! 1) Traite un grand flux de ticks (5M) en une seule passe streaming,
//!    mesure le débit (ticks/seconde), exporte un echantillon en CSV.
//! 2) Benchmark : compare le cout d'une fenetre glissante "naive"
//!    (recalcul complet de l'ecart-type sur les W derniers ticks a
//!    CHAQUE tick, O(n*W)) contre l'approche streaming O(n) -- pour
//!    quantifier, pas juste affirmer, le gain de la conception O(1).

mod features;
mod generator;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::process::Command;
use std::time::Instant;

use features::StreamingFeatureEngine;
use generator::TickGenerator;

fn run_streaming_pipeline(n_ticks: u64, sample_every: u64) {
    let mut engine = StreamingFeatureEngine::new(0.05);
    let gen = TickGenerator::new(n_ticks, 100.0, 0.01, 42);

    let file = File::create("tick_features_sample.csv").expect("creation du CSV");
    let mut writer = BufWriter::new(file);   // ecriture bufferisee : pas d'appel systeme par ligne
    writeln!(writer, "index,mid,spread,microprice,imbalance,ema_microprice,realized_vol").unwrap();

    let t0 = Instant::now();
    let mut n_processed: u64 = 0;

    for tick in gen {
        let row = engine.update(&tick);
        n_processed += 1;

        if row.index % sample_every == 0 {
            writeln!(writer, "{},{:.5},{:.5},{:.5},{:.5},{:.5},{:.7}",
                     row.index, row.mid, row.spread, row.microprice,
                     row.imbalance, row.ema_microprice, row.realized_vol).unwrap();
        }
    }
    writer.flush().unwrap();

    let elapsed = t0.elapsed().as_secs_f64();
    println!("Pipeline streaming : {} ticks traites en {:.3} s ({:.2} millions de ticks/s)",
             n_processed, elapsed, n_processed as f64 / elapsed / 1e6);
    println!("Echantillon exporte -> tick_features_sample.csv (1 ligne / {} ticks)\n", sample_every);
}

/// Version "naive" pedagogique : recalcule l'ecart-type sur toute la
/// fenetre des W derniers rendements A CHAQUE TICK, en stockant la
/// fenetre dans un Vec -- ce que ferait un premier jet non optimise
/// (`prices[i-W..i].std()` appele en boucle). Complexite O(n*W).
fn naive_windowed_std(n_ticks: u64, window: usize) -> f64 {
    let gen = TickGenerator::new(n_ticks, 100.0, 0.01, 42);
    let mut returns: Vec<f64> = Vec::with_capacity(window);
    let mut prev_mid = 100.0;
    let mut last_std = 0.0;

    for tick in gen {
        let mid = 0.5 * (tick.bid + tick.ask);
        let ret = mid / prev_mid - 1.0;
        prev_mid = mid;

        returns.push(ret);
        if returns.len() > window {
            returns.remove(0);   // O(W) a chaque tick : c'est la ou le cout explose
        }

        if returns.len() >= 2 {
            let mean = returns.iter().sum::<f64>() / returns.len() as f64;
            let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (returns.len() - 1) as f64;
            last_std = var.sqrt();
        }
    }
    last_std
}

fn benchmark_naive_vs_streaming() {
    println!("--- Benchmark : fenetre naive O(n*W) vs streaming Welford O(n) ---\n");
    let n_bench = 200_000u64;
    let window = 500usize;

    let t0 = Instant::now();
    let _ = naive_windowed_std(n_bench, window);
    let naive_time = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let mut engine = StreamingFeatureEngine::new(0.05);
    for tick in TickGenerator::new(n_bench, 100.0, 0.01, 42) {
        let _ = engine.update(&tick);
    }
    let streaming_time = t0.elapsed().as_secs_f64();

    println!("Naive (fenetre {} recalculee a chaque tick) : {:.3} s", window, naive_time);
    println!("Streaming (Welford, O(1) par tick)            : {:.3} s", streaming_time);
    println!("Facteur d'acceleration                          : {:.1}x\n", naive_time / streaming_time);
    println!("(Note : le streaming calcule une vol sur TOUT l'historique, la version naive");
    println!(" sur une fenetre de {} -- la comparaison porte sur le COUT PAR TICK de la mise", window);
    println!(" a jour incrementale vs. le recalcul complet, pas sur une metrique identique.)\n");
}

fn generate_plot() {
    let python = if cfg!(windows) { "python" } else { "python3" };
    let status = Command::new(python)
        .arg("plot_results.py")
        .current_dir(".")
        .status()
        .expect("impossible de lancer plot_results.py");

    if !status.success() {
        panic!("plot_results.py s'est termine avec une erreur");
    }
}

fn main() {
    println!("=======================================================\n");
    println!(" Pipeline de features tick-by-tick haute performance (Rust)\n");
    println!("=======================================================\n");

    run_streaming_pipeline(5_000_000, 5_000);
    generate_plot();
    benchmark_naive_vs_streaming();
}
