//! Lead-Lag Matrix pour données tick asynchrones (HFT)
//! =====================================================
//!
//! Version Rust, pensée pour la production :
//!   - estimateur de Hayashi-Yoshida en O(n_i + n_j) par paire (two-pointer),
//!   - parallélisation sur toutes les paires d'actifs via rayon,
//!   - pas d'allocation inutile dans la boucle chaude,
//!   - sortie CSV directement exploitable par un moteur de signaux.
//!
//! Référence : Hayashi & Yoshida (2005), Huth & Abergel (2014).

use rand::prelude::*;
use rand_distr::{Exp, Normal};
use std::fs::File;
use std::io::Write;
use std::sync::Mutex;
use std::thread;

/// Une série de ticks pour un actif : timestamps (secondes) et prix, triés par temps.
#[derive(Clone)]
struct TickSeries {
    time: Vec<f64>,
    price: Vec<f64>,
}

impl TickSeries {
    /// Pré-calcule les intervalles (t_start, t_end, log-return) entre ticks consécutifs.
    fn intervals(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = self.time.len();
        let mut t0 = Vec::with_capacity(n - 1);
        let mut t1 = Vec::with_capacity(n - 1);
        let mut r = Vec::with_capacity(n - 1);
        for k in 0..n - 1 {
            t0.push(self.time[k]);
            t1.push(self.time[k + 1]);
            r.push(self.price[k + 1].ln() - self.price[k].ln());
        }
        (t0, t1, r)
    }
}

/// Simule `n_assets` séries de ticks asynchrones avec une structure lead-lag connue :
/// l'actif 0 est le leader, l'actif k suit avec un délai `lag_ms * k` et du bruit idiosyncratique.
fn simulate_asynchronous_ticks(
    n_assets: usize,
    duration_seconds: f64,
    mean_intertrade_ms: f64,
    lag_ms: f64,
    noise_std: f64,
    seed: u64,
) -> Vec<TickSeries> {
    let mut rng = StdRng::seed_from_u64(seed);

    let dt_fine = 0.001_f64; // 1 ms
    let n_steps = (duration_seconds / dt_fine) as usize;

    let ret_dist = Normal::new(0.0, 0.0005).unwrap();
    let mut leader_returns = vec![0.0_f64; n_steps];
    for x in leader_returns.iter_mut() {
        *x = ret_dist.sample(&mut rng);
    }
    let mut leader_price = vec![0.0_f64; n_steps];
    let mut cum = 0.0_f64;
    for i in 0..n_steps {
        cum += leader_returns[i];
        leader_price[i] = 100.0 * cum.exp();
    }

    let lag_steps = ((lag_ms / 1000.0) / dt_fine) as usize;

    let mut asset_paths: Vec<Vec<f64>> = vec![leader_price.clone()];
    let idio_dist = Normal::new(0.0, noise_std * 0.0005).unwrap();
    for k in 1..n_assets {
        let shift = lag_steps * k;
        let mut shifted = vec![leader_price[0]; n_steps];
        for i in shift..n_steps {
            shifted[i] = leader_price[i - shift];
        }
        let mut idio_cum = 0.0_f64;
        let mut path = vec![0.0_f64; n_steps];
        for i in 0..n_steps {
            idio_cum += idio_dist.sample(&mut rng);
            path[i] = shifted[i] * idio_cum.exp();
        }
        asset_paths.push(path);
    }

    let intertrade_dist = Exp::new(1.0 / (mean_intertrade_ms / 1000.0)).unwrap();
    let mut ticks = Vec::with_capacity(n_assets);
    for path in asset_paths {
        let mut t = 0.0_f64;
        let mut times = Vec::new();
        while t < duration_seconds {
            t += intertrade_dist.sample(&mut rng);
            times.push(t);
        }
        let mut prices = Vec::with_capacity(times.len());
        for &tt in &times {
            let idx = ((tt / dt_fine) as usize).min(n_steps - 1);
            prices.push(path[idx]);
        }
        ticks.push(TickSeries { time: times, price: prices });
    }
    ticks
}

/// Covariance de Hayashi-Yoshida entre deux séries tick asynchrones, avec un lag donné.
/// lag > 0 teste "i mène j" : on ramène le futur de j (t+lag) au présent de i (t).
/// Complexité O(n_i + n_j) grâce à un algorithme à deux pointeurs (les temps sont triés).
fn hayashi_yoshida_cov(
    ti0: &[f64], ti1: &[f64], ri: &[f64],
    tj0: &[f64], tj1: &[f64], rj: &[f64],
    lag: f64,
) -> f64 {
    let n_i = ri.len();
    let n_j = rj.len();
    let mut cov = 0.0_f64;
    let mut j_start = 0usize;

    for k in 0..n_i {
        let a0 = ti0[k];
        let a1 = ti1[k];
        while j_start < n_j && (tj1[j_start] - lag) < a0 {
            j_start += 1;
        }
        let mut l = j_start;
        while l < n_j && (tj0[l] - lag) <= a1 {
            let b0 = tj0[l] - lag;
            let b1 = tj1[l] - lag;
            if a0.max(b0) <= a1.min(b1) {
                cov += ri[k] * rj[l];
            }
            l += 1;
        }
    }
    cov
}

fn hayashi_yoshida_corr(
    ti0: &[f64], ti1: &[f64], ri: &[f64],
    tj0: &[f64], tj1: &[f64], rj: &[f64],
    var_i: f64, var_j: f64,
    lag: f64,
) -> f64 {
    let cov = hayashi_yoshida_cov(ti0, ti1, ri, tj0, tj1, rj, lag);
    let denom = (var_i * var_j).sqrt();
    if denom > 0.0 { cov / denom } else { 0.0 }
}

struct LeadLagResult {
    n: usize,
    optimal_lag: Vec<f64>,       // n x n, en secondes
    leadership_score: Vec<f64>,  // n x n
}

/// Construit la matrice lead-lag complète. Parallélisée sur toutes les paires (i,j) avec rayon.
fn build_lead_lag_matrix(ticks: &[TickSeries], lags_s: &[f64]) -> LeadLagResult {
    let n = ticks.len();

    // pré-calcul des intervalles et variances une seule fois par actif
    let precomp: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = ticks.iter().map(|t| t.intervals()).collect();
    let vars: Vec<f64> = precomp
        .iter()
        .map(|(t0, t1, r)| hayashi_yoshida_cov(t0, t1, r, t0, t1, r, 0.0))
        .collect();

    // toutes les paires (i,j), i != j
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .filter(|&(i, j)| i != j)
        .collect();

    // Parallélisation manuelle avec des threads scopés (std::thread::scope, stable
    // depuis Rust 1.63) : on répartit les paires (i,j) entre les coeurs disponibles,
    // chaque thread empruntant `precomp`/`vars`/`lags_s` sans copie ni `Arc`.
    let n_threads = thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let chunk_size = (pairs.len() + n_threads - 1) / n_threads.max(1);
    let collected: Mutex<Vec<((usize, usize), f64, f64)>> = Mutex::new(Vec::with_capacity(pairs.len()));

    thread::scope(|scope| {
        for chunk in pairs.chunks(chunk_size.max(1)) {
            let precomp = &precomp;
            let vars = &vars;
            let lags_s = &lags_s;
            let collected = &collected;
            scope.spawn(move || {
                let mut local = Vec::with_capacity(chunk.len());
                for &(i, j) in chunk {
                    let (ti0, ti1, ri) = &precomp[i];
                    let (tj0, tj1, rj) = &precomp[j];

                    let corrs: Vec<f64> = lags_s
                        .iter()
                        .map(|&lag| {
                            hayashi_yoshida_corr(ti0, ti1, ri, tj0, tj1, rj, vars[i], vars[j], lag)
                        })
                        .collect();

                    let best_idx = corrs
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
                        .map(|(idx, _)| idx)
                        .unwrap();
                    let optimal_lag = lags_s[best_idx];

                    let mut pos_sum = 0.0;
                    let mut neg_sum = 0.0;
                    for (&lag, &c) in lags_s.iter().zip(corrs.iter()) {
                        if lag > 0.0 {
                            pos_sum += c;
                        } else if lag < 0.0 {
                            neg_sum += c;
                        }
                    }
                    let leadership_score = pos_sum - neg_sum;

                    local.push(((i, j), optimal_lag, leadership_score));
                }
                collected.lock().unwrap().extend(local);
            });
        }
    });

    let results = collected.into_inner().unwrap();

    let mut optimal_lag = vec![0.0_f64; n * n];
    let mut leadership_score = vec![0.0_f64; n * n];
    for ((i, j), lag, score) in results {
        optimal_lag[i * n + j] = lag;
        leadership_score[i * n + j] = score;
    }

    LeadLagResult { n, optimal_lag, leadership_score }
}

fn write_csv(result: &LeadLagResult, path: &str) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    writeln!(f, "i,j,optimal_lag_ms,leadership_score")?;
    for i in 0..result.n {
        for j in 0..result.n {
            if i == j {
                continue;
            }
            writeln!(
                f,
                "{},{},{:.3},{:.6}",
                i,
                j,
                result.optimal_lag[i * result.n + j] * 1000.0,
                result.leadership_score[i * result.n + j]
            )?;
        }
    }
    Ok(())
}

fn main() {
    let n_assets = 4;
    let start = std::time::Instant::now();

    println!("Simulation de {} actifs (actif 0 = leader, lag=50ms)...", n_assets);
    let ticks = simulate_asynchronous_ticks(n_assets, 600.0, 200.0, 50.0, 0.3, 42);
    for (k, t) in ticks.iter().enumerate() {
        println!("  actif {}: {} ticks", k, t.time.len());
    }

    let lags_ms: Vec<i32> = (-150..=150).step_by(10).collect();
    let lags_s: Vec<f64> = lags_ms.iter().map(|&l| l as f64 / 1000.0).collect();

    println!("\nCalcul de la matrice lead-lag (Hayashi-Yoshida, {} lags x {} paires)...",
             lags_s.len(), n_assets * (n_assets - 1));
    let result = build_lead_lag_matrix(&ticks, &lags_s);

    println!("\nMatrice des lags optimaux (ms) :");
    print!("      ");
    for j in 0..n_assets { print!("{:>8}", j); }
    println!();
    for i in 0..n_assets {
        print!("  {:>3} ", i);
        for j in 0..n_assets {
            print!("{:>8.0}", result.optimal_lag[i * n_assets + j] * 1000.0);
        }
        println!();
    }

    println!("\nScore de leadership net par actif :");
    for i in 0..n_assets {
        let net: f64 = (0..n_assets).map(|j| result.leadership_score[i * n_assets + j]).sum();
        println!("  actif {}: {:+.3}", i, net);
    }

    write_csv(&result, "leadlag_matrix.csv").expect("écriture CSV échouée");
    println!("\nMatrice complète exportée vers leadlag_matrix.csv");

    println!("\nTemps total : {:.3} ms", start.elapsed().as_secs_f64() * 1000.0);
}
