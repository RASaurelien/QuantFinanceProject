//! montecarlo.rs
//! =============
//! Lance M réplications indépendantes de la simulation + des deux
//! estimateurs, réparties sur plusieurs threads via `std::thread::scope`
//! (même approche que le projet lead-lag existant : pas de dépendance
//! externe de type rayon, un flux RNG indépendant par thread, aucun
//! verrou).

use crate::estimators::{realized_variance, tsrv};
use crate::simulate::simulate_noisy_gbm;
use rand::SeedableRng;
use rand::rngs::StdRng;

pub struct McResult {
    pub naive_rv: Vec<f64>,
    pub tsrv: Vec<f64>,
}

/// Exécute `n_reps` réplications indépendantes en parallèle et retourne
/// les valeurs brutes des deux estimateurs pour chaque réplication
/// (l'appelant calcule ensuite moyenne/biais/RMSE — on garde le détail
/// pour pouvoir aussi tracer la distribution si besoin).
pub fn run_parallel(
    n_reps: usize,
    n_ticks: usize,
    sigma: f64,
    t_horizon: f64,
    noise_std: f64,
    k: usize,
    base_seed: u64,
    n_threads: usize,
) -> McResult {
    let n_threads = n_threads.max(1).min(n_reps.max(1));
    let chunk = (n_reps + n_threads - 1) / n_threads;

    let mut naive_rv = vec![0.0_f64; n_reps];
    let mut tsrv_vals = vec![0.0_f64; n_reps];

    // On découpe les buffers de résultats en tranches disjointes, une
    // par thread : chaque thread écrit dans sa propre tranche, donc
    // aucune synchronisation n'est nécessaire pendant le calcul.
    {
        let naive_chunks = naive_rv.chunks_mut(chunk);
        let tsrv_chunks = tsrv_vals.chunks_mut(chunk);

        std::thread::scope(|scope| {
            for (thread_id, (naive_slice, tsrv_slice)) in naive_chunks.zip(tsrv_chunks).enumerate() {
                let seed = base_seed.wrapping_add((thread_id as u64).wrapping_mul(0x9E3779B97F4A7C15));
                scope.spawn(move || {
                    let mut rng = StdRng::seed_from_u64(seed);
                    for i in 0..naive_slice.len() {
                        let path = simulate_noisy_gbm(&mut rng, n_ticks, sigma, t_horizon, noise_std);
                        naive_slice[i] = realized_variance(&path);
                        tsrv_slice[i] = tsrv(&path, k);
                    }
                });
            }
        });
    }

    McResult { naive_rv, tsrv: tsrv_vals }
}

/// Statistiques usuelles (biais, RMSE) d'un estimateur par rapport à
/// la vraie variance intégrée, à partir des valeurs de Monte Carlo.
pub fn summarize(values: &[f64], true_iv: f64) -> (f64, f64, f64) {
    let n = values.len() as f64;
    let mean: f64 = values.iter().sum::<f64>() / n;
    let bias = mean - true_iv;
    let mse: f64 = values.iter().map(|v| (v - true_iv).powi(2)).sum::<f64>() / n;
    (mean, bias, mse.sqrt())
}
