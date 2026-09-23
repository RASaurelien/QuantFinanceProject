#include "MonteCarlo.hpp"
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace MonteCarlo {

// ------------------------------------------------------------
// simulateSum : coeur du moteur.
// Simule nPaths trajectoires terminales sous GBM risque-neutre
// et retourne (somme des payoffs actualisés, somme des carrés)
// pour permettre à l'appelant de calculer prix + variance sans
// stocker tous les tirages en mémoire (streaming, O(1) mémoire
// par thread).
//
// Le couple (seed, index de thread) définit intégralement le
// flux aléatoire utilisé : à seed fixé, deux appels avec des
// paramètres d'option différents (ex. S0 bumpé) consomment
// EXACTEMENT les mêmes nombres aléatoires. C'est ce qui permet
// les Common Random Numbers dans priceWithGreeks().
// ------------------------------------------------------------
static void simulateSum(const Option& opt, long nPaths, unsigned seed,
                         double& outSum, double& outSumSq)
{
    const double drift = (opt.r - opt.q - 0.5 * opt.sigma * opt.sigma) * opt.T;
    const double diffusion = opt.sigma * std::sqrt(opt.T);
    const double discount = std::exp(-opt.r * opt.T);
    const bool isCall = (opt.type == OptionType::Call);
    const double K = opt.K;

    double totalSum = 0.0, totalSumSq = 0.0;

    // Chaque thread traite un bloc de paires antithétiques et
    // maintient son propre générateur (mt19937_64), initialisé
    // avec un seed dérivé mais déterministe -> résultats
    // reproductibles et parallélisme sans verrou.
    #pragma omp parallel reduction(+:totalSum, totalSumSq)
    {
        #ifdef _OPENMP
        const int nThreads = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        #else
        const int nThreads = 1;
        const int tid = 0;
        #endif

        std::mt19937_64 rng(seed + static_cast<unsigned>(tid) * 7919u); // 7919 premier: espace bien les flux
        std::normal_distribution<double> stdNormal(0.0, 1.0);

        const long pairsTotal = nPaths / 2;               // on travaille par paires antithétiques
        const long pairsPerThread = pairsTotal / nThreads;
        const long start = tid * pairsPerThread;
        const long end = (tid == nThreads - 1) ? pairsTotal : start + pairsPerThread;

        double localSum = 0.0, localSumSq = 0.0;

        for (long i = start; i < end; ++i) {
            const double z = stdNormal(rng);

            // Trajectoire "normale"
            const double ST1 = opt.S0 * std::exp(drift + diffusion * z);
            // Trajectoire antithétique (miroir -z) : corrèle
            // négativement l'erreur et réduit la variance sans
            // biaiser l'estimateur (l'espérance de -Z est la même).
            const double ST2 = opt.S0 * std::exp(drift - diffusion * z);

            const double payoff1 = isCall ? std::max(ST1 - K, 0.0) : std::max(K - ST1, 0.0);
            const double payoff2 = isCall ? std::max(ST2 - K, 0.0) : std::max(K - ST2, 0.0);

            // Moyenne de la paire : c'est CE couple qui constitue
            // un seul "tirage" pour l'estimateur antithétique.
            const double avgPayoff = 0.5 * (payoff1 + payoff2) * discount;

            localSum += avgPayoff;
            localSumSq += avgPayoff * avgPayoff;
        }

        totalSum += localSum;
        totalSumSq += localSumSq;
    }

    outSum = totalSum;
    outSumSq = totalSumSq;
}

PricingResult price(const Option& opt, long nPaths, unsigned seed) {
    if (nPaths % 2 != 0) nPaths += 1;  // on force un nombre pair (paires antithétiques)

    double sum = 0.0, sumSq = 0.0;
    simulateSum(opt, nPaths, seed, sum, sumSq);

    const long nSamples = nPaths / 2;  // une paire antithétique = un échantillon
    const double mean = sum / nSamples;
    const double variance = (sumSq / nSamples) - (mean * mean);
    const double stderrEst = std::sqrt(std::max(variance, 0.0) / nSamples);

    PricingResult res;
    res.price = mean;
    res.stderr_ = stderrEst;
    return res;
}

PricingResult priceWithGreeks(const Option& opt, long nPaths, unsigned seed) {
    // --- Pas de discrétisation pour chaque Greek ---
    const double hS   = 0.01 * opt.S0;   // bump relatif sur le spot
    const double hSig = 0.0001;          // bump absolu sur la vol (1 bp)
    const double hR   = 0.0001;          // bump absolu sur le taux (1 bp)
    const double dT   = 1.0 / 365.0;     // un jour calendaire, pour theta

    // Petit utilitaire local : reconstruit une Option bumpée sur
    // un seul paramètre à la fois (les autres restent inchangés).
    auto bumped = [&](double dS, double dSig, double dR, double dT_) {
        return Option(opt.S0 + dS, opt.K, opt.T - dT_, opt.r + dR,
                      opt.sigma + dSig, opt.q, opt.type);
    };

    // IMPORTANT : le même seed est réutilisé pour CHAQUE scénario.
    // Comme simulateSum() consomme les nombres aléatoires dans le
    // même ordre indépendamment des paramètres de l'option, les
    // tirages sont identiques entre scénarios -> Common Random
    // Numbers -> les différences de prix ne reflètent (quasiment)
    // que l'effet du bump, pas le bruit Monte Carlo.
    auto simplePrice = [&](const Option& o) {
        double s = 0.0, sq = 0.0;
        simulateSum(o, nPaths, seed, s, sq);
        return s / (nPaths / 2);
    };

    const double V0    = simplePrice(opt);
    const double Vsp   = simplePrice(bumped(+hS, 0, 0, 0));
    const double Vsm   = simplePrice(bumped(-hS, 0, 0, 0));
    const double Vsigp = simplePrice(bumped(0, +hSig, 0, 0));
    const double Vsigm = simplePrice(bumped(0, -hSig, 0, 0));
    const double Vrp   = simplePrice(bumped(0, 0, +hR, 0));
    const double Vrm   = simplePrice(bumped(0, 0, -hR, 0));
    const double VtMinus = simplePrice(bumped(0, 0, 0, +dT)); // T réduit de dT

    PricingResult res;
    res.price = V0;
    res.delta = (Vsp - Vsm) / (2.0 * hS);
    res.gamma = (Vsp - 2.0 * V0 + Vsm) / (hS * hS);
    res.vega  = (Vsigp - Vsigm) / (2.0 * hSig) / 100.0;   // pour 1% de vol
    res.rho   = (Vrp - Vrm) / (2.0 * hR) / 100.0;         // pour 1% de taux
    res.theta = (VtMinus - V0) / dT / 365.0;              // decay par jour calendaire

    // Écart-type indicatif : on renvoie celui de l'estimateur de
    // prix de base (les Greeks par CRN n'ont pas une formule de
    // variance aussi directe, mais leur precision réelle est très
    // supérieure grâce à l'annulation du bruit commun).
    double s0 = 0.0, sq0 = 0.0;
    simulateSum(opt, nPaths, seed, s0, sq0);
    const long nSamples = nPaths / 2;
    const double mean = s0 / nSamples;
    const double variance = (sq0 / nSamples) - (mean * mean);
    res.stderr_ = std::sqrt(std::max(variance, 0.0) / nSamples);

    return res;
}

} // namespace MonteCarlo
