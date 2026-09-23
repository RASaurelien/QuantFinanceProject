#pragma once
#include "Option.hpp"

// ============================================================
//  MonteCarlo.hpp
//  Pricer par simulation sous mesure risque-neutre (modèle GBM).
//  Sert de validation croisée du pricer analytique, et surtout
//  de brique réutilisable pour des payoffs qui n'ont PAS de
//  formule fermée (options exotiques, futurs projets Tier 1/2).
//
//  Techniques de réduction de variance / performance :
//   - Antithetic variates (paires Z / -Z)
//   - Common Random Numbers (CRN) pour les Greeks par différences
//     finies : on réutilise EXACTEMENT les mêmes tirages aléatoires
//     entre le scénario "de base" et le scénario "bumpé", ce qui
//     annule une grande partie du bruit de Monte Carlo dans la
//     différence.
//   - Parallélisation OpenMP avec un flux RNG indépendant par
//     thread (pas de verrou, pas de contention).
// ============================================================

namespace MonteCarlo {

    // Prix + écart-type de l'estimateur, sans les Greeks.
    // nPaths doit être pair (paires antithétiques).
    PricingResult price(const Option& opt, long nPaths, unsigned seed = 42);

    // Prix + Greeks calculés par différences finies avec Common
    // Random Numbers. Plus coûteux (plusieurs simulations bumpées)
    // mais chaque Greek converge beaucoup plus vite que le prix
    // brut grâce à l'annulation du bruit commun.
    PricingResult priceWithGreeks(const Option& opt, long nPaths, unsigned seed = 42);

}
