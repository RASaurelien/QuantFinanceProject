#include "BlackScholes.hpp"
#include <cmath>

namespace BlackScholes {

double normCDF(double x) {
    // Utilise erfc (bibliothèque standard, précise et rapide)
    // plutôt qu'une approximation polynomiale maison.
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double normPDF(double x) {
    static const double INV_SQRT_2PI = 0.3989422804014327;
    return INV_SQRT_2PI * std::exp(-0.5 * x * x);
}

PricingResult price(const Option& opt) {
    const double S = opt.S0, K = opt.K, T = opt.T;
    const double r = opt.r, sig = opt.sigma, q = opt.q;

    // d1 et d2 : les deux quantités pivots de Black-Scholes
    // d1 mesure (en écarts-types) à quel point l'option est dans
    // la monnaie, ajusté de la dérive risque-neutre.
    const double sqrtT = std::sqrt(T);
    const double d1 = (std::log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * sqrtT);
    const double d2 = d1 - sig * sqrtT;

    const double discR = std::exp(-r * T);   // Facteur d'actualisation domestique
    const double discQ = std::exp(-q * T);   // Facteur d'actualisation "dividende"

    PricingResult res;

    if (opt.type == OptionType::Call) {
        res.price = S * discQ * normCDF(d1) - K * discR * normCDF(d2);
        res.delta = discQ * normCDF(d1);
        res.rho   = K * T * discR * normCDF(d2) / 100.0;   // pour 1% de variation de r
        // Theta en decay par jour calendaire (division par 365),
        // convention la plus lisible pour un desk.
        res.theta = (-(S * sig * discQ * normPDF(d1)) / (2.0 * sqrtT)
                     - r * K * discR * normCDF(d2)
                     + q * S * discQ * normCDF(d1)) / 365.0;
    } else { // Put (relation de parité call-put appliquée directement)
        res.price = K * discR * normCDF(-d2) - S * discQ * normCDF(-d1);
        res.delta = discQ * (normCDF(d1) - 1.0);
        res.rho   = -K * T * discR * normCDF(-d2) / 100.0;
        res.theta = (-(S * sig * discQ * normPDF(d1)) / (2.0 * sqrtT)
                     + r * K * discR * normCDF(-d2)
                     - q * S * discQ * normCDF(-d1)) / 365.0;
    }

    // Gamma et Vega sont identiques pour call et put
    // (propriété classique de Black-Scholes).
    res.gamma = discQ * normPDF(d1) / (S * sig * sqrtT);
    res.vega  = S * discQ * normPDF(d1) * sqrtT / 100.0;  // pour 1% de variation de sigma

    res.stderr_ = 0.0;  // Formule fermée : pas d'erreur d'échantillonnage

    return res;
}

} // namespace BlackScholes
