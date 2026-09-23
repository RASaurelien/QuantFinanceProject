#pragma once
#include <cmath>

// Formule fermée Black-Scholes, utilisée uniquement comme référence de
// validation pour le moteur générique (pricing_lib.hpp) — pas un objet
// du système de templates lui-même.
namespace BlackScholesRef {

inline double norm_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

inline double call(double S, double K, double T, double r, double sigma, double q) {
    double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    return S * std::exp(-q * T) * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

inline double put(double S, double K, double T, double r, double sigma, double q) {
    double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    return K * std::exp(-r * T) * norm_cdf(-d2) - S * std::exp(-q * T) * norm_cdf(-d1);
}

}
