#pragma once
#include <stdexcept>

// ============================================================
//  Option.hpp
//  Représente un contrat d'option vanille européenne.
//  Toutes les autres briques (pricers, calcul des Greeks)
//  consomment cette structure : c'est le "contrat de données"
//  du projet.
// ============================================================

enum class OptionType { Call, Put };

struct Option {
    double S0;      // Prix spot du sous-jacent à t=0
    double K;       // Strike (prix d'exercice)
    double T;       // Maturité en années (ex : 0.5 = 6 mois)
    double r;       // Taux sans risque (continu, annualisé)
    double sigma;   // Volatilité annualisée du sous-jacent
    double q;       // Taux de dividende continu (0 si pas de dividende)
    OptionType type;

    Option(double S0_, double K_, double T_, double r_,
           double sigma_, double q_, OptionType type_)
        : S0(S0_), K(K_), T(T_), r(r_), sigma(sigma_), q(q_), type(type_)
    {
        // Garde-fous : on refuse de pricer un contrat mal spécifié
        // plutôt que de laisser un NaN se propager silencieusement.
        if (S0 <= 0.0 || K <= 0.0)
            throw std::invalid_argument("S0 et K doivent être strictement positifs.");
        if (T <= 0.0)
            throw std::invalid_argument("La maturité T doit être strictement positive.");
        if (sigma <= 0.0)
            throw std::invalid_argument("La volatilité sigma doit être strictement positive.");
    }
};

// Regroupe le prix + toutes les sensibilités (Greeks) d'un contrat.
// Retourné à la fois par le pricer analytique et par Monte Carlo,
// ce qui permet de comparer les deux directement.
struct PricingResult {
    double price   = 0.0;
    double delta   = 0.0;   // dV/dS
    double gamma   = 0.0;   // d²V/dS²
    double vega    = 0.0;   // dV/dsigma
    double theta   = 0.0;   // dV/dt (décroissance temporelle, par jour calendaire)
    double rho     = 0.0;   // dV/dr
    double stderr_ = 0.0;   // Écart-type de l'estimateur (0 pour l'analytique, >0 pour Monte Carlo)
};
