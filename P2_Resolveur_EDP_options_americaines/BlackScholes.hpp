#pragma once
#include "Option.hpp"

// ============================================================
//  BlackScholes.hpp
//  Pricing fermé (closed-form) sous le modèle de Black-Scholes-
//  Merton, avec dividende continu q. Sert de référence "vérité
//  terrain" pour valider le moteur Monte Carlo (Tier 1 #1).
// ============================================================

namespace BlackScholes {

    // Fonction de répartition (CDF) et densité (PDF) de la loi
    // normale centrée réduite. Utilisées partout dans le pricing
    // et le calcul des Greeks fermés.
    double normCDF(double x);
    double normPDF(double x);

    // Calcule le prix + tous les Greeks sous forme fermée.
    // C'est O(1) : pas de simulation, juste l'évaluation de la
    // formule de Black-Scholes-Merton et de ses dérivées.
    PricingResult price(const Option& opt);

}
