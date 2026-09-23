#pragma once
#include "Option.hpp"
#include <vector>
#include <limits>

// ============================================================
//  AmericanPDE.hpp
//  Résout l'équation de Black-Scholes-Merton par différences
//  finies (schéma de Crank-Nicolson) sur une grille (S, t), pour
//  des options européennes ET américaines.
//
//  Le point clé du projet : pour l'exercice anticipé américain,
//  on ne fait PAS une simple actualisation de payoff (ce serait
//  faux), mais on résout un problème de complémentarité linéaire
//  (LCP) : à chaque pas de temps, V(S,t) = max(continuation, payoff).
//
//  Algorithme retenu : Brennan-Schwartz (1977). C'est un choix
//  d'optimisation déliberé par rapport à un PSOR (relaxation
//  itérative, O(N x iterations) par pas de temps) : Brennan-
//  Schwartz résout le LCP en une seule passe O(N) par pas de
//  temps, en projetant directement pendant la substitution
//  arrière du solveur tridiagonal (valide car la matrice du
//  schéma est une M-matrice à diagonale dominante).
// ============================================================

enum class ExerciseStyle { European, American };

struct GridResult {
    std::vector<double> S;                        // grille spatiale (prix du sous-jacent)
    std::vector<std::vector<double>> V;            // V[n][i] : valeur de l'option au pas de temps n, noeud S[i]
    std::vector<double> exerciseBoundary;          // frontière d'exercice anticipé par pas de temps (NaN si aucune)
    double dt = 0.0;

    double price = 0.0;
    double delta = 0.0;
    double gamma = 0.0;
    double theta = 0.0;
};

namespace AmericanPDE {

    // N : nombre de pas spatiaux (grille S : N+1 noeuds, dont 2 frontières)
    // M : nombre de pas temporels
    // SmaxMult : la grille couvre S in [0, SmaxMult * K]
    GridResult solve(const Option& opt, ExerciseStyle style,
                      int N = 400, int M = 400, double SmaxMult = 3.0);

}
