#include "Option.hpp"
#include "BlackScholes.hpp"
#include "MonteCarlo.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <cstdlib>
using namespace std;

// ============================================================
//  main.cpp
//  Point d'entrée : construit une option vanille, la price avec
//  les deux moteurs (analytique + Monte Carlo), affiche une
//  table comparative des prix / Greeks / temps de calcul.
//
//  Usage :
//    ./pricer [nPaths] [S0] [K] [T] [r] [sigma] [q] [call|put]
//  Tous les arguments sont optionnels (valeurs par défaut ci-dessous).
// ============================================================

// Petit chronomètre RAII-friendly pour mesurer un bloc de calcul.
class Timer {
public:
    Timer() : start_(chrono::high_resolution_clock::now()) {}
    double elapsedMs() const {
        auto end = chrono::high_resolution_clock::now();
        return chrono::duration<double, milli>(end - start_).count();
    }
private:
    chrono::high_resolution_clock::time_point start_;
};

static void printRow(const string& label, double analytic, double mc, const string& unit = "") {
    cout << left << setw(18) << label
              << right << setw(14) << fixed << setprecision(6) << analytic
              << setw(14) << mc
              << setw(14) << (analytic - mc) << unit << "\n";
}

int main(int argc, char** argv) {
    // --- Paramètres par défaut : option at-the-money 6 mois ---
    long nPaths     = (argc > 1) ? atol(argv[1]) : 2'000'000;
    double S0       = (argc > 2) ? atof(argv[2]) : 100.0;
    double K        = (argc > 3) ? atof(argv[3]) : 100.0;
    double T        = (argc > 4) ? atof(argv[4]) : 0.5;
    double r        = (argc > 5) ? atof(argv[5]) : 0.03;
    double sigma    = (argc > 6) ? atof(argv[6]) : 0.20;
    double q        = (argc > 7) ? atof(argv[7]) : 0.0;
    string typeStr = (argc > 8) ? argv[8] : "call";
    OptionType type = (typeStr == "put") ? OptionType::Put : OptionType::Call;

    Option opt(S0, K, T, r, sigma, q, type);

    cout << "==================================================\n";
    cout << " Pricing Engine -- Black-Scholes vs Monte Carlo\n";
    cout << "==================================================\n";
    cout << "Option : " << (type == OptionType::Call ? "CALL" : "PUT")
              << " | S0=" << S0 << " K=" << K << " T=" << T
              << " r=" << r << " sigma=" << sigma << " q=" << q << "\n";
    cout << "Chemins Monte Carlo : " << nPaths << "\n\n";

    // --- 1) Pricer analytique (référence, O(1)) ---
    Timer t1;
    PricingResult bs = BlackScholes::price(opt);
    double bsTimeMs = t1.elapsedMs();

    // --- 2) Pricer Monte Carlo, prix seul (sans Greeks) ---
    Timer t2;
    PricingResult mcSimple = MonteCarlo::price(opt, nPaths);
    double mcTimeMs = t2.elapsedMs();

    // --- 3) Pricer Monte Carlo avec Greeks (differences finies + CRN) ---
    Timer t3;
    PricingResult mcFull = MonteCarlo::priceWithGreeks(opt, nPaths);
    double mcFullTimeMs = t3.elapsedMs();

    // --- Table comparative ---
    cout << left << setw(18) << "Grandeur"
              << right << setw(14) << "Analytique"
              << setw(14) << "Monte Carlo"
              << setw(14) << "Ecart" << "\n";
    cout << string(60, '-') << "\n";
    printRow("Prix",  bs.price, mcFull.price);
    printRow("Delta", bs.delta, mcFull.delta);
    printRow("Gamma", bs.gamma, mcFull.gamma);
    printRow("Vega",  bs.vega,  mcFull.vega);
    printRow("Theta", bs.theta, mcFull.theta);
    printRow("Rho",   bs.rho,   mcFull.rho);
    cout << "\n";

    cout << "Erreur standard Monte Carlo (prix) : " << mcSimple.stderr_ << "\n\n";

    cout << "--- Temps de calcul ---\n";
    cout << "Black-Scholes (analytique)      : " << bsTimeMs     << " ms\n";
    cout << "Monte Carlo (prix seul)          : " << mcTimeMs    << " ms\n";
    cout << "Monte Carlo (prix + Greeks, CRN)  : " << mcFullTimeMs << " ms\n";

    return 0;
}
