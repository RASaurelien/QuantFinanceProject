#include "Option.hpp"
#include "BlackScholes.hpp"
#include "MonteCarlo.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <cstdlib>

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
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    double elapsedMs() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
private:
    std::chrono::high_resolution_clock::time_point start_;
};

static void printRow(const std::string& label, double analytic, double mc, const std::string& unit = "") {
    std::cout << std::left << std::setw(18) << label
              << std::right << std::setw(14) << std::fixed << std::setprecision(6) << analytic
              << std::setw(14) << mc
              << std::setw(14) << (analytic - mc) << unit << "\n";
}

int main(int argc, char** argv) {
    // --- Paramètres par défaut : option at-the-money 6 mois ---
    long nPaths     = (argc > 1) ? std::atol(argv[1]) : 2'000'000;
    double S0       = (argc > 2) ? std::atof(argv[2]) : 100.0;
    double K        = (argc > 3) ? std::atof(argv[3]) : 100.0;
    double T        = (argc > 4) ? std::atof(argv[4]) : 0.5;
    double r        = (argc > 5) ? std::atof(argv[5]) : 0.03;
    double sigma    = (argc > 6) ? std::atof(argv[6]) : 0.20;
    double q        = (argc > 7) ? std::atof(argv[7]) : 0.0;
    std::string typeStr = (argc > 8) ? argv[8] : "call";
    OptionType type = (typeStr == "put") ? OptionType::Put : OptionType::Call;

    Option opt(S0, K, T, r, sigma, q, type);

    std::cout << "==================================================\n";
    std::cout << " Pricing Engine -- Black-Scholes vs Monte Carlo\n";
    std::cout << "==================================================\n";
    std::cout << "Option : " << (type == OptionType::Call ? "CALL" : "PUT")
              << " | S0=" << S0 << " K=" << K << " T=" << T
              << " r=" << r << " sigma=" << sigma << " q=" << q << "\n";
    std::cout << "Chemins Monte Carlo : " << nPaths << "\n\n";

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
    std::cout << std::left << std::setw(18) << "Grandeur"
              << std::right << std::setw(14) << "Analytique"
              << std::setw(14) << "Monte Carlo"
              << std::setw(14) << "Ecart" << "\n";
    std::cout << std::string(60, '-') << "\n";
    printRow("Prix",  bs.price, mcFull.price);
    printRow("Delta", bs.delta, mcFull.delta);
    printRow("Gamma", bs.gamma, mcFull.gamma);
    printRow("Vega",  bs.vega,  mcFull.vega);
    printRow("Theta", bs.theta, mcFull.theta);
    printRow("Rho",   bs.rho,   mcFull.rho);
    std::cout << "\n";

    std::cout << "Erreur standard Monte Carlo (prix) : " << mcSimple.stderr_ << "\n\n";

    std::cout << "--- Temps de calcul ---\n";
    std::cout << "Black-Scholes (analytique)      : " << bsTimeMs     << " ms\n";
    std::cout << "Monte Carlo (prix seul)          : " << mcTimeMs    << " ms\n";
    std::cout << "Monte Carlo (prix + Greeks, CRN)  : " << mcFullTimeMs << " ms\n";

    return 0;
}
