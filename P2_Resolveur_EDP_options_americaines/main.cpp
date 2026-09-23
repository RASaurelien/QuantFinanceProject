#include "Option.hpp"
#include "BlackScholes.hpp"
#include "AmericanPDE.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <string>
#include <cstdlib>

// ============================================================
//  main.cpp
//  1) Valide le solveur EDP en mode européen contre la formule
//     fermée de Black-Scholes (doit converger quand N, M augmentent).
//  2) Calcule le prix américain et la prime d'exercice anticipé
//     (American - European).
//  3) Exporte la frontière d'exercice anticipé dans un CSV,
//     directement exploitable pour un graphique.
//
//  Usage :
//    ./american_pde [N] [M] [S0] [K] [T] [r] [sigma] [q] [call|put]
// ============================================================

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

int main(int argc, char** argv) {
    int N            = (argc > 1) ? std::atoi(argv[1]) : 400;
    int M            = (argc > 2) ? std::atoi(argv[2]) : 400;
    double S0        = (argc > 3) ? std::atof(argv[3]) : 100.0;
    double K         = (argc > 4) ? std::atof(argv[4]) : 100.0;
    double T         = (argc > 5) ? std::atof(argv[5]) : 1.0;
    double r         = (argc > 6) ? std::atof(argv[6]) : 0.05;
    double sigma     = (argc > 7) ? std::atof(argv[7]) : 0.25;
    double q         = (argc > 8) ? std::atof(argv[8]) : 0.03;   // dividende non nul : rend l'exercice anticipé du call intéressant
    std::string typeStr = (argc > 9) ? argv[9] : "put";
    OptionType type = (typeStr == "call") ? OptionType::Call : OptionType::Put;

    Option opt(S0, K, T, r, sigma, q, type);

    std::cout << "==================================================\n";
    std::cout << " Solveur EDP (Crank-Nicolson + Brennan-Schwartz)\n";
    std::cout << " Options americaines vs europeennes\n";
    std::cout << "==================================================\n";
    std::cout << "Option : " << (type == OptionType::Call ? "CALL" : "PUT")
              << " | S0=" << S0 << " K=" << K << " T=" << T
              << " r=" << r << " sigma=" << sigma << " q=" << q << "\n";
    std::cout << "Grille : N=" << N << " (espace) x M=" << M << " (temps)\n\n";

    // --- 1) Validation : EDP europeenne vs Black-Scholes analytique ---
    Timer t1;
    GridResult euro = AmericanPDE::solve(opt, ExerciseStyle::European, N, M);
    double euroTimeMs = t1.elapsedMs();
    PricingResult bs = BlackScholes::price(opt);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "--- Validation (europeenne) ---\n";
    std::cout << std::left << std::setw(12) << "Grandeur"
              << std::right << std::setw(14) << "Black-Scholes"
              << std::setw(14) << "EDP (Euro)"
              << std::setw(14) << "Ecart" << "\n";
    std::cout << std::string(54, '-') << "\n";
    std::cout << std::left << std::setw(12) << "Prix"
              << std::right << std::setw(14) << bs.price << std::setw(14) << euro.price
              << std::setw(14) << (bs.price - euro.price) << "\n";
    std::cout << std::left << std::setw(12) << "Delta"
              << std::right << std::setw(14) << bs.delta << std::setw(14) << euro.delta
              << std::setw(14) << (bs.delta - euro.delta) << "\n";
    std::cout << std::left << std::setw(12) << "Gamma"
              << std::right << std::setw(14) << bs.gamma << std::setw(14) << euro.gamma
              << std::setw(14) << (bs.gamma - euro.gamma) << "\n";
    std::cout << std::left << std::setw(12) << "Theta"
              << std::right << std::setw(14) << bs.theta << std::setw(14) << euro.theta
              << std::setw(14) << (bs.theta - euro.theta) << "\n\n";

    // --- 2) Prix americain + prime d'exercice anticipe ---
    Timer t2;
    GridResult amer = AmericanPDE::solve(opt, ExerciseStyle::American, N, M);
    double amerTimeMs = t2.elapsedMs();

    const double earlyExercisePremium = amer.price - euro.price;

    std::cout << "--- Americaine vs Europeenne ---\n";
    std::cout << "Prix europeen (EDP)  : " << euro.price << "\n";
    std::cout << "Prix americain (EDP) : " << amer.price << "\n";
    std::cout << "Prime d'exercice anticipe : " << earlyExercisePremium
               << "  (" << (100.0 * earlyExercisePremium / euro.price) << " % du prix europeen)\n";
    std::cout << "Delta americain : " << amer.delta << " | Gamma : " << amer.gamma
               << " | Theta : " << amer.theta << "\n\n";

    // Garde-fou de cohérence économique : le prix américain ne peut
    // jamais être inférieur au prix européen, ni à la valeur intrinsèque.
    if (amer.price < euro.price - 1e-4)
        std::cerr << "ATTENTION : incoherence detectee (americain < europeen)\n";

    std::cout << "--- Temps de calcul ---\n";
    std::cout << "EDP europeenne : " << euroTimeMs << " ms\n";
    std::cout << "EDP americaine : " << amerTimeMs << " ms (Brennan-Schwartz, O(N) par pas de temps)\n\n";

    // --- 3) Export CSV de la frontiere d'exercice anticipe ---
    if (type == OptionType::Put || type == OptionType::Call) {
        std::ofstream csv("exercise_boundary.csv");
        csv << "t,S_boundary\n";
        for (int nStep = 0; nStep <= M; ++nStep) {
            double t = nStep * amer.dt;
            csv << t << "," << amer.exerciseBoundary[nStep] << "\n";
        }
        csv.close();
        std::cout << "Frontiere d'exercice anticipe exportee -> exercise_boundary.csv\n";
    }

    return 0;
}
