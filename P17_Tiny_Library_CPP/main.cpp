#include "pricing_lib.hpp"
#include "black_scholes_ref.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>

// ============================================================
//  main.cpp
//  Valide le moteur générique contre Black-Scholes analytique
//  (Call, Put), démontre un payoff path-dépendant (Asian) via le
//  MÊME moteur générique, puis benchmark template vs virtual.
// ============================================================

class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    double elapsedMs() const {
        return std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - start_).count();
    }
private:
    std::chrono::high_resolution_clock::time_point start_;
};

int main() {
    BlackScholesModel model{100.0, 0.03, 0.0, 0.20, 1.0};
    const int N_PATHS = 2'000'000;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "===================================================\n";
    std::cout << " Librairie de pricing generique (C++ templates)\n";
    std::cout << "===================================================\n\n";

    // --- 1) Validation : Call/Put via le moteur generique vs formule fermee ---
    auto call_engine = make_engine(model, CallPayoff{100.0}, N_PATHS);
    auto put_engine  = make_engine(model, PutPayoff{100.0}, N_PATHS);

    double call_mc = call_engine.price();
    double put_mc  = put_engine.price();
    double call_bs = BlackScholesRef::call(model.S0, 100.0, model.T, model.r, model.sigma, model.q);
    double put_bs  = BlackScholesRef::put(model.S0, 100.0, model.T, model.r, model.sigma, model.q);

    std::cout << "--- Validation vs Black-Scholes analytique ---\n";
    std::cout << "Call : MC(template)=" << call_mc << "  BS(analytique)=" << call_bs
              << "  ecart=" << std::abs(call_mc - call_bs) << "\n";
    std::cout << "Put  : MC(template)=" << put_mc  << "  BS(analytique)=" << put_bs
              << "  ecart=" << std::abs(put_mc - put_bs) << "\n\n";

    // --- 2) Payoff path-dependant (Asian), MEME moteur generique ---
    auto asian_engine = make_engine(model, AsianCallPayoff{100.0}, 200'000);
    double asian_price = asian_engine.price();
    std::cout << "--- Payoff path-dependant, meme moteur generique ---\n";
    std::cout << "Asian Call (moyenne arithmetique, 252 fixings) : " << asian_price
              << "  (attendu < prix Call europeen " << call_bs
              << ", car la moyenne reduit la variance de la trajectoire)\n";
    std::cout << "  -> " << (asian_price < call_bs ? "OK (coherent)" : "ATTENTION")
              << "\n\n";

    // --- 3) Benchmark : dispatch statique (template) vs dynamique (virtual) ---
    std::cout << "--- Benchmark : template (statique) vs virtual (dynamique) ---\n";

    Timer t1;
    double price_template = make_engine(model, CallPayoff{100.0}, N_PATHS).price();
    double time_template = t1.elapsedMs();

    CallPayoffVirtual virtual_payoff(100.0);
    Timer t2;
    double price_virt = price_virtual(model, virtual_payoff, N_PATHS);
    double time_virtual = t2.elapsedMs();

    std::cout << "Template (dispatch statique) : " << time_template << " ms  (prix=" << price_template << ")\n";
    std::cout << "Virtual  (dispatch dynamique) : " << time_virtual << " ms  (prix=" << price_virt << ")\n";
    std::cout << "Facteur : " << (time_virtual / time_template) << "x\n";

    return 0;
}
