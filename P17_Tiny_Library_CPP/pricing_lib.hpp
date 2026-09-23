#pragma once
#include <cmath>
#include <random>
#include <vector>
#include <numeric>

// ============================================================
//  pricing_lib.hpp
//  ================
//  Librairie de pricing "header-only" utilisant les templates C++
//  pour resoudre le payoff et le modele A LA COMPILATION, plutot
//  qu'a l'execution (polymorphisme dynamique / virtual functions).
//
//  Interet : MonteCarloEngine<Model, Payoff> genere une version
//  SPECIALISEE et entierement inlinee du moteur pour chaque
//  combinaison (Model, Payoff) utilisee dans le programme -- le
//  compilateur voit le code complet du payoff a l'interieur de la
//  boucle de simulation et peut l'optimiser (inlining, vectorisation)
//  comme s'il avait ete ecrit a la main pour ce cas precis. Avec des
//  `virtual`, chaque appel de payoff() passe par une indirection de
//  table de fonctions virtuelles (vtable) que le compilateur ne peut
//  pas voir a travers.
//
//  C'est le meme principe que la Standard Template Library ou Eigen :
//  "zero-cost abstraction" -- le code generique ne coute rien de plus
//  qu'une version ecrite a la main, une fois compile.
// ============================================================

// --- Payoffs : chaque payoff est un TYPE distinct, pas une valeur ---
// Le compilateur genere un moteur different pour chaque payoff.
struct CallPayoff {
    double strike;
    double operator()(double S) const { return std::max(S - strike, 0.0); }
};

struct PutPayoff {
    double strike;
    double operator()(double S) const { return std::max(strike - S, 0.0); }
};

struct DigitalCallPayoff {
    double strike;
    double cash;
    double operator()(double S) const { return S > strike ? cash : 0.0; }
};

// --- Payoff path-dependant : recoit tout le chemin, pas juste S_T ---
struct AsianCallPayoff {
    double strike;
    double operator()(const std::vector<double>& path) const {
        double avg = std::accumulate(path.begin(), path.end(), 0.0) / path.size();
        return std::max(avg - strike, 0.0);
    }
};

// --- Modele : Black-Scholes (GBM), parametrise a la construction ---
struct BlackScholesModel {
    double S0, r, q, sigma, T;

    // Genere UNE trajectoire complete (n_steps points), pour les payoffs path-dependants.
    template <typename RNG>
    std::vector<double> simulate_path(int n_steps, RNG& rng) const {
        std::normal_distribution<double> normal(0.0, 1.0);
        double dt = T / n_steps;
        double drift = (r - q - 0.5 * sigma * sigma) * dt;
        double diffusion = sigma * std::sqrt(dt);

        std::vector<double> path(n_steps + 1);
        path[0] = S0;
        for (int i = 1; i <= n_steps; ++i) {
            path[i] = path[i - 1] * std::exp(drift + diffusion * normal(rng));
        }
        return path;
    }

    // Terminal seul (plus rapide quand le payoff ne depend que de S_T)
    template <typename RNG>
    double simulate_terminal(RNG& rng) const {
        std::normal_distribution<double> normal(0.0, 1.0);
        double drift = (r - q - 0.5 * sigma * sigma) * T;
        double diffusion = sigma * std::sqrt(T);
        return S0 * std::exp(drift + diffusion * normal(rng));
    }

    double discount() const { return std::exp(-r * T); }
};

// --- Moteur Monte Carlo generique : Model x Payoff resolus a la compilation ---
// On utilise `if constexpr` pour distinguer payoff(double) [terminal]
// de payoff(vector<double>) [path-dependant], sans aucun cout a l'execution.
template <typename Model, typename Payoff>
class MonteCarloEngine {
public:
    MonteCarloEngine(Model model, Payoff payoff, int n_paths, unsigned seed = 42)
        : model_(model), payoff_(payoff), n_paths_(n_paths), seed_(seed) {}

    double price() const {
        std::mt19937_64 rng(seed_);
        double sum = 0.0;

        if constexpr (std::is_invocable_v<Payoff, double>) {
            // Payoff terminal (Call, Put, Digital...) : plus rapide, pas de stockage de chemin.
            for (int i = 0; i < n_paths_; ++i) {
                double ST = model_.simulate_terminal(rng);
                sum += payoff_(ST);
            }
        } else {
            // Payoff path-dependant (Asian...) : simule le chemin complet.
            for (int i = 0; i < n_paths_; ++i) {
                auto path = model_.simulate_path(252, rng);
                sum += payoff_(path);
            }
        }

        return model_.discount() * sum / n_paths_;
    }

private:
    Model model_;
    Payoff payoff_;
    int n_paths_;
    unsigned seed_;
};

// Fonction "usine" pour deduire les types automatiquement (evite d'ecrire
// MonteCarloEngine<BlackScholesModel, CallPayoff> explicitement a l'appel).
template <typename Model, typename Payoff>
MonteCarloEngine<Model, Payoff> make_engine(Model model, Payoff payoff, int n_paths, unsigned seed = 42) {
    return MonteCarloEngine<Model, Payoff>(model, payoff, n_paths, seed);
}

// ============================================================
//  Version a dispatch dynamique (virtual), pour le benchmark de
//  comparaison contre l'approche template. Meme logique de calcul,
//  interface polymorphique classique (comme le ferait une librairie
//  orientee "plugin", ou un debutant en C++).
// ============================================================
struct IPayoffVirtual {
    virtual double operator()(double S) const = 0;
    virtual ~IPayoffVirtual() = default;
};

struct CallPayoffVirtual : IPayoffVirtual {
    double strike;
    explicit CallPayoffVirtual(double k) : strike(k) {}
    double operator()(double S) const override { return std::max(S - strike, 0.0); }
};

double price_virtual(const BlackScholesModel& model, const IPayoffVirtual& payoff, int n_paths, unsigned seed = 42) {
    std::mt19937_64 rng(seed);
    double sum = 0.0;
    for (int i = 0; i < n_paths; ++i) {
        double ST = model.simulate_terminal(rng);
        sum += payoff(ST);   // appel via vtable : indirection que le compilateur ne peut pas inliner
    }
    return model.discount() * sum / n_paths;
}
