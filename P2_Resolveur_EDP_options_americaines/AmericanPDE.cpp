#include "AmericanPDE.hpp"
#include <cmath>
#include <algorithm>

namespace AmericanPDE {

// ------------------------------------------------------------
// Résout le système tridiagonal  a_i*y_{i-1} + b_i*y_i + c_i*y_{i+1} = d_i
// pour i = 0..n-1 (indices locaux), avec un algorithme de Thomas
// (élimination avant / substitution arrière), en O(n).
//
// Si style == American, la substitution arrière applique la
// projection de Brennan-Schwartz : y_i = max(valeur continuation, payoff_i).
// C'est ce qui transforme un simple solveur linéaire en solveur
// de problème de complémentarité linéaire, sans boucle itérative.
// ------------------------------------------------------------
static void thomasSolve(const std::vector<double>& a,
                         const std::vector<double>& b,
                         const std::vector<double>& c,
                         std::vector<double> d,               // RHS, copié (modifié en place)
                         const std::vector<double>& payoff,    // contrainte d'exercice (ignorée si European)
                         ExerciseStyle style,
                         std::vector<double>& y)
{
    const int n = static_cast<int>(b.size());
    std::vector<double> cPrime(n);

    // --- Élimination avant ---
    cPrime[0] = c[0] / b[0];
    d[0] = d[0] / b[0];
    for (int i = 1; i < n; ++i) {
        const double m = b[i] - a[i] * cPrime[i - 1];
        cPrime[i] = c[i] / m;
        d[i] = (d[i] - a[i] * d[i - 1]) / m;
    }

    // --- Substitution arrière (+ projection Brennan-Schwartz) ---
    y[n - 1] = d[n - 1];
    if (style == ExerciseStyle::American)
        y[n - 1] = std::max(y[n - 1], payoff[n - 1]);

    for (int i = n - 2; i >= 0; --i) {
        y[i] = d[i] - cPrime[i] * y[i + 1];
        if (style == ExerciseStyle::American)
            y[i] = std::max(y[i], payoff[i]);   // <-- coeur de Brennan-Schwartz
    }
}

GridResult solve(const Option& opt, ExerciseStyle style, int N, int M, double SmaxMult)
{
    GridResult result;

    const double Smax = SmaxMult * opt.K;
    const double dS = Smax / N;
    const double dt = opt.T / M;
    result.dt = dt;

    // --- Grille spatiale et payoff terminal (condition à t = T) ---
    result.S.resize(N + 1);
    for (int i = 0; i <= N; ++i) result.S[i] = i * dS;

    std::vector<double> payoff(N + 1);
    const bool isCall = (opt.type == OptionType::Call);
    for (int i = 0; i <= N; ++i) {
        payoff[i] = isCall ? std::max(result.S[i] - opt.K, 0.0)
                            : std::max(opt.K - result.S[i], 0.0);
    }

    // On stocke toute la surface V[n][i] : mémoire négligeable pour
    // des grilles de quelques centaines de points, et ça permet
    // d'extraire la frontière d'exercice + le thêta par différence
    // de tranches temporelles sans repasser par une résolution.
    result.V.assign(M + 1, std::vector<double>(N + 1, 0.0));
    result.V[M] = payoff;   // condition terminale : V(S,T) = payoff(S)

    result.exerciseBoundary.assign(M + 1, std::numeric_limits<double>::quiet_NaN());

    // --- Coefficients du schéma de Crank-Nicolson (noeuds intérieurs i=1..N-1) ---
    // Discrétisation standard de l'EDP de Black-Scholes-Merton :
    //   dV/dt + 0.5*sigma^2*S^2*d2V/dS2 + (r-q)*S*dV/dS - r*V = 0
    const int n = N - 1;   // nombre d'inconnues intérieures
    std::vector<double> aL(n), bL(n), cL(n);   // matrice implicite (inconnue au temps n)
    std::vector<double> aE(n), bE(n), cE(n);   // matrice explicite (connue au temps n+1)

    for (int k = 0; k < n; ++k) {
        const int i = k + 1;   // indice global du noeud (1..N-1)
        const double sig2i2 = opt.sigma * opt.sigma * i * i;
        const double drift = (opt.r - opt.q) * i;

        const double alpha = 0.25 * dt * (sig2i2 - drift);
        const double beta  = -0.5 * dt * (sig2i2 + opt.r);
        const double gamma_ = 0.25 * dt * (sig2i2 + drift);

        // Membre implicite : (I - A/2) V^n = ...
        aL[k] = -alpha;
        bL[k] = 1.0 - beta;
        cL[k] = -gamma_;

        // Membre explicite : ... = (I + A/2) V^{n+1}
        aE[k] = alpha;
        bE[k] = 1.0 + beta;
        cE[k] = gamma_;
    }

    // --- Marche arrière en temps : de n=M (maturité) à n=0 (aujourd'hui) ---
    for (int nStep = M - 1; nStep >= 0; --nStep) {
        const double tauNew = opt.T - nStep * dt;   // temps résiduel à la tranche recherchée

        // Conditions aux limites au temps t_n (tranche inconnue qu'on résout)
        double V0, VN;
        if (isCall) {
            V0 = 0.0;
            VN = (style == ExerciseStyle::American)
                     ? (Smax - opt.K)
                     : (Smax * std::exp(-opt.q * tauNew) - opt.K * std::exp(-opt.r * tauNew));
        } else {
            V0 = (style == ExerciseStyle::American) ? opt.K : opt.K * std::exp(-opt.r * tauNew);
            VN = 0.0;
        }

        const std::vector<double>& Vnext = result.V[nStep + 1];

        // Construction du second membre : (I + A/2) V^{n+1}
        std::vector<double> rhs(n);
        for (int k = 0; k < n; ++k) {
            const int i = k + 1;
            rhs[k] = aE[k] * Vnext[i - 1] + bE[k] * Vnext[i] + cE[k] * Vnext[i + 1];
        }
        // Contribution des bords (connus) au système implicite
        rhs[0]     -= aL[0]     * V0;
        rhs[n - 1] -= cL[n - 1] * VN;

        std::vector<double> payoffInterior(payoff.begin() + 1, payoff.end() - 1);
        std::vector<double> yInterior(n);
        thomasSolve(aL, bL, cL, rhs, payoffInterior, style, yInterior);

        std::vector<double>& Vcur = result.V[nStep];
        Vcur[0] = V0;
        Vcur[N] = VN;
        for (int k = 0; k < n; ++k) Vcur[k + 1] = yInterior[k];

        // --- Frontière d'exercice anticipé (uniquement pertinent en américain) ---
        if (style == ExerciseStyle::American) {
            if (isCall) {
                // Zone d'exercice = S élevé : on cherche le plus petit S
                // où V touche le payoff (au-delà, exercice optimal).
                for (int i = N; i >= 1; --i) {
                    if (Vcur[i] - payoff[i] > 1e-6) {
                        result.exerciseBoundary[nStep] = result.S[i];
                        break;
                    }
                }
            } else {
                // Zone d'exercice = S faible : on cherche le plus grand S
                // où V touche le payoff.
                for (int i = 0; i <= N; ++i) {
                    if (Vcur[i] - payoff[i] > 1e-6) {
                        result.exerciseBoundary[nStep] = result.S[i];
                        break;
                    }
                }
            }
        }
    }

    // --- Extraction du prix et des Greeks au noeud S0 (interpolation linéaire) ---
    const double S0 = opt.S0;
    int idx = static_cast<int>(S0 / dS);
    idx = std::clamp(idx, 1, N - 2);   // marge pour les différences finies centrées
    const double w = (S0 - result.S[idx]) / dS;

    const auto& V0slice = result.V[0];
    const auto& V1slice = result.V[1];

    auto interp = [&](const std::vector<double>& slice, int i) {
        return slice[i] * (1.0 - w) + slice[i + 1] * w;
    };

    result.price = interp(V0slice, idx);

    // Delta / Gamma par différences finies centrées sur la grille (déjà "gratuites" : le maillage existe déjà)
    const double Vm = interp(V0slice, idx - 1);
    const double Vp = interp(V0slice, idx + 1);
    const double Vc = result.price;
    result.delta = (Vp - Vm) / (2.0 * dS);
    result.gamma = (Vp - 2.0 * Vc + Vm) / (dS * dS);

    // Theta : différence entre la tranche t=0 et t=dt, ramenée à une base "par jour"
    const double priceNextSlice = interp(V1slice, idx);
    result.theta = (priceNextSlice - result.price) / dt / 365.0;

    return result;
}

} // namespace AmericanPDE
