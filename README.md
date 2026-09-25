HERE:
You will find around 30 projects here. 

Most are written in Python, which is where I started due to its versatility. 

To better understand memory management and code interpretation, and because speed can be a major edge in quantitative finance,
i also built projects in C++ and Rust. 

Additionally, I explored R for data science applications.

As I learned quickly, and being early in my journey, 
I wanted to optimize these projects and make the code cleaner, faster, and more readable. 

I used GitHub Copilot to refine certain parts, though it was mainly used for debugging within Visual Studio Code.

## PROJECTS

| # | Project | Language(s) | Focus |
|---|---|---|---|
| P0 | Lead-lag matrix (Hayashi-Yoshida) | Python, Rust | Async tick correlation, ~280x speedup (Rust vs Python) |
| P1 | Pricing engine BS / Monte Carlo / Greeks | C++ (OpenMP) | Closed-form BSM vs parallel MC, full Greeks on both |
| P2 | PDE solver, American options | C++ | Crank-Nicolson finite differences + early exercise (Brennan-Schwartz) |
| P3 | Vol surface calibration (SVI / SSVI) | Python | Implied vol → SVI smile per maturity → no-arbitrage check → 3D surface |
| P4 | Realized volatility (TSRV) | Rust | Two-Scale RV vs naive RV, robust to microstructure noise |
| P5 | CVA/DVA calculator (Hull-White) | Python | MC exposure simulation, one-factor short-rate model |
| P6 | Markowitz & Black-Litterman | Python, R | Python efficient frontier feeds R's Bayesian Black-Litterman |
| P7 | Econometrics (GARCH, cointegration, VAR) | R | GARCH MLE, ADF, VAR — all hand-coded, no CRAN packages |
| P8 | Vol regime prediction (ML & SHAP) | Python | Calm/stress classifier + SHAP interpretability |
| P9 | Denoising (RMT, Marchenko-Pastur) | Python | Spectral denoising, validated via min-variance portfolio (-28% variance) |
| P10 | VaR / Expected Shortfall + Kupiec test | Python | Gaussian/historical/MC VaR, rolling out-of-sample backtest + Kupiec POF |
| P11 | Event-driven backtesting engine | Rust, Python | True FIFO event queue, momentum vs mean-reversion under regime change |
| P12 | Autocall pricer | VBA, Python | Autocall MC pricer (Python) + client-facing Excel/VBA sheet |
| P13 | Multi-agent market (Cont-Bouchaud percolation) | Python | Stylized facts (fat tails, vol clustering) from agent percolation, no rationality assumption |
| P14 | Tick-by-tick feature pipeline | Rust, Python | Streaming microprice/OFI/RV features, O(1)/tick, tested to 5M ticks |
| P15 | Bond toolkit (duration, convexity) | VBA, Python | Macaulay/modified duration, convexity — Python validates VBA formulas |
| P16 | Brownian reflection principle | Python | MC validation of max-process and first-passage-time laws |
| P17 | Tiny library | C++ | Compile-time generic MC engine (templates) vs virtual dispatch |
| P18 | Q-learning trading agent | Python | Tabular Q-learning, honest out-of-sample eval (separate seeds) |
| P19 | Factor research (Fama-MacBeth) | Python | Two-step regressions, Newey-West SEs, placebo factor as control |
| P20 | Event study (PEAD) | Python | MacKinlay event-study methodology on post-earnings drift |
| P21 | Information Coefficient decay | Python | How long a predictive signal retains forecasting power |
| P22 | Pairs trading (mean reversion) | Python | Universe-wide cointegration scan + risk-managed backtest (stop-loss, vol sizing) |
| P23 | Market making, inventory (Avellaneda-Stoikov) | Python | Inventory-aware quoting vs naive fixed spread, matched MC sessions |
| P24 | P&L attribution, options market maker | Python | Delta-hedged option P&L split into Delta/Gamma/Theta/Vega |
| P25 | Pricing/hedging via path signatures | Python | Rough-vol hedging via linear regression on signatures, hand-coded |
| P26 | Rough volatility | Python | Rough Heston calibration via fractional Riccati equation |
| P27 | Deep hedging | Python | Neural hedging strategy under transaction costs (MLP + Adam from scratch), entropic risk 44% lower than BS delta |
| P28 | Optimal execution, market impact | Python | Optimal execution under transient impact, QP solved via KKT |
| P29 | Order book modeling (Hawkes) | Python | Order book via Hawkes processes, hand-coded simulation + MLE |
| P30 | Live vol surface | Python | Live Yahoo Finance feed → SVI/SSVI surface + Dupire/MC pricing |

ABOUT ME:
My interest in quantitative finance began in early 2026 (around late January / early February).

At the time of writing, I am nearing the end of my Bachelor's degree in Economics. 

Prior to this, I had a background in advanced mathematics through a Classe Préparatoire aux Grandes Écoles (CPGE).

Feeling uncertain about pursuing a career in traditional engineering, as I was not yet aware of financial engineering, 
despite loving mathematics and related subjects—I decided to switch to economics.

During my first year and the start of my second year, I felt somewhat lost regarding my future. 

I also realized that the mathematics curriculum in my economics program lacked the depth I was looking for. 

To bridge this gap, I began self-studying mathematics, physics, economics, finance, and computer science.

This journey led me to discover quantitative finance—a field where every subject deeply resonates with me. 

Since the start of the year, I have been continuously reading, practicing, and building projects.

This GitHub repository is a testament to that dedication.
