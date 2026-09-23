"""
live_vol_surface.py  —  Surface de volatilité live : SVI (par échéance) + SSVI (globale)
=========================================================================================
Script unique, données options Yahoo Finance (yfinance), recalcul de l'IV à partir des prix (mid bid/ask), 
calibration SVI par échéance,
calibration SSVI globale sous contraintes de non-arbitrage, affichage 3D rafraîchi.

Usage :
    python live_vol_surface.py                    # SPY, rafraîchi toutes les 30 s
    python live_vol_surface.py --symbol AAPL --interval 60
    python live_vol_surface.py --demo             # hors-ligne, données synthétiques
    python live_vol_surface.py --report           # rapport statique (PNG + CSV) + Dupire + Monte Carlo
                                                  # (Yahoo, ou synthétique si le réseau échoue)

Les dépendances manquantes (numpy, scipy, matplotlib, yfinance) s'installent seules.
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import time
from dataclasses import dataclass


def _ensure(*pkgs):
    for p in pkgs:
        try:
            __import__(p)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])


_ensure("numpy", "scipy", "matplotlib")

import matplotlib  # noqa: E402

if "--report" in sys.argv:
    matplotlib.use("Agg")  # export PNG sans fenêtre
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.widgets import Button  # noqa: E402
from scipy.optimize import brentq, least_squares  # noqa: E402
from scipy.special import erf  # noqa: E402

plt.style.use("dark_background")
YEAR = 365.25 * 86400.0

# ---------------------------------------------------------------------------
# 1. Black-76 + volatilité implicite
# ---------------------------------------------------------------------------


def _N(x):
    return 0.5 * (1.0 + erf(x / 2 ** 0.5))


def black_price(F, K, T, r, sig, is_call):
    st = sig * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * st ** 2) / st
    d2 = d1 - st
    df = np.exp(-r * T)
    return df * (F * _N(d1) - K * _N(d2)) if is_call else df * (K * _N(-d2) - F * _N(-d1))


def implied_vol(price, F, K, T, r, is_call):
    df = np.exp(-r * T)
    intrinsic = df * max(F - K if is_call else K - F, 0.0)
    if price <= intrinsic + 1e-8:
        return np.nan
    try:
        return brentq(lambda s: black_price(F, K, T, r, s, is_call) - price, 1e-3, 4.0, xtol=1e-7)
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# 2. SVI raw par échéance  (Gatheral 2004)
# ---------------------------------------------------------------------------


@dataclass
class SVI:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def w(self, k):
        km = np.asarray(k, float) - self.m
        return np.maximum(self.a + self.b * (self.rho * km + np.sqrt(km ** 2 + self.sigma ** 2)), 0.0)

    def iv(self, k, T):
        return np.sqrt(np.maximum(self.w(k), 1e-12) / max(T, 1e-6))


def fit_svi(k, T, iv):
    lo = np.array([-1.0, 1e-5, -0.999, -1.0, 1e-3])
    hi = np.array([1.0, 5.0, 0.999, 1.0, 2.0])
    x0 = np.clip([np.min(iv ** 2 * T) * 0.9, 0.1, -0.3, 0.0, 0.1], lo, hi)
    res = least_squares(lambda x: SVI(*x).iv(k, T) - iv, x0, bounds=(lo, hi),
                        loss="soft_l1", f_scale=0.02, max_nfev=500)
    return SVI(*res.x)


# ---------------------------------------------------------------------------
# 3. SSVI  (Gatheral-Jacquier 2014)  phi(theta) = eta / (theta^g (1+theta)^(1-g))
#    theta(T) = variance totale ATM, interpolée sur les échéances observées
# ---------------------------------------------------------------------------


class SSVI:
    def __init__(self, Ts, thetas, rho, eta, gamma):
        self.Ts, self.thetas = np.asarray(Ts, float), np.asarray(thetas, float)
        self.rho, self.eta, self.gamma = rho, eta, gamma

    def theta(self, T):
        T = np.asarray(T, float)
        th = np.interp(T, np.r_[0.0, self.Ts], np.r_[0.0, self.thetas])
        return np.where(T > self.Ts[-1], self.thetas[-1] * T / self.Ts[-1], th)

    def phi(self, th):
        return self.eta / (th ** self.gamma * (1.0 + th) ** (1.0 - self.gamma))

    def w(self, k, T):
        th = np.maximum(self.theta(T), 1e-10)
        pk = self.phi(th) * np.asarray(k, float)
        return 0.5 * th * (1.0 + self.rho * pk + np.sqrt((pk + self.rho) ** 2 + 1.0 - self.rho ** 2))

    def iv(self, k, T):
        return np.sqrt(np.maximum(self.w(k, T), 1e-12) / np.maximum(T, 1e-6))

    def arbitrage_check(self):
        """Butterfly : th*phi*(1+|rho|) < 4 et th*phi^2*(1+|rho|) <= 4 (Thm 4.2 G&J).
        Calendar : theta(T) croissante (garantie par construction)."""
        ph, c = self.phi(self.thetas), 1.0 + abs(self.rho)
        v1, v2 = float((self.thetas * ph * c).max()), float((self.thetas * ph ** 2 * c).max())
        return v1 < 4.0 and v2 <= 4.0, v1, v2


def fit_ssvi(slices):
    Ts = np.array([s["T"] for s in slices])
    th = []
    for s in slices:  # variance totale ATM : médiane des 5 points les plus proches de k=0
        near = np.argsort(np.abs(s["k"]))[:5]
        th.append(np.median(s["iv"][near]) ** 2 * s["T"])
    thetas = np.maximum.accumulate(np.array(th))  # calendar spread : theta croissante

    K = np.concatenate([s["k"] for s in slices])
    T = np.concatenate([np.full(len(s["k"]), s["T"]) for s in slices])
    IV = np.concatenate([s["iv"] for s in slices])

    def res(x):
        m = SSVI(Ts, thetas, *x)
        ph, c = m.phi(thetas), 1.0 + abs(x[0])
        pen = np.r_[np.maximum(0, thetas * ph * c / 3.9 - 1), np.maximum(0, thetas * ph ** 2 * c / 3.9 - 1)]
        return np.r_[m.iv(K, T) - IV, 100.0 * pen]  # pénalité de non-arbitrage butterfly

    best = None
    for x0 in ([-0.5, 0.5, 0.5], [-0.3, 1.5, 0.3], [-0.7, 0.3, 0.7]):
        r = least_squares(res, x0, bounds=([-0.999, 1e-3, 0.01], [0.999, 10.0, 0.99]),
                          loss="soft_l1", f_scale=0.02, max_nfev=500)
        if best is None or r.cost < best.cost:
            best = r
    return SSVI(Ts, thetas, *best.x)


# ---------------------------------------------------------------------------
# 4. Sources de données : Yahoo Finance (live) ou synthétique (--demo)
# ---------------------------------------------------------------------------


def fetch_yahoo(symbol, r, q, band=0.20, targets_days=(7, 14, 30, 60, 90, 180, 270, 365)):
    _ensure("yfinance")
    import yfinance as yf

    tk = yf.Ticker(symbol)
    spot = None
    try:
        spot = float(tk.fast_info["last_price"])
    except Exception:
        pass
    if not spot or not np.isfinite(spot):
        spot = float(tk.history(period="5d")["Close"].dropna().iloc[-1])

    now = dt.datetime.now(dt.timezone.utc)
    exps = []
    for e in tk.options:  # clôture ~16h New York ≈ 20h UTC
        T = (dt.datetime.strptime(e, "%Y-%m-%d").replace(hour=20, tzinfo=dt.timezone.utc) - now).total_seconds() / YEAR
        if T >= 5 / 365.25:  # on écarte le 0DTE / très court terme, IV Yahoo inexploitable
            exps.append((e, T))
    chosen = []
    for d in targets_days:  # échéances réparties : ~1 sem, 2 sem, 1 m, 2 m, 3 m, 6 m, 9 m, 1 an
        if exps:
            b = min(exps, key=lambda x: abs(x[1] * 365.25 - d))
            if b not in chosen:
                chosen.append(b)
    chosen.sort(key=lambda x: x[1])

    slices = []
    for e, T in chosen:
        ch = tk.option_chain(e)
        F = spot * np.exp((r - q) * T)
        ks, ivs = [], []
        for table, is_call in ((ch.calls, True), (ch.puts, False)):
            for row in table.itertuples():
                K = float(row.strike)
                if is_call != (K >= F) or abs(np.log(K / F)) > band:  # options OTM uniquement
                    continue
                bid, ask, last = row.bid, row.ask, row.lastPrice
                if bid > 0 and ask >= bid:
                    px = 0.5 * (bid + ask)
                    if ask - bid > 0.5 * px:  # spread trop large
                        continue
                else:
                    px = last
                if not px or px != px or px <= 0:
                    continue
                iv = implied_vol(px, F, K, T, r, is_call)
                if np.isfinite(iv) and 0.03 < iv < 2.0:
                    ks.append(np.log(K / F))
                    ivs.append(iv)
        if len(ks) >= 6:
            o = np.argsort(ks)
            slices.append(dict(expiry=e, T=T, k=np.array(ks)[o], iv=np.array(ivs)[o]))
    if len(slices) < 3:
        raise RuntimeError("Pas assez de données options valides (marché fermé / symbole sans options ?).")
    return spot, slices


def fetch_demo(symbol="DEMO", r=0.03, q=0.01, noise_bps=25.0):
    """Chaîne synthétique : SVI vrai -> prix Black bruités -> IV recalculée (comme un vrai pipeline)."""
    rng = np.random.default_rng()
    S0, slices = 100.0, []
    for T in (1 / 12, 3 / 12, 6 / 12, 1.0, 2.0):
        F = S0 * np.exp((r - q) * T)
        k = np.linspace(-0.4, 0.4, 15)
        true = SVI(0.015 + 0.02 * T, 0.12 + 0.05 * T, -0.45, 0.0, 0.10 + 0.05 * np.sqrt(T))
        iv_true = true.iv(k, T)
        ks, ivs = [], []
        for kk, sg in zip(k, iv_true):
            is_call = kk >= 0
            px = black_price(F, F * np.exp(kk), T, r, sg, is_call) * (1 + rng.normal(0, noise_bps * 1e-4))
            iv = implied_vol(px, F, F * np.exp(kk), T, r, is_call)
            if np.isfinite(iv):
                ks.append(kk)
                ivs.append(iv)
        slices.append(dict(expiry=f"T={T:.2f}a", T=T, k=np.array(ks), iv=np.array(ivs)))
    return S0, slices


# ---------------------------------------------------------------------------
# 5. Affichage live
# ---------------------------------------------------------------------------


def redraw(ax3, axs, spot, slices, ssvi, svis, idx, symbol):
    elev, azim = ax3.elev, ax3.azim
    ax3.clear()
    ax3.set_facecolor("#0b0d0f")
    kmax = max(np.abs(s["k"]).max() for s in slices)
    KK, TT = np.meshgrid(np.linspace(-kmax, kmax, 41), np.linspace(slices[0]["T"], slices[-1]["T"], 25))
    ax3.plot_surface(KK, TT, ssvi.iv(KK, TT), cmap="magma", alpha=0.55, linewidth=0)
    ax3.plot_wireframe(KK, TT, ssvi.iv(KK, TT), color="#00f2ff", linewidth=0.4, alpha=0.6)
    for s, sv in zip(slices, svis):
        ax3.scatter(s["k"], np.full(s["k"].size, s["T"]), s["iv"], c="#ff9f1c", s=6, depthshade=False)
        kk = np.linspace(-kmax, kmax, 60)
        ax3.plot(kk, np.full(kk.size, s["T"]), sv.iv(kk, s["T"]), color="#00ff88", lw=1)
    ax3.set_zlim(0, min(1.0, max(s["iv"].max() for s in slices) * 1.15))
    ax3.set_xlabel("log-moneyness ln(K/F)")
    ax3.set_ylabel("T (années)")
    ax3.set_zlabel("Vol. implicite")
    ok, v1, v2 = ssvi.arbitrage_check()
    ax3.set_title(f"{symbol} spot={spot:.2f} | SSVI rho={ssvi.rho:.2f} eta={ssvi.eta:.2f} g={ssvi.gamma:.2f} | "
                  f"{'sans arbitrage (butterfly)' if ok else 'ATTENTION arbitrage'} | {time.strftime('%H:%M:%S')}",
                  fontsize=9)
    ax3.view_init(elev=elev, azim=azim)

    s, sv = slices[idx], svis[idx]
    kk = np.linspace(s["k"].min(), s["k"].max(), 100)
    axs.clear()
    axs.set_facecolor("#161b22")
    axs.scatter(s["k"], s["iv"], c="#ff9f1c", s=10, label="Marché (IV recalculée)")
    axs.plot(kk, sv.iv(kk, s["T"]), color="#00ff88", label="SVI (échéance)")
    axs.plot(kk, ssvi.iv(kk, s["T"]), color="#00f2ff", ls="--", label="SSVI (globale)")
    axs.axvline(0, color="#ff3e3e", ls=":", label="Forward")
    axs.set_title(f"Smile {s['expiry']}  (T={s['T']:.3f} a)", fontsize=10)
    axs.set_xlabel("ln(K/F)")
    axs.legend(fontsize=8)


# ---------------------------------------------------------------------------
# 6. Vol locale de Dupire (sur la SSVI) + Monte Carlo (Euler, variates antithétiques)
# ---------------------------------------------------------------------------


def dupire_local_vol(k, t, m, dk=1e-3, dT=1e-4):
    """sigma_loc^2 = dw/dT / [1 - k/w w_k + 1/4(-1/4 - 1/w + k^2/w^2) w_k^2 + 1/2 w_kk]  (Gatheral)
    avec k = ln(S/F(t)) et T = t (temps calendaire, pas le temps restant)."""
    t = max(t, 1e-3)
    w0, wp, wm = m.w(k, t), m.w(k + dk, t), m.w(k - dk, t)
    wk, wkk = (wp - wm) / (2 * dk), (wp - 2 * w0 + wm) / dk ** 2
    wT = (m.w(k, t + dT) - m.w(k, t - dT)) / (2 * dT)
    w0 = np.maximum(w0, 1e-10)
    den = 1 - k / w0 * wk + 0.25 * (-0.25 - 1 / w0 + k ** 2 / w0 ** 2) * wk ** 2 + 0.5 * wkk
    return np.clip(np.sqrt(np.maximum(wT / np.maximum(den, 1e-8), 1e-8)), 0.02, 1.5)


def simulate_paths(S0, r, q, T, n_steps, n_sims, m, seed=42):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Zh = rng.normal(size=(n_sims // 2, n_steps))
    Z = np.vstack([Zh, -Zh])
    S = np.full(Z.shape[0], float(S0))
    paths = np.empty((Z.shape[0], n_steps + 1))
    paths[:, 0] = S
    for i in range(n_steps):
        t = i * dt
        vol = dupire_local_vol(np.log(S / (S0 * np.exp((r - q) * t))), t, m)
        S = S * np.exp((r - q - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * Z[:, i])
        paths[:, i + 1] = S
    return paths


def mc_call(paths, K, r, T, asian=False):
    pay = np.maximum((paths[:, 1:].mean(axis=1) if asian else paths[:, -1]) - K, 0.0)
    df = np.exp(-r * T)
    return df * pay.mean(), df * pay.std() / np.sqrt(len(pay))


# ---------------------------------------------------------------------------
# 7. Rapport statique : calibration, non-arbitrage, PNG, CSV, Dupire + MC
# ---------------------------------------------------------------------------


def report(spot, slices, r, q, outdir, T_mc, n_sims):
    import csv
    import os

    os.makedirs(outdir, exist_ok=True)
    svis = [fit_svi(s["k"], s["T"], s["iv"]) for s in slices]
    ssvi = fit_ssvi(slices)
    print(f"\n[report] Spot={spot:.2f} | {len(slices)} échéances | {sum(len(s['k']) for s in slices)} cotations")
    print("[report] Calibration SVI par maturité :")
    rows = []
    for s, p in zip(slices, svis):
        e_svi = np.sqrt(np.mean((p.iv(s["k"], s["T"]) - s["iv"]) ** 2)) * 100
        e_ssvi = np.sqrt(np.mean((ssvi.iv(s["k"], s["T"]) - s["iv"]) ** 2)) * 100
        wmin = p.a + p.b * p.sigma * np.sqrt(1 - p.rho ** 2)
        print(f"  T={s['T']:5.3f} a={p.a:+.4f} b={p.b:.4f} rho={p.rho:+.3f} m={p.m:+.3f} sigma={p.sigma:.4f} "
              f"| RMSE SVI={e_svi:.3f}% SSVI={e_ssvi:.3f}% | butterfly(wmin>=0): {wmin >= -1e-8}")
        rows.append(dict(expiry=s["expiry"], T=s["T"], a=p.a, b=p.b, rho=p.rho, m=p.m, sigma=p.sigma,
                         min_total_variance=wmin, rmse_svi_pct=e_svi, rmse_ssvi_pct=e_ssvi))

    lo, hi = max(s["k"].min() for s in slices), min(s["k"].max() for s in slices)
    kg = np.linspace(lo, hi, 200)
    cal_svi = bool(np.all(np.diff(np.array([p.w(kg) for p in svis]), axis=0) >= -1e-8))
    Tg = np.linspace(slices[0]["T"], slices[-1]["T"], 60)
    cal_ssvi = bool(np.all(np.diff(ssvi.w(kg[None, :], Tg[:, None]), axis=0) >= -1e-8))
    ok, v1, v2 = ssvi.arbitrage_check()
    print(f"[report] Calendar (w croissante en T) : SVI par tranche={cal_svi} | SSVI={cal_ssvi}")
    print(f"[report] SSVI rho={ssvi.rho:+.3f} eta={ssvi.eta:.3f} gamma={ssvi.gamma:.3f} | butterfly "
          f"th*phi*(1+|rho|)={v1:.3f}<4, th*phi^2*(1+|rho|)={v2:.3f}<=4 -> {ok}")

    with open(f"{outdir}/svi_params.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with open(f"{outdir}/ssvi_params.csv", "w", newline="") as f:
        f.write("rho,eta,gamma,butterfly_ok\n" + f"{ssvi.rho},{ssvi.eta},{ssvi.gamma},{ok}\n")

    n = len(slices)
    nc = min(3, n)
    nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(5 * nc, 4 * nr), squeeze=False)
    for i, (s, p) in enumerate(zip(slices, svis)):
        ax = axs[i // nc][i % nc]
        kk = np.linspace(s["k"].min(), s["k"].max(), 200)
        ax.scatter(s["k"], s["iv"] * 100, s=14, c="#ff9f1c", label="Marché")
        ax.plot(kk, p.iv(kk, s["T"]) * 100, c="#00ff88", label="SVI")
        ax.plot(kk, ssvi.iv(kk, s["T"]) * 100, c="#00f2ff", ls="--", label="SSVI")
        ax.set_title(f"{s['expiry']}  T={s['T']:.3f}")
        ax.set_xlabel("ln(K/F)")
        ax.set_ylabel("IV (%)")
        ax.legend(fontsize=7)
    for i in range(n, nr * nc):
        axs[i // nc][i % nc].axis("off")
    fig.tight_layout()
    fig.savefig(f"{outdir}/smile_by_maturity.png", dpi=130)
    plt.close(fig)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    KK, TT = np.meshgrid(np.linspace(-max(abs(lo), hi), max(abs(lo), hi), 60), Tg)
    ax.plot_surface(KK, TT, ssvi.iv(KK, TT) * 100, cmap="magma", alpha=0.8, linewidth=0)
    for s in slices:
        ax.scatter(s["k"], np.full(s["k"].size, s["T"]), s["iv"] * 100, c="#ff9f1c", s=5)
    ax.set_xlabel("ln(K/F)")
    ax.set_ylabel("T (années)")
    ax.set_zlabel("IV (%)")
    ax.set_title("Surface SSVI calibrée + cotations")
    fig.savefig(f"{outdir}/vol_surface_3d.png", dpi=130)
    plt.close(fig)
    print(f"[report] Exports -> {outdir}/ (smile_by_maturity.png, vol_surface_3d.png, svi_params.csv, ssvi_params.csv)")

    T_mc = min(T_mc, slices[-1]["T"])
    n_steps = max(50, int(T_mc * 252))
    paths = simulate_paths(spot, r, q, T_mc, n_steps, n_sims, ssvi)
    F = spot * np.exp((r - q) * T_mc)
    print(f"\n[dupire/MC] T={T_mc:.2f}a, {n_steps} pas, {paths.shape[0]} trajectoires (antithétiques)")
    print("  Vol locale de Dupire (t=T/2) pour k=-0.1, 0, +0.1 :",
          np.round(dupire_local_vol(np.array([-0.1, 0.0, 0.1]), T_mc / 2, ssvi), 4))
    for mny in (0.9, 1.0, 1.1):
        K = mny * F
        pc, se = mc_call(paths, K, r, T_mc)
        ref = black_price(F, K, T_mc, r, float(ssvi.iv(np.log(K / F), T_mc)), True)
        print(f"  Call européen K={mny:.1f}F : MC={pc:.4f} ± {se:.4f} | Black(IV SSVI)={ref:.4f}  (doit coller)")
    pa, sa = mc_call(paths, F, r, T_mc, asian=True)
    print(f"  Call asiatique K=F : MC={pa:.4f} ± {sa:.4f}")


def main():
    ap = argparse.ArgumentParser(description="Surface de vol live SVI + SSVI (Yahoo Finance)")
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--interval", type=float, default=30.0, help="secondes entre deux rafraîchissements")
    ap.add_argument("--rate", type=float, default=0.04, help="taux sans risque annuel")
    ap.add_argument("--div", type=float, default=0.0, help="rendement du dividende annuel")
    ap.add_argument("--demo", action="store_true", help="données synthétiques, hors-ligne")
    ap.add_argument("--report", action="store_true", help="rapport statique (PNG/CSV) + Dupire + Monte Carlo")
    ap.add_argument("--outdir", default="outputs", help="dossier d'export du rapport")
    ap.add_argument("--T-mc", type=float, default=0.5, help="maturité du Monte Carlo (années)")
    ap.add_argument("--sims", type=int, default=20000, help="nombre de trajectoires Monte Carlo")
    a = ap.parse_args()
    fetch = fetch_demo if a.demo else fetch_yahoo

    if a.report:
        try:
            spot, slices = fetch(a.symbol, a.rate, a.div)
        except Exception as exc:
            print(f"[report] Données réelles indisponibles ({exc}) -> bascule synthétique.")
            spot, slices = fetch_demo("DEMO", a.rate, a.div)
        report(spot, slices, a.rate, a.div, a.outdir, a.T_mc, a.sims)
        return

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("#0b0d0f")
    ax3 = plt.subplot2grid((1, 3), (0, 0), colspan=2, projection="3d")
    axs = plt.subplot2grid((1, 3), (0, 2))
    st = {"lock": False, "idx": 0, "data": None}

    b1 = Button(plt.axes([0.30, 0.03, 0.12, 0.04]), "LOCK UPDATES", color="#1f2329", hovercolor="#2d333b")
    b2 = Button(plt.axes([0.45, 0.03, 0.12, 0.04]), "NEXT EXPIRY", color="#1f2329", hovercolor="#2d333b")

    def toggle(_):
        st["lock"] = not st["lock"]
        b1.label.set_text("UNLOCK UPDATES" if st["lock"] else "LOCK UPDATES")

    def nxt(_):
        st["idx"] += 1
        draw()

    def draw():
        if st["data"]:
            spot, slices, ssvi, svis = st["data"]
            redraw(ax3, axs, spot, slices, ssvi, svis, st["idx"] % len(slices), a.symbol)
            fig.canvas.draw_idle()

    b1.on_clicked(toggle)
    b2.on_clicked(nxt)

    plt.ion()
    plt.show(block=False)
    last = 0.0
    print(f"[live] {a.symbol} — Ctrl+C ou fermer la fenêtre pour arrêter.")
    try:
        while plt.fignum_exists(fig.number):
            if not st["lock"] and time.time() - last >= a.interval:
                last = time.time()
                try:
                    spot, slices = fetch(a.symbol, a.rate, a.div)
                    svis = [fit_svi(s["k"], s["T"], s["iv"]) for s in slices]
                    st["data"] = (spot, slices, fit_ssvi(slices), svis)
                    draw()
                except Exception as exc:
                    print(f"[live] Rafraîchissement ignoré : {exc}")
            plt.pause(0.2)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
