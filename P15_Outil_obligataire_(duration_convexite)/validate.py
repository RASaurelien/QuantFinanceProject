"""
validate.py
===========
Reimplemente en Python les memes formules que BondPricer.bas, 
pour valider le calcul contre des cas connus en forme fermee,
donc cette validation porte sur les FORMULES (identiques ligne a ligne
dans les deux langages), pas sur l'execution VBA elle-meme. A retester
directement dans Excel avant usage.
"""


def bond_price(face, coupon_rate, ytm, freq, maturity):
    n = round(maturity * freq)
    c = face * coupon_rate / freq
    r = ytm / freq
    pv = sum(c / (1 + r) ** i for i in range(1, n + 1))
    pv += face / (1 + r) ** n
    return pv


def macaulay_duration(face, coupon_rate, ytm, freq, maturity):
    n = round(maturity * freq)
    c = face * coupon_rate / freq
    r = ytm / freq
    price = bond_price(face, coupon_rate, ytm, freq, maturity)
    ws = 0.0
    for i in range(1, n + 1):
        t = i / freq
        cf = c + (face if i == n else 0)
        ws += t * cf / (1 + r) ** i
    return ws / price


def modified_duration(face, coupon_rate, ytm, freq, maturity):
    return macaulay_duration(face, coupon_rate, ytm, freq, maturity) / (1 + ytm / freq)


def bond_convexity(face, coupon_rate, ytm, freq, maturity):
    n = round(maturity * freq)
    c = face * coupon_rate / freq
    r = ytm / freq
    price = bond_price(face, coupon_rate, ytm, freq, maturity)
    ws = 0.0
    for i in range(1, n + 1):
        t = i / freq
        cf = c + (face if i == n else 0)
        ws += cf * t * (t + 1 / freq) / (1 + r) ** i
    return ws / (price * (1 + r) ** 2)


if __name__ == "__main__":
    print("=" * 55)
    print(" Validation des formules obligataires (vs cas connus)")
    print("=" * 55 + "\n")

    # --- Test 1 : obligation zero-coupon -> duration = maturite EXACTEMENT ---
    face, ytm, freq, T = 100.0, 0.05, 1, 10.0
    price_zc = bond_price(face, 0.0, ytm, freq, T)
    dur_zc = macaulay_duration(face, 0.0, ytm, freq, T)
    print(f"[Test 1] Zero-coupon 10 ans : prix={price_zc:.4f} "
          f"(theorie: {face/(1+ytm)**T:.4f})")
    print(f"          Duration de Macaulay = {dur_zc:.6f} (theorie EXACTE: {T:.6f}) "
          f"-> {'OK' if abs(dur_zc-T)<1e-9 else 'ECART'}\n")

    # --- Test 2 : obligation au pair (coupon = ytm) -> prix = valeur nominale ---
    price_par = bond_price(100, 0.05, 0.05, 2, 7)
    print(f"[Test 2] Obligation au pair (coupon=ytm=5%, 7 ans, semestriel) : "
          f"prix={price_par:.6f} (theorie: 100.000000) "
          f"-> {'OK' if abs(price_par-100)<1e-6 else 'ECART'}\n")

    # --- Test 3 : duration modifiee vs choc de prix par difference finie ---
    face, coupon, ytm, freq, T = 100.0, 0.06, 0.05, 2, 10.0
    price0 = bond_price(face, coupon, ytm, freq, T)
    mod_dur = modified_duration(face, coupon, ytm, freq, T)
    convexity = bond_convexity(face, coupon, ytm, freq, T)

    shock = 0.01  # +100 bps
    price_up = bond_price(face, coupon, ytm + shock, freq, T)
    exact_change_pct = (price_up - price0) / price0

    approx_dur_only = -mod_dur * shock
    approx_dur_convex = -mod_dur * shock + 0.5 * convexity * shock ** 2

    err_dur = abs(approx_dur_only - exact_change_pct)
    err_convex = abs(approx_dur_convex - exact_change_pct)

    print(f"[Test 3] Choc de +100 bps sur une obligation 10 ans (coupon 6%, ytm 5%) :")
    print(f"  Variation de prix EXACTE (repricing complet)      : {exact_change_pct*100:+.4f}%")
    print(f"  Approximation duration seule (1er ordre)            : {approx_dur_only*100:+.4f}%  "
          f"(erreur: {err_dur*100:.4f} pts)")
    print(f"  Approximation duration + convexite (2eme ordre)      : {approx_dur_convex*100:+.4f}%  "
          f"(erreur: {err_convex*100:.4f} pts)")
    print(f"  -> la convexite reduit l'erreur d'approximation de {(1 - err_convex/err_dur)*100:.1f}%\n")

    print(f"Duration de Macaulay : {macaulay_duration(face, coupon, ytm, freq, T):.4f} ans")
    print(f"Duration modifiee     : {mod_dur:.4f}")
    print(f"Convexite               : {convexity:.4f}")
