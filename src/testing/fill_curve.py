"""Fill curve for the need-vector engine, measured against the arena's fitted rivals.

    python src/testing/fill_curve.py      # writes src/models/need_curve.json, prints the tables

phi(x) is the expected share of an RFQ we win when we quote mid + x. Rank k pays share s_k
(our 212 real fills: first ~55%, second ~25%, lower places a few %), so
    phi(x)    = sum_k (s_k - s_k+1) G_k(x)          G_k(x) = P(k-th best rival quote < mid + x)
    E[f^2](x) = sum_k (q_k - q_k+1) G_k(x)          q_k = E[s_k^2]   (for the fill noise)
G_k comes from the arena's rival desks on the real RFQs, half from game 1's fitted field and
half from game 2's, as the arena's synthetic games draw them.

For an RFQ worth v to our book (per $ of notional), the best bid maximises phi(x) (v - x):
    x*(v) = argmax_x phi(x) (v - x)       phi*(v) = phi(x*(v))       var*(v) = E[f^2] - phi^2
These three tables (plus phi*(v) against each field alone: phi1, phi2) are what need_engine.py uses, at knots picked so that straight lines
between them stay within TOL of every table (the best bid can jump where the curve is flat).
Below the value at which any bid fills at all, the bid falls 1bp per bp of value. The script also
checks the regularity the proof needs: m(x) = x + phi/phi' increasing.
"""
import json
import os

import numpy as np

import arena as A

OUT = os.path.join(A.ROOT, "src", "models", "need_curve.json")
X = np.round(np.arange(-199.0, 199.01, 0.25), 2) * 1e-4                 # bid - mid grid
V_FINE = np.round(np.arange(-120.0, 600.01, 0.25), 2) * 1e-4             # value grid
TOL = (0.3e-4, 0.004, 0.0015)      # knot tolerances: bid (0.3bp), fill share, fill variance
REPS = 300


def rank_moments(field, rng, n=400_000):
    """E[s_k] and E[s_k^2] for ranks 1..K under the arena's share model."""
    s = A.Shares(field).draw(rng, (n,))
    return s.mean(0), (s ** 2).mean(0)


def regime_curve(k):
    """phi(x) and E[f^2](x) against the field fitted to real game k, on game k's RFQs."""
    field = A.field_for(k)
    g = A.RealGame(k)
    rng = np.random.default_rng(1000 + k)
    *_, riv = A.sim_outcomes(g, field, rng, REPS)                     # (reps, 384, 10), sorted
    mid = np.array([r[:, 1] for r in g.rfqs]).ravel()
    rel = (riv - mid[None, :, None]).reshape(-1, riv.shape[-1])        # spread of each rival quote
    m1, m2 = rank_moments(field, rng)
    K = len(m1)
    G = np.array([np.searchsorted(np.sort(rel[:, j]), X, side="left") / len(rel) for j in range(K)])
    d1, d2 = m1 - np.r_[m1[1:], 0.0], m2 - np.r_[m2[1:], 0.0]
    return d1 @ G, d2 @ G, m1, K


def simplify(v, ys, tols):
    """Knot indices (Ramer-Douglas-Peucker) so linear interpolation stays within tols of every series."""
    keep, stack = {0, len(v) - 1}, [(0, len(v) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        worst, idx = 0.0, None
        for y, tol in zip(ys, tols):
            line = y[a] + (y[b] - y[a]) * (v[a + 1:b] - v[a]) / (v[b] - v[a])
            err = np.abs(y[a + 1:b] - line) / tol
            i = int(err.argmax())
            if err[i] > worst:
                worst, idx = err[i], a + 1 + i
        if worst > 1:
            keep.add(idx)
            stack += [(a, idx), (idx, b)]
    return sorted(keep)


def main():
    curves = [regime_curve(k) for k in (1, 2)]
    phi = np.mean([c[0] for c in curves], 0)
    ef2 = np.mean([c[1] for c in curves], 0)

    obj = phi[None, :] * (V_FINE[:, None] - X[None, :])                 # (values, bids)
    best = np.maximum.accumulate(obj.argmax(1))                          # x*(v) never decreases in v
    xs, ps = X[best], phi[best]
    vs = np.maximum(ef2[best] - ps ** 2, 0.0)
    i0 = int(np.argmax(ps >= 0.002))                                    # first value worth any fill
    xs[:i0] = xs[i0] - (V_FINE[i0] - V_FINE[:i0])                       # below it: 1bp lower per bp
    ps[:i0], vs[:i0] = 0.0, 0.0
    xs = np.maximum(xs, -0.0195)
    knots = simplify(V_FINE, (xs, ps, vs), TOL)
    V_KNOTS = V_FINE[knots]
    kx, kp, kv = xs[knots], ps[knots], vs[knots]
    kp1, kp2 = (np.interp(kx, X, c[0]) * (kp > 0) for c in curves)          # the same bids against each field
    print(len(V_KNOTS), "knots")

    # regularity (A4): m(x) = x + phi / phi' should increase wherever we would actually bid
    sm = np.convolve(phi, np.ones(9) / 9, mode="same")
    dphi = np.gradient(sm, X)
    live = (sm > 0.002) & (sm < phi.max() - 0.01) & (dphi > 0)
    mx = X + sm / np.where(dphi > 0, dphi, np.inf)
    rising = np.mean(np.diff(mx[live]) > -1e-6)

    print("fill curve phi(x), half game-1 rivals, half game-2 rivals")
    print("   bid - mid (bp):  " + "".join(f"{x:>8.0f}" for x in (-60, -30, -20, -10, -5, -2, 0, 2, 5, 10, 20, 30, 50, 100, 199)))
    at = lambda arr, x: np.interp(x * 1e-4, X, arr)
    for label, arr in (("game 1 field", curves[0][0]), ("game 2 field", curves[1][0]), ("both", phi)):
        print(f"   {label:15s}" + "".join(f"{at(arr, x):8.3f}" for x in (-60, -30, -20, -10, -5, -2, 0, 2, 5, 10, 20, 30, 50, 100, 199)))
    print(f"   share by rank (mean): game 1 field {np.round(curves[0][2], 3)}, game 2 field {np.round(curves[1][2], 3)}")
    print(f"   regularity: m(x) = x + phi/phi' rises on {rising:.0%} of the live range")
    print("\nbest bid for a value v (bp of notional): x*(v) = argmax phi(x)(v - x)")
    print(f"   {'v':>6s} {'bid':>7s} {'fill':>6s} {'sd':>6s}")
    for v, x, p, s in zip(V_KNOTS, kx, kp, kv):
        print(f"   {v * 1e4:6.1f} {x * 1e4:+7.1f} {p:6.3f} {np.sqrt(s):6.3f}")

    with open(OUT, "w") as f:
        json.dump(dict(v=V_KNOTS.tolist(), x=kx.tolist(), phi=kp.tolist(), var=kv.tolist(),
                       phi1=kp1.tolist(), phi2=kp2.tolist(),
                       curve=dict(x=X.tolist(), phi=phi.tolist(), ef2=ef2.tolist()),
                       meta=dict(reps=REPS, regimes=["game 1", "game 2"],
                                 shares=[c[2].tolist() for c in curves], regularity=float(rising))), f)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
