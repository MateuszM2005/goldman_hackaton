"""Optimal engine with the exotic greeks weighted 0.9 when an RFQ is valued (EXOTIC_W). Otherwise identical to
optimal_engine.py; with EXOTIC_W = 1.0 it reproduces submissions/optimal/team.onnx.

    python src/models/optimal_exw_engine.py   ->  submissions/optimal_exw/<TEAM>.onnx  and  TestPredict.csv

Why 0.9: in games 2 and 3 the realised exotic moves overshot the plan (1.31x in game 3), because the fill curve
was too pessimistic for those rivals. Scaling the exotic part of each RFQ's value trims the moderate bids most and
the big breach-repair bids least (the best bid x*(v) flattens at high v). Measured against optimal ($mm per game):
arena 600 games -0.11 +- 0.04 (all of it against aggressive rivals), game 2 replay +0.24, game 3 replay +0.08 / +0.15.

Original description of optimal_engine.py:
Optimal engine: the profit-maximising quote, from the maths on the "Optimal Quoting Formulas" page.

Every one of the 32 copies of the engine computes, from its own row of X:

  1. lambda  the price of each of the 5 risks at the round's expected end: the root of
             lambda = C'(b + G(lambda)), found risk by risk on a grid of candidate prices.
             C'  this round's fees averaged over fill noise, plus (exotics) the expected cost
                 next round of bringing the book back inside the limits after the forecast
                 private move. That last part is the margin for next round.
             G   where the 32 fills land if every copy plays its part. The cube makes it the
                 same for any turn: G = -gamma(sigma_lambda) M lambda (Gaussian form).
  2. value   v = -<lambda, r> / N, what one dollar of this RFQ is worth to us.
  3. quote   mid + x*(v), with x*(v) = argmax_x s(x) (v - x) on the fill curve measured in arena.
"""
import os
from types import SimpleNamespace

import numpy as np
from onnx import helper

import cube_engine as ce          # the catalogue of private exotic paths, laid out for lookups
import engine as en               # numpy / ONNX builder, cols(), submit()
from engine import cols

TEAM = "Shame Street"

# ---- knobs
SIGMA_M    = 1.5      # forecast: how far our own carried exotics sit from a path's move (rounds 2-12)
SIGMA_M1   = 0.15     # ...in round 1, when nothing is carried yet
NONE_W     = 0.05     # forecast: weight of "no catalogue path fits"
FLOOR      = 0.3      # forecast: sd floor on next round's private exotic move
CARRY      = 0.5      # share of the later rounds' capital charged on an exotic held now
LIM_NEXT   = 2.6      # next round brings an exotic back inside this
NEXT_NOISE = 0.5      # next round's own fill noise (sd) on an exotic
LIQ        = 1.0      # scale on how far a price of risk moves the book
NOISE      = 1.0      # scale on this round's fill-noise variance
RISK       = 1.0      # scale on the price of risk when valuing an RFQ
EXTRA      = 0.0      # extra margin taken off every bid (coupon units)
EXOTIC_W   = 0.9      # weight on the exotic greeks when valuing an RFQ (1.0 = optimal_engine.py)
OUTER      = 3        # fixed-point passes (fill noise, competition between risks)
BOUND      = 0.0195   # |quote - mid| > 0.02 is off-market

# ---- desk rules                 delta   vega   vanna  volga  gap
UNIT   = np.array([0.006, 0.025, 0.006, 0.006, 0.006], np.float32)   # hedge / capital per unit
BETA   = np.array([0.05,  0.05,  1.0,   1.0,   1.0  ], np.float32)   # penalty per unit over the limit
LIMIT  = np.array([12.0,  3.0,   3.0,   3.0,   3.0  ], np.float32)
EXOTIC = np.array([0.0,   0.0,   1.0,   1.0,   1.0  ], np.float32)
M      = np.array([16.21, 1.013, 1.003, 1.003, 1.003], np.float32)   # E[r_k^2] over a round's 32 RFQs
NBAR   = 99.5                                                         # mean notional, $mm
LMAX   = np.array([0.07,  0.09,  3.3,   3.3,   3.3  ], np.float32)   # above any possible price of each risk

# ---- fill curve in arena (jittered fields of games 1 and 2), constant quote mid + x on all 32 RFQs:
# expected share of the RFQ and its variance. 150 games, 57600 RFQs per point. Below -40bp the arena
# still fills a little (a field of only 4-5 desks fills any 5th place); a real field won't, so the curve
# goes to 0 at -60bp.
CURVE_BP = np.array([-60, -40, -30, -20, -15, -12, -10, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 7, 10,
                     12, 15, 20, 25, 30, 40, 50, 65, 80, 100, 120, 150, 195], float)
SHARE = np.array([0.0, 0.0026, 0.0026, 0.0035, 0.0078, 0.0165, 0.0272, 0.0419, 0.0518, 0.0617, 0.0741, 0.0870,
                  0.0995, 0.1125, 0.1271, 0.1416, 0.1568, 0.1718, 0.1884, 0.2072, 0.2277, 0.2658, 0.3129, 0.3379,
                  0.3733, 0.4127, 0.4358, 0.4532, 0.4633, 0.4687, 0.4724, 0.4814, 0.4983, 0.5105, 0.5264, 0.5485])
VAR = np.array([0.0, 0.00009, 0.00009, 0.00013, 0.00040, 0.00146, 0.00363, 0.00710, 0.00979, 0.01244, 0.01609,
                0.01980, 0.02297, 0.02603, 0.02935, 0.03213, 0.03481, 0.03702, 0.03892, 0.04091, 0.04242, 0.04447,
                0.04404, 0.04300, 0.04138, 0.03659, 0.03288, 0.02961, 0.02796, 0.02702, 0.02641, 0.02475, 0.02140,
                0.01833, 0.01438, 0.00840])


# ---- tables worked out once from the curve ------------------------------------------------------
def _simplify(x, y, tol):
    """Fewest knots that keep the polyline (x, y) within tol (Douglas-Peucker)."""
    keep = np.zeros(len(x), bool)
    keep[[0, -1]] = True
    stack = [(0, len(x) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        line = y[i] + (y[j] - y[i]) * (x[i + 1:j] - x[i]) / (x[j] - x[i])
        k = int(np.argmax(np.abs(y[i + 1:j] - line)))
        if abs(y[i + 1 + k] - line[k]) > tol:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return x[keep], y[keep]


def _best_bid():
    """x*(v) = argmax_x s(x)(v - x) by brute force (0.05bp grid), with the share and fill variance it gets."""
    xg = np.linspace(-BOUND, BOUND, 7801)
    sg, vg = (np.interp(xg, CURVE_BP * 1e-4, c, left=0.0) for c in (SHARE, VAR))
    vv = np.linspace(-0.04, 0.10, 1401)                                       # 1bp steps
    i = np.concatenate([np.argmax(sg * (c[:, None] - xg), 1) for c in np.array_split(vv, 20)])
    i = np.maximum(i, np.searchsorted(xg, CURVE_BP[0] * 1e-4))            # nothing to earn: stay at the floor
    return vv, xg[i], sg[i], vg[i]


VV, XSTAR, SSTAR, VSTAR = _best_bid()
XK, XV = _simplify(VV, XSTAR, 2e-5)                    # x*(v) as a piecewise-linear function, within 0.2bp

# How far the 32 fills move the book for a price lambda: G = -GAMMA(sigma) M lambda, sigma^2 = lambda' M lambda,
# and the fill variance they bring: Var_k = M_k NU(sigma). Gaussian (Stein) form over u = <lambda, r>.
SIG = np.array([1e-4, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.4, 2.0,
                3.0, 4.5, 6.5, 10.0])
_Z = np.linspace(-8, 8, 3201)
_WZ = np.exp(-0.5 * _Z ** 2) / np.exp(-0.5 * _Z ** 2).sum()
GAMMA = np.array([-32.0 / s * np.sum(_WZ * np.interp(-s * _Z / NBAR, VV, SSTAR) * _Z) for s in SIG])
NU = np.array([32.0 * np.sum(_WZ * np.interp(-s * _Z / NBAR, VV, VSTAR)) for s in SIG])

# Candidate prices for the root search, lambda = LMAX sinh(ALPHA z) / sinh(ALPHA): dense near 0. A coarse
# pass over z in [-1, 1] (K1 points), then a fine one around its root (K2 points, +- one coarse step).
ALPHA, K1, K2 = 6.0, 17, 9
Z1 = np.linspace(-1, 1, K1).astype(np.float32)
DZ = float(Z1[1] - Z1[0])
Z2 = (DZ * np.linspace(-1, 1, K2)).astype(np.float32)
SH = float(np.sinh(ALPHA))
CAND = (LMAX[:, None] * np.sinh(ALPHA * Z1) / SH).astype(np.float32)          # 5 x K1
LO1, HI1 = np.eye(K1, dtype=np.float32)[:, :-1], np.eye(K1, dtype=np.float32)[:, 1:]
LO2, HI2 = np.eye(K2, dtype=np.float32)[:, :-1], np.eye(K2, dtype=np.float32)[:, 1:]
FIRST2 = np.eye(K2, dtype=np.float32)[:, :1]
ONES5 = np.ones((5, 1), np.float32)
PICK, PUT = ce.EMBED, ce.EMBED.T                                      # 3 x 5 and 5 x 3: exotics out / in
_NEXT = np.concatenate([ce.PATHS[:, 1:], np.zeros_like(ce.PATHS[:, :1])], 1)    # 36 x 12 x 3: next round's move
PRIOR1, PRIOR2 = _NEXT.mean(0), (_NEXT ** 2).mean(0)                  # 12 x 3, if no path fits


def _next_knots():
    """Next round's cost of moving one exotic by g: marginal cost l(g) as relu pieces, and its reach,
    the move at which one more unit costs as much as a breach (1 per unit)."""
    lam = np.linspace(0.0, 4.0, 4001)
    move = LIQ * np.interp(lam * np.sqrt(M[2]), SIG, GAMMA) * M[2] * lam
    reach = float(np.interp(1.0, lam, move))
    a = reach * np.array([0.0, 0.3, 0.55, 0.75, 0.9])
    ell = np.interp(np.r_[a, reach], move, lam)
    slope = np.diff(ell) / np.diff(np.r_[a, reach])
    return reach, a, np.r_[slope[0], np.diff(slope)]


# ---- numpy / ONNX helpers -----------------------------------------------------------------------
def _node(kind, a, **attrs):
    out = f"t{len(en.NODES)}"
    en.NODES.append(helper.make_node(kind, [a.name], [out], **attrs))
    return en.Sym(out)


en.Sym.__rmatmul__ = lambda s, o: en.op("MatMul", o, s)      # constant @ tensor (engine.py only has tensor @ x)


NP = SimpleNamespace(**vars(en.NUMPY), sqrt=np.sqrt, sinh=np.sinh, asinh=np.arcsinh,
                     unsq=lambda a: a[..., None], sq=lambda a: a[..., 0])
GR = SimpleNamespace(**vars(en.GRAPH), sqrt=lambda a: en.op("Sqrt", a), sinh=lambda a: en.op("Sinh", a),
                     asinh=lambda a: en.op("Asinh", a),
                     unsq=lambda a: _node("Unsqueeze", a, axes=[2]), sq=lambda a: _node("Squeeze", a, axes=[2]))
_ZS = (0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429)   # Zelen & Severo, error < 1e-7


def _pdf(z, F):
    return F.exp(z * z * -0.5) * 0.3989422804014327


def _cdf(z, F):
    a = F.abs(z)
    t = 1.0 / (1.0 + 0.2316419 * a)
    q = _pdf(a, F) * (t * (_ZS[0] + t * (_ZS[1] + t * (_ZS[2] + t * (_ZS[3] + t * _ZS[4])))))   # tail beyond |z|
    return q + (1.0 - 2.0 * q) * F.min(F.max(z * 1e7, 0.0), 1.0)


def _h(m, s, F):
    """E[(m + s Z)^+] for Z standard normal."""
    z = m / s
    return m * _cdf(z, F) + s * _pdf(z, F)


def _pwl(x, xs, ys, F):
    """Piecewise-linear interpolation through (xs, ys) as a sum of relus, flat outside."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    sl = np.diff(ys) / np.diff(xs)
    x = F.min(F.max(x, float(xs[0])), float(xs[-1]))
    out = (x - float(xs[0])) * float(sl[0]) + float(ys[0])
    for k in range(1, len(sl)):
        ds = float(sl[k] - sl[k - 1])
        if abs(ds) > 1e-12:
            out = out + F.relu(x - float(xs[k])) * ds
    return out


def _next_slope(y, s, F, reach, a, d):
    """Slope of next round's expected cost in its expected start y (sd s): moving back inside LIM_NEXT costs
    l(g) at the margin up to the reach, and a breach (1 per unit) beyond it."""
    L = LIM_NEXT
    out = _cdf((y - (L + reach)) / s, F) - _cdf((0.0 - y - (L + reach)) / s, F)
    far = _h(y - (L + reach), s, F) - _h(0.0 - y - (L + reach), s, F)
    for aj, dj in zip(a, d):
        out = out + float(dj) * (_h(y - (L + float(aj)), s, F) - _h(0.0 - y - (L + float(aj)), s, F) - far)
    return out


def _root(f, cand, lo_m, hi_m, F):
    """Root of f (increasing along the candidate axis), by linear interpolation in the bracketing interval.
    cand: the candidates, 5 x K or n x 5 x K."""
    lo, hi = f @ lo_m, f @ hi_m
    frac = F.min(F.max((0.0 - lo) / (hi - lo + 1e-9), 0.0), 1.0)
    return F.sq((frac * (cand @ hi_m - cand @ lo_m)) @ np.ones((lo_m.shape[1], 1), np.float32)
                + cand @ lo_m[:, :1])


# ---- the engine -----------------------------------------------------------------------------------
def parts(X, F):
    F = GR if F is en.GRAPH else NP if F is en.NUMPY else F
    book, t = X @ cols(0, 1, 2, 3, 4), F.min(F.max(X @ cols(5), 1.0), 12.0)
    N, mid, r = F.max(X @ cols(6), 1.0), X @ cols(7), X @ cols(8, 9, 10, 11, 12)
    exo = X @ cols(2, 3, 4)

    # 1. next round's private exotic move: catalogue posterior, mean and variance (n x 3)
    now = F.relu(1.0 - F.abs(t - ce.ROUNDS))                                  # n x 12: this round
    look = now @ ce.NOW                                                       # each path's move this round
    dist = (exo * exo) @ ce.ONES[3] - 2.0 * ((look * (exo @ ce.SPREAD)) @ ce.GROUP) + (look * look) @ ce.GROUP
    sm = SIGMA_M1 + (SIGMA_M - SIGMA_M1) * F.min(F.relu(t - 1.0), 1.0)
    like = F.exp(dist / (sm * sm) * -0.5)                                     # n x 36
    norm = like @ ce.ONES[36] + NONE_W
    w, w0 = (like / norm) @ ce.GROUP.T, NONE_W / norm
    nxt = now @ ce.NEXT                                                       # each path's move next round
    n_hat = (w * nxt) @ ce.SPREAD.T + w0 * (now @ PRIOR1)
    n_var = F.relu((w * nxt * nxt) @ ce.SPREAD.T + w0 * (now @ PRIOR2) - n_hat * n_hat) + FLOOR * FLOOR

    # 2. the price of each risk: root of lambda = C'(b + G(lambda)), risk by risk on CAND
    reach, a, d = _next_knots()
    live = F.unsq(F.min(F.relu(12.0 - t), 1.0))                               # n x 1 x 1: a next round exists
    hold = F.unsq(1.0 + (12.0 - t) * (EXOTIC * CARRY))                        # n x 5 x 1: rounds of capital
    b3, y_shift = F.unsq(book), F.unsq(n_hat)
    y_var = F.unsq(n_var) + NEXT_NOISE ** 2
    def excess(cand, others, noise):
        """lambda - C'(b + G(lambda)) for candidate prices cand: increasing in lambda, 0 at the answer."""
        gam = _pwl(F.sqrt(others + cand * cand * M[:, None]), SIG, GAMMA, F) * LIQ
        end = b3 - gam * cand * M[:, None]                                    # expected end of the round
        sd = F.sqrt(noise)
        cp = (UNIT[:, None] * hold * (2.0 * _cdf(end / sd, F) - 1.0)
              + BETA[:, None] * (_cdf((end - LIMIT[:, None]) / sd, F) - _cdf((0.0 - end - LIMIT[:, None]) / sd, F)))
        ye = PICK @ end + y_shift                                             # next round's expected start
        sy = F.sqrt(PICK @ noise + y_var)
        return cand - cp - PUT @ (_next_slope(ye, sy, F, reach, a, d) * live)

    lam = book * 0.0
    for _ in range(OUTER):
        s2 = (lam * lam) @ M[:, None]                                         # n x 1: sigma_lambda^2
        noise = F.unsq(_pwl(F.sqrt(s2), SIG, NU, F) * (M * NOISE))            # n x 5 x 1: fill variance
        others = F.unsq(F.relu(s2 - lam * lam * M))                           # the other risks' share of it
        coarse = _root(excess(CAND, others, noise), CAND, LO1, HI1, F)        # n x 5
        z = F.asinh(coarse * (SH / LMAX)) * (1.0 / ALPHA)                     # back to the z scale
        fine = F.sinh(F.min(F.max(F.unsq(z) + Z2, -1.0), 1.0) * ALPHA) * (LMAX[:, None] / SH)   # n x 5 x K2
        lam = _root(excess(fine, others, noise), fine, LO2, HI2, F)

    # 3. this RFQ's value per dollar, and the bid that earns the most on it
    v = (r * lam * np.array([[1.0, 1.0, EXOTIC_W, EXOTIC_W, EXOTIC_W]], np.float32)) @ ONES5 * (-RISK) / N
    x = _pwl(v, XK, XV, F) - EXTRA
    quote = F.min(F.max(mid + x, mid - BOUND), mid + BOUND)
    return dict(quote=quote, lam=lam, v=v, n_hat=n_hat, n_var=n_var)


def price(X, F):
    return parts(X, F)["quote"]


if __name__ == "__main__":
    en.submit(os.path.join(en.ROOT, "submissions", "optimal_exw"), TEAM, price)
