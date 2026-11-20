"""Fit backtest.py's market model to real rounds, using our team's actual quotes there.

    python src/calibrate.py                       # fit to round 1; prints settings for backtest.py
    python src/calibrate.py "round 1" "round 2"   # pool several rounds (names from backtest.ROUNDS)

The market model (see backtest.py): N_RIVALS rivals quote the benchmark's price plus
RIVAL_SHIFT, their own lasting style (sd TEAM_SPREAD: some desks are always aggressive) and
RIVAL_NOISE per quote, except that with probability P_WILD a rival quote is wild:
anywhere between mid + WILD_LO and the +0.02 limit. Fills go by rank, FILL_BY_RANK: best
RANK1, second the observed share, then a geometric tail down to rank K that shares the rest.
The fit replays our real quotes against simulated rivals many times and matches what really
happened: how often we came first / second / lower / got nothing, how high the best and
second-best quotes were, and how big our lower-rank fills were.
"""
import sys

import numpy as np

import backtest as bt

RANK1 = 0.50        # best quote's share; round 0's only first place got 49.5%
OFF = 0.0199        # quotes stay inside mid +- 0.02
PCT = [10, 25, 50, 75, 90]
SPACE = dict(N_RIVALS=(5, 25), RIVAL_SHIFT=(-0.001, 0.003), TEAM_SPREAD=(0, 0.002),
             RIVAL_NOISE=(0.0001, 0.002), P_WILD=(0, 0.2), WILD_LO=(0.001, 0.015), K=(4, 12), RHO=(0.3, 0.95))


def observed(key):
    """Our real quotes in a round, the RFQs, and what happened to each quote."""
    quotes, history, _ = bt.ROUNDS[key]
    q, g = bt.table(quotes), bt.table(history)
    mine = bt.num(q["Quoted_coupon"])
    if np.all(np.round(mine, 2) == mine):
        sys.exit(f"{key}: its coupons are rounded to 2 decimals, so it can't be used")
    X = np.c_[tuple(bt.num(q[c]) for c in ["RFQ|Round", "RFQ|Notional", "RFQ|MidValue"] + ["RFQ|" + r for r in bt.RISKS])]
    at = {rid: i for i, rid in enumerate(q["RFQ ID"])}
    fill, best, second = np.zeros(len(mine)), np.full(len(mine), np.nan), np.full(len(mine), np.nan)
    for i, rid in enumerate(g["RFQ ID"]):
        if rid in at:
            j = at[rid]
            fill[j] = float(g["Notional"][i]) / X[j, 1]
            best[j], second[j] = float(g["Best_Quoted_coupon"][i]), float(g["Second_Quoted_coupon"][i])
    return X, mine, fill, best, second


def summary(mid, mine, fill, best, second, reps=1):
    """What the fit matches, for real or simulated (pooled over reps) outcomes."""
    filled = fill > 0
    first = filled & (mine >= best)
    second_place = filled & ~first & (mine >= second)
    lower = filled & (mine < second)
    pct = lambda x: np.percentile(x, PCT) if len(x) else np.full(len(PCT), np.nan)
    return dict(counts=np.array([first.sum(), second_place.sum(), lower.sum(), (~filled).sum()]) / reps,
                best=pct((best - mid)[filled]) * 1e4, second=pct((second - mid)[lower]) * 1e4,
                tail=np.r_[fill[lower].mean() if lower.any() else np.nan, pct(fill[lower])[[0, 2, 4]]],
                rank2=fill[second_place].mean() if second_place.any() else np.nan)


def shares(rank2, k, rho):
    rest = 1.0 - RANK1 - rank2
    w = rho ** np.arange(k - 2)
    return np.r_[RANK1, rank2, rest * w / w.sum()]


def simulate(c, mid, mine, bench, rng, reps):
    n, k = len(mid), c["N_RIVALS"]
    style = c["TEAM_SPREAD"] * rng.standard_normal((reps, k, 1))      # each desk's lasting style
    riv = bench[:k][None] + c["RIVAL_SHIFT"] + style + c["RIVAL_NOISE"] * rng.standard_normal((reps, k, n))
    wild = rng.random(riv.shape) < c["P_WILD"]
    riv = np.minimum(np.where(wild, mid + rng.uniform(c["WILD_LO"], OFF, riv.shape), riv), mid + OFF)
    rank = 1 + (riv > mine).sum(1)                                   # our rank, per rep and RFQ
    s = c["FILL_BY_RANK"]
    fill = np.where(rank <= len(s), s[np.minimum(rank, len(s)) - 1], 0.0)
    allq = np.concatenate([riv, np.broadcast_to(mine, (reps, 1, n))], axis=1)
    top = np.partition(allq, -2, axis=1)
    tile = lambda x: np.tile(x, reps)
    return summary(tile(mid), tile(mine), fill.ravel(), top[:, -1].ravel(), top[:, -2].ravel(), reps)


def loss(sim, obs):
    l = 2 * np.sum((sim["counts"] - obs["counts"]) ** 2 / (obs["counts"] + 5))     # fills drive P&L most
    l += np.nansum(((sim["best"] - obs["best"]) / 15) ** 2) + np.nansum(((sim["second"] - obs["second"]) / 4) ** 2)
    return l + np.nansum(((sim["tail"] - obs["tail"]) / 0.02) ** 2)


def main():
    keys = sys.argv[1:] or ["round 1"]
    parts = [observed(k) for k in keys]
    X = np.vstack([p[0] for p in parts])
    mine, fill, best, second = (np.concatenate([p[i] for p in parts]) for i in range(1, 5))
    mid = X[:, 2]
    obs = summary(mid, mine, fill, best, second)
    rank2 = obs["rank2"] if np.isfinite(obs["rank2"]) else 0.24

    # benchmark prices for 25 rival desks with books like ours (exotics from private moves only)
    rng = np.random.default_rng(0)
    rival = bt.Engine(f"{bt.ROOT}/{bt.BENCHMARK}")
    bench = np.zeros((25, len(mid)))
    for i in range(25):
        seat, exo = bt.synthetic_seat(rng), np.zeros(3)
        for t in range(1, 13):
            exo += seat[t - 1, 2:]
            rows = X[:, 0] == t
            bench[i, rows] = rival.quote(np.r_[seat[t - 1, :2], exo], t, X[rows, 1:])

    def make(v):
        c = {k: int(round(x)) if k in ("N_RIVALS", "K") else float(x) for k, x in v.items()}
        c["FILL_BY_RANK"] = shares(rank2, c["K"], c["RHO"])
        return c
    draw = lambda: make({k: rng.uniform(lo, hi) for k, (lo, hi) in SPACE.items()})
    vary = lambda c: make({k: np.clip(c[k] + rng.normal(0, 0.08 * (hi - lo)), lo, hi) for k, (lo, hi) in SPACE.items()})
    ranked = lambda cs, reps: sorted(((loss(simulate(c, mid, mine, bench, rng, reps), obs), c) for c in cs),
                                     key=lambda lc: lc[0])
    search = [c for _, c in ranked([draw() for _ in range(4000)], 8)[:10]]          # 1. random search
    refine = ranked(search + [vary(c) for c in search for _ in range(30)], 20)      # 2. refine the best 10
    l, c = ranked([c for _, c in refine[:15]], 100)[0]                              # 3. re-check the best 15
    sim = simulate(c, mid, mine, bench, np.random.default_rng(1), 200)

    print(f"fitted to {', '.join(keys)}: {len(mid)} RFQs, loss {l:.2f}\n")
    print(f"{'':34s}{'real':>34s}{'simulated':>34s}")
    for name, key, fmt in [("first / second / lower / no fill", "counts", "{:5.1f}"),
                           ("best quote - mid, bp (p10..p90)", "best", "{:6.1f}"),
                           ("2nd quote - mid when we're lower", "second", "{:6.1f}"),
                           ("our lower fills: mean, p10/50/90", "tail", "{:6.3f}")]:
        cell = lambda v: " ".join(fmt.format(x) for x in v)
        print(f"{name:34s}{cell(obs[key]):>34s}{cell(sim[key]):>34s}")
    print("\nsettings for backtest.py:")
    print(f"N_RIVALS     = {c['N_RIVALS']}\nRIVAL_SHIFT  = {c['RIVAL_SHIFT']:.5f}\nTEAM_SPREAD  = {c['TEAM_SPREAD']:.5f}"
          f"\nRIVAL_NOISE  = {c['RIVAL_NOISE']:.5f}")
    print(f"P_WILD       = {c['P_WILD']:.3f}\nWILD_LO      = {c['WILD_LO']:.4f}")
    print("FILL_BY_RANK = [" + ", ".join(f"{x:.3f}" for x in c["FILL_BY_RANK"]) + "]")


if __name__ == "__main__":
    main()
