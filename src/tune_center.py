"""Tune engine_center.py's settings on catalogue-based games (src/sim_center.py).

    python src/tune_center.py          # prints the best settings and a validation table
    python src/tune_center.py 1234     # repeat a run: the seed it printed

Every stage plays new games: new RFQs (any cube rotation), private exotic paths from the
33 catalogue paths the real games did not use (sizes 0.5 / 1 / 2, so a third are game-2
style storms), random delta/vega phase, a random market around the calibrated one. Within
a stage all candidates play the same games. The real games' paths are only used at the end.
  1. search    current + old-engine-equivalent + random settings
  2. refine    the best 12 and 8 variations of each
  3. polish    the best 6 and 10 smaller variations of each
  4. validate  the best 5, the current and old settings on 400 new games and the 3 real games
"""
import multiprocessing as mp
import os
import sys
import time

import numpy as np

import engine_center as ec
import sim_center as sc

SPACE = {
    "MARGIN":       (0.0, 0.0015),
    "PASS":         (0.0, 1.0),
    "MAX_DISCOUNT": (0.0, 0.004),
    "CEILING":      (-0.001, 0.0005),
    "RELIEF_PASS":  (0.0, 1.0),       # kept <= 1 - PASS
    "MAX_RELIEF":   (0.0, 0.005),
    "BREACH_H":     (1.0, 12.0),
    "CAPITAL_H":    (1.0, 12.0),
    "MOVE_DV":      (0.1, 4.0),
    "MOVE_EX":      (0.1, 4.0),
    "SKEW":         (0.0, 2.0),
    "WL_DELTA":     (8.0, 12.0),
    "WL_VEGA":      (2.0, 3.0),
    "WL_EXOTIC":    (1.5, 3.5),       # the first run's best sat at 2.44; 3.0 (the real limit) beat it
}
# src/engine.py's tuned settings (commit 0fcf42b), written in this engine's terms
OLD = dict(MARGIN=0.00001, PASS=1.0, MAX_DISCOUNT=0.00174, CEILING=-0.00024, RELIEF_PASS=0.0,
           MAX_RELIEF=0.0, BREACH_H=12.0, CAPITAL_H=12.0, MOVE_DV=0.3, MOVE_EX=0.3, SKEW=0.05,
           WL_DELTA=11.1, WL_VEGA=2.25, WL_EXOTIC=2.6)
CURRENT = {k: float(ec.P[k]) for k in SPACE}


def fix(s):
    s["RELIEF_PASS"] = min(s["RELIEF_PASS"], 1.0 - s["PASS"])
    return s


def init():
    global RIVAL
    RIVAL = sc.Onnx(sc.BENCHMARK)
    sc.exotic_paths()


def score(task):
    """One candidate on every seed: per game (P&L, round P&L sd, breaches, penalty)."""
    settings, seeds = task
    engine = ec.Engine(settings)
    out = []
    for seed in seeds:
        t = sc.total(sc.synthetic_game(engine, RIVAL, seed))
        out.append((t["pnl"], t["round_sd"], t["breaches"], t["penalty"]))
    return np.array(out)


def real(task):
    settings, key, draws = task
    engine = sc.NeverTrade() if settings == "never" else ec.Engine(settings)
    return np.mean([sc.total(sc.real_replay(engine, RIVAL, key, 7_000 + d))["pnl"] for d in range(draws)])


def run(pool, cands, seeds, label):
    res = pool.map(score, [(c, seeds) for c in cands])
    ranked = sorted(zip(cands, res), key=lambda cr: -cr[1][:, 0].mean())
    print(f"{label}: {len(cands)} settings x {len(seeds)} games, best {ranked[0][1][:, 0].mean():+.2f}"
          f"  (current {next(r for c, r in zip(cands, res) if c is cands[0])[:, 0].mean():+.2f})", flush=True)
    return ranked


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else int(time.time())
    print("seed", seed, flush=True)
    rng = np.random.default_rng(seed)
    games = lambda n: [int(s) for s in rng.integers(10_000, 2**31, n)]
    draw = lambda: fix({k: float(rng.uniform(lo, hi)) for k, (lo, hi) in SPACE.items()})
    vary = lambda s, w: fix({k: float(np.clip(s[k] + rng.normal(0, w * (hi - lo)), lo, hi))
                             for k, (lo, hi) in SPACE.items()})
    now, old = dict(CURRENT), dict(OLD)
    with mp.Pool(initializer=init) as pool:
        r1 = run(pool, [now, old] + [draw() for _ in range(318)], games(60), "1 search")
        best = [c for c, _ in r1[:12]]
        r2 = run(pool, [now] + best + [vary(c, 0.10) for c in best for _ in range(8)], games(100), "2 refine")
        top = [c for c, _ in r2[:6] if c is not now][:6]
        r3 = run(pool, [now] + top + [vary(c, 0.04) for c in top for _ in range(10)], games(150), "3 polish")
        final = [c for c, _ in r3 if c is not now][:5]
        cands = final + [now, old]
        seeds = games(400)
        val = pool.map(score, [(c, seeds) for c in cands])
        reals = {key: pool.map(real, [(c, key, 10) for c in cands + ["never"]]) for key in sc.REAL}

    base = val[-1][:, 0]
    print("\nvalidation: 400 new games ($mm per game) and the 3 real games (mean of 10 markets)")
    print(f"{'':10s}{'P&L':>7s}{'se':>6s}{'vs old':>13s}{'worst':>8s}{'round sd':>9s}{'breach':>7s}{'penalty':>8s}"
          + "".join(f"{' '.join(k.split()[:2]):>10s}" for k in sc.REAL))
    for i, (c, res) in enumerate(zip(cands, val)):
        tag = "old" if c is old else "current" if c is now else f"cand {i + 1}"
        p, d = res[:, 0], res[:, 0] - base
        print(f"{tag:10s}{p.mean():+7.2f}{p.std() / np.sqrt(len(p)):6.2f}{d.mean():+8.2f}+-{d.std() / np.sqrt(len(d)):.2f}"
              f"{p.min():+8.1f}{res[:, 1].mean():9.3f}{res[:, 2].mean():7.1f}{res[:, 3].mean():8.2f}"
              + "".join(f"{reals[k][i]:+10.2f}" for k in sc.REAL))
    print(f"{'never':10s}{'':55s}" + "".join(f"{reals[k][-1]:+10.2f}" for k in sc.REAL))
    print("\nsettings")
    for i, c in enumerate(final):
        print(f"cand {i + 1}: " + ", ".join(f"{k}={c[k]:.4g}" for k in SPACE))
    print("old:    " + ", ".join(f"{k}={old[k]:.4g}" for k in SPACE))


if __name__ == "__main__":
    main()
