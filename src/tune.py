"""Tune engine.py's settings in simulation, on fresh random games at every stage.

    python src/tune.py        # about two minutes on 12 cores; prints the best settings
    python src/tune.py 1234   # repeat a run: the seed it printed

Every run and every stage plays new games: new RFQs, private positions and rival books, and a
random market per game around the one calibrate.py fitted to round 1 (rivals +-5, +-5bp more
or less aggressive, styles +-30%, noise and wild share +-50%), so the settings can't fit one
particular set of games or one exact market. Private positions have the size
both real games had (backtest.SEAT_SIZES). Within a stage all candidates play the same games,
so they are compared like for like.
  1. search:   the current settings plus 159 random ones, 40 games
  2. refine:   the best 10 plus 6 variations of each, 60 new games
  3. validate: the best 6, the current and the old settings, and never trading, 300 new games
FILL and SKEW stay at engine.py's values.
"""
import multiprocessing as mp
import os
import sys
import time

import numpy as np

import backtest as bt
import engine as en

SPACE = {                    # setting: (low, high)
    "MARGIN":       (0.0, 0.002),
    "PASS_THROUGH": (0.0, 1.0),
    "MAX_DISCOUNT": (0.0, 0.008),
    "CEILING":      (-0.001, 0.001),
    "BREACH_PASS":  (0.0, 1.0),      # kept <= 1 - PASS_THROUGH, so a won trade never loses
    "MAX_RELIEF":   (0.0, 0.004),
    "WL_DELTA":     (8.0, 12.0),     # working limits; official limits are 12 / 3 / 3
    "WL_VEGA":      (2.0, 3.0),
    "WL_EXOTIC":    (1.0, 3.0),
}
OLD = dict(MARGIN=0.001, PASS_THROUGH=0.5, MAX_DISCOUNT=0.002, CEILING=0.02, BREACH_PASS=0.0,
           MAX_RELIEF=0.0, SKEW=0.0, WL_DELTA=10.0, WL_VEGA=2.5, WL_EXOTIC=2.0)  # first version


def current():
    wl = en.WORKING_LIMIT[0]
    s = {k: float(getattr(en, k)) for k in SPACE if not k.startswith("WL_")}
    return dict(s, SKEW=float(en.SKEW), WL_DELTA=float(wl[0]), WL_VEGA=float(wl[1]), WL_EXOTIC=float(wl[2]))


DEFAULTS = current()         # engine.py as it is now; every candidate starts from these
MARKET = dict(N_RIVALS=bt.N_RIVALS, RIVAL_SHIFT=bt.RIVAL_SHIFT, TEAM_SPREAD=bt.TEAM_SPREAD,
              RIVAL_NOISE=bt.RIVAL_NOISE, P_WILD=bt.P_WILD)          # the calibrated market


class Candidate:
    """engine.price() with the given settings, run in numpy (same numbers as the ONNX)."""
    def __init__(self, s):
        self.s = dict(DEFAULTS, **s)

    def quote(self, book, rnd, rfqs):
        for k, v in self.s.items():
            if not k.startswith("WL_"):
                setattr(en, k, v)
        en.WORKING_LIMIT = np.array([[self.s["WL_DELTA"], self.s["WL_VEGA"]] + [self.s["WL_EXOTIC"]] * 3],
                                    np.float32)
        X = np.c_[np.tile(book, (len(rfqs), 1)), np.full(len(rfqs), rnd), rfqs].astype(np.float32)
        return en.price(X, en.NUMPY)[:, 0].astype(float)


class NeverTrade:
    """Reference: always quotes the lowest allowed price, so it never wins a trade."""
    def quote(self, book, rnd, rfqs):
        return rfqs[:, 1] - 0.0195


def fix(s):
    s["BREACH_PASS"] = min(s["BREACH_PASS"], 1.0 - s["PASS_THROUGH"])
    return s


def init():
    global RIVAL, SIZES
    RIVAL = bt.Engine(os.path.join(bt.ROOT, bt.BENCHMARK))
    SIZES = bt.rfq_sizes()


def score(task):
    """One candidate on every seed: per game (P&L, sd of round P&L, breaches)."""
    settings, seeds = task
    engine = NeverTrade() if settings == "never" else Candidate(settings)
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        bt.RIVAL_SHIFT = MARKET["RIVAL_SHIFT"] + rng.uniform(-0.0005, 0.0005)
        bt.TEAM_SPREAD = MARKET["TEAM_SPREAD"] * rng.uniform(0.7, 1.3)
        bt.RIVAL_NOISE = MARKET["RIVAL_NOISE"] * rng.uniform(0.5, 1.5)
        bt.P_WILD = MARKET["P_WILD"] * rng.uniform(0.5, 1.5)
        rounds = bt.synthetic_rfqs(rng, SIZES)
        seats = [bt.synthetic_seat(rng) for _ in range(1 + max(2, MARKET["N_RIVALS"] + int(rng.integers(-5, 6))))]
        rows = bt.play(engine, RIVAL, rounds, seats, rng)
        pnl = [r["pnl"] for r in rows]
        out.append((sum(pnl), np.std(pnl), sum(r["breaches"] for r in rows)))
    return np.array(out)


def run(pool, cands, seeds, label):
    res = pool.map(score, [(c, seeds) for c in cands])
    ranked = sorted(zip(cands, res), key=lambda cr: -cr[1][:, 0].mean())
    print(f"{label}: {len(cands)} settings x {len(seeds)} new games, best mean P&L {ranked[0][1][:, 0].mean():+.2f}")
    return ranked


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else int(time.time())
    print("seed", seed)
    rng = np.random.default_rng(seed)
    games = lambda n: [int(s) for s in rng.integers(1_000, 2**31, n)]   # never backtest.py's 0-30
    draw = lambda: fix({k: float(rng.uniform(lo, hi)) for k, (lo, hi) in SPACE.items()})
    vary = lambda s: fix({k: float(np.clip(s[k] + rng.normal(0, 0.1 * (hi - lo)), lo, hi))
                          for k, (lo, hi) in SPACE.items()})
    now = {k: DEFAULTS[k] for k in SPACE}
    with mp.Pool(initializer=init) as pool:
        r1 = run(pool, [now] + [draw() for _ in range(159)], games(40), "1 search")
        best = [c for c, _ in r1[:10]]
        r2 = run(pool, best + [vary(c) for c in best for _ in range(6)], games(60), "2 refine")
        top = [c for c, _ in r2[:6]]
        r3 = run(pool, top + [c for c in (now, OLD) if c not in top] + ["never"], games(300), "3 validate")

    base = next(res for c, res in r3 if c == now)
    print("\nvalidation, 300 new games, $mm per game"
          "\n              P&L    se  vs current  >0 games  round sd  breaches   " + "  ".join(SPACE))
    for c, res in r3:
        tag = {id(now): "(current)", id(OLD): "(old)"}.get(id(c), "(never trade)" if c == "never" else "")
        d, p = res[:, 0] - base[:, 0], res[:, 0]
        cells = "" if c == "never" else "  ".join(f"{c[k]:{len(k)}.4g}" for k in SPACE)
        print(f"{tag:13s}{p.mean():+6.2f} {p.std() / np.sqrt(len(p)):5.2f}  {d.mean():+5.2f}+-{d.std() / np.sqrt(len(d)):.2f}"
              f"  {np.mean(p > 0):7.0%}  {res[:, 1].mean():8.3f}  {res[:, 2].mean():8.2f}   {cells}")


if __name__ == "__main__":
    main()
