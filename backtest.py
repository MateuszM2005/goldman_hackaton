"""Backtest ONNX quoting engines on the example game plus synthetic games.

    python backtest.py                            # submission/team.onnx vs benchmark.onnx
    python backtest.py a.onnx b.onnx --games 50   # any engines, more synthetic games
    python backtest.py --rivals 10 --shift 3      # bigger field, quoting 3bp more aggressively
    python backtest.py --check                    # rebuild the example trader's fees from its fills

Every engine sits in the same seat of the same market (same RFQs, private positions and
rivals), one run each, so a difference in P&L comes from the engine alone.

Real:     the example game's RFQs and private positions, and the fee rules (--check matches
          trader_pnl_by_round.csv). Synthetic games copy the example's structure: each round,
          one RFQ per sign pattern of the 5 risks (32), sizes resampled from the real RFQs;
          private delta/vega follow the example's 12-round sine wave at 0.5x, 1x or 2x size.
Assumed:  the files round every coupon to 2 decimals, so who won can't be read off them.
          Rivals run benchmark.onnx on their own books plus RIVAL_NOISE (real teams differ more
          than benchmark copies, which tie a third of the time). The best quote gets TOP_FILL of
          the RFQ; other quotes get up to REST_FILL, shrinking to nothing GAP below the best.
          Calibrated by putting benchmark.onnx in the example trader's seat: it reproduces the
          real fill count (141 RFQs), best-quote count (69 vs 60) and round-by-round pattern
          (corr 0.92), but edge comes out ~35% low and penalties ~65% high. Trust differences
          between engines more than absolute P&L, and recalibrate after each training game.
"""
import argparse
import csv
import os

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- fee rules from the handout           delta   vega  vanna  volga    gap
UNIT_COST   = np.array([0.006, 0.025, 0.006, 0.006, 0.006])   # hedge / capital per $mm
LIMIT       = np.array([12.0,  3.0,   3.0,   3.0,   3.0  ])
BREACH_RATE = np.array([0.05,  0.05,  1.0,   1.0,   1.0  ])   # penalty per $mm over the limit
OFF_MARKET  = 0.02                                             # |quote - mid| above this: no fill

# ---- market assumptions (see docstring), calibrated on the example game
N_RIVALS    = 4
TOP_FILL    = 0.25         # the example's fills: ~21-26% when best, ~2-10% otherwise
REST_FILL   = 0.10
GAP         = 0.0002
RIVAL_NOISE = 0.0004       # sd of random noise on each rival quote
RIVAL_SHIFT = -0.00015     # added to every rival quote; raise it to test a more aggressive field

RISKS = ["Delta", "Vega", "Exotic Risk 1-Vanna", "Exotic Risk 2-Volga", "Exotic Risk 3-Gap risk"]
SIGNS = np.array([[1 if k >> (4 - i) & 1 else -1 for i in range(5)] for k in range(32)])


def table(name):
    """CSV as {column: [cells]}, with headers and cells stripped and blank rows dropped."""
    with open(os.path.join(HERE, name), newline="") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f)]
    body = [r for r in rows[1:] if r and r[0]]
    return {h: [r[i] for r in body] for i, h in enumerate(rows[0]) if h}


def num(cells):
    return np.array([0.0 if c in ("", "-") else float(c) for c in cells])


def example_rfqs(rng):
    """The example game's RFQs per round: 32 x [notional, mid, 5 risks]. Mids are rounded to
    0.05 / 0.06 in the file, so each is redrawn inside its rounding interval."""
    q = table("trader_quote_history.csv")
    X = np.c_[tuple(num(q[c]) for c in ["RFQ|Notional", "RFQ|MidValue"] + ["RFQ|" + r for r in RISKS])]
    X[:, 1] = rng.uniform(np.maximum(X[:, 1] - 0.005, 0.047), np.minimum(X[:, 1] + 0.005, 0.058))
    rnd = num(q["RFQ|Round"])
    return [X[rnd == t] for t in range(1, 13)]


def example_seat():
    """The example trader's 12 private positions."""
    g = table("trader_game_history.csv")
    private = np.array([i.startswith("R") for i in g["RFQ ID"]])
    return np.c_[tuple(num(g[c]) for c in RISKS)][private]


def synthetic_rfqs(rng, sizes):
    """12 rounds like the example: one RFQ per sign pattern, sizes drawn from real RFQs."""
    rounds = []
    for _ in range(12):
        s = sizes[rng.integers(len(sizes), size=32)]
        rounds.append(np.c_[s[:, 0], rng.uniform(0.047, 0.058, 32), SIGNS * s[:, 1:]])
    return rounds


def synthetic_seat(rng):
    """12 private positions shaped like the example's: delta/vega on a sine wave at 0.5x, 1x or
    2x the example's 15 / 3, an inherited exotic book in round 1, then small exotic moves."""
    size = rng.choice([0.5, 1.0, 2.0])
    angle = rng.uniform(0, 2 * np.pi) + np.arange(1, 13) * np.pi / 6
    seat = np.zeros((12, 5))
    seat[:, 0] = -15 * size * np.sin(angle)
    seat[:, 1] = 3 * size * np.cos(angle)
    seat[:, 2:] = 0.3 * np.sin(angle[:, None] + rng.uniform(0, 2 * np.pi, 3))
    seat[0, 2:] += rng.uniform(-2.5, 2.5, 3)
    return seat


class Engine:
    def __init__(self, path):
        self.name = os.path.basename(path)
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name

    def quote(self, book, rnd, rfqs):
        X = np.c_[np.tile(book, (len(rfqs), 1)), np.full(len(rfqs), rnd), rfqs].astype(np.float32)
        return self.sess.run(None, {self.input: X})[0].reshape(-1).astype(float)


def fees(pos):
    """End-of-round bill on pos = [delta, vega, vanna, volga, gap] (exotics cumulative)."""
    excess = np.maximum(np.abs(pos) - LIMIT, 0)
    return (UNIT_COST[:2] @ np.abs(pos[:2]), UNIT_COST[2:] @ np.abs(pos[2:]),
            BREACH_RATE @ excess, int((excess > 0).sum()))


def play(engine, rival, rounds, seats, rng):
    """One game with `engine` in seat 0 and `rival` in the others. Returns seat 0's rounds."""
    exo = np.zeros((len(seats), 3))
    out = []
    for t, rfqs in enumerate(rounds, 1):
        mid = rfqs[:, 1]
        quotes = np.empty((len(seats), len(rfqs)))
        for i, seat in enumerate(seats):
            exo[i] += seat[t - 1, 2:]                       # private exotics join the book
            book = np.r_[seat[t - 1, :2], exo[i]]
            quotes[i] = (engine if i == 0 else rival).quote(book, t, rfqs)
        quotes[1:] += RIVAL_SHIFT + RIVAL_NOISE * rng.standard_normal(quotes[1:].shape)
        ok = np.isfinite(quotes) & (np.abs(quotes - mid) <= OFF_MARKET)
        with np.errstate(invalid="ignore"):
            gap = np.where(ok, quotes, -np.inf).max(0) - quotes
            fill = np.where(ok, np.where(gap <= 1e-12, TOP_FILL,
                                         REST_FILL * np.clip(1 - gap / GAP, 0, 1)), 0.0)
        took = np.einsum("sr,rk->sk", fill, rfqs[:, 2:])   # risk each seat takes on
        exo += took[:, 2:]
        pos = np.c_[np.array([s[t - 1, :2] for s in seats]) + took[:, :2], exo]
        edge = (fill[0] * rfqs[:, 0] * (mid - np.where(ok[0], quotes[0], 0))).sum()
        hedge, capital, penalty, breaches = fees(pos[0])
        out.append(dict(pnl=edge - hedge - capital - penalty, edge=edge, hedge=hedge,
                        capital=capital, penalty=penalty, breaches=breaches,
                        filled=int((fill[0] > 0).sum()), best=int((fill[0] == TOP_FILL).sum()),
                        off_market=int((~ok[0]).sum())))
    return out


def check():
    """Rebuild the example trader's fees from its own fills; compare with trader_pnl_by_round.csv."""
    g, p = table("trader_game_history.csv"), table("trader_pnl_by_round.csv")
    rnd, risk = num(g["Round"]), np.c_[tuple(num(g[c]) for c in RISKS)]
    exo, worst = np.zeros(3), 0.0
    print("round   costs ours/file   penalty ours/file   breaches ours/file   P&L ours/file")
    for t in range(1, 13):
        took = risk[rnd == t].sum(0)                        # private position + fills
        exo += took[2:]
        hedge, capital, penalty, breaches = fees(np.r_[took[:2], exo])
        f = {k: num(p[k])[t - 1] for k in ["Costs", "Aggregate Penalty", "Breaches", "Edge", "PnL"]}
        costs, pnl = hedge + capital + penalty, f["Edge"] - (hedge + capital + penalty)
        worst = max(worst, abs(costs - f["Costs"]), abs(penalty - f["Aggregate Penalty"]), abs(pnl - f["PnL"]))
        print(f"{t:5d}   {costs:6.2f} / {f['Costs']:5.2f}    {penalty:6.2f} / {f['Aggregate Penalty']:5.2f}"
              f"    {breaches:8d} / {f['Breaches']:2.0f}       {pnl:6.2f} / {f['PnL']:5.2f}")
    print(f"largest difference: {worst:.3f} $mm (the files round to 0.01)")


def report(results):
    names = list(results)
    w = max(16, *(len(n) + 2 for n in names))
    tot = {n: [{k: sum(r[k] for r in game) for k in game[0]} for game in results[n]] for n in names}
    line = lambda label, f: print(f"{label:34s}" + "".join(f"{f(n):>{w}}" for n in names))
    print(f"{'':34s}" + "".join(f"{n:>{w}}" for n in names))
    print("example game (real RFQs and positions)")
    for k in ["pnl", "edge", "hedge", "capital", "penalty", "breaches", "filled", "best"]:
        line(f"  {k}", lambda n: f"{tot[n][0][k]:.2f}" if isinstance(tot[n][0][k], float) else f"{tot[n][0][k]}")
    games = len(tot[names[0]]) - 1
    if games:
        print(f"{games} synthetic games, average per game")
        mean = lambda n, k: np.mean([g[k] for g in tot[n][1:]])
        for k in ["pnl", "edge", "hedge", "capital", "penalty", "breaches", "filled", "off_market"]:
            line(f"  {k}", lambda n: f"{mean(n, k):.2f}")
        line("  P&L sd across games", lambda n: f"{np.std([g['pnl'] for g in tot[n][1:]]):.2f}")
        line("  round P&L sd (consistency)", lambda n: f"{np.mean([np.std([r['pnl'] for r in game]) for game in results[n][1:]]):.3f}")
        line("  worst game P&L", lambda n: f"{min(g['pnl'] for g in tot[n][1:]):.2f}")
        if len(names) > 1:
            base = tot[names[-1]][1:]
            line(f"  games beating {names[-1]}", lambda n: "-" if n == names[-1] else
                 f"{sum(a['pnl'] > b['pnl'] for a, b in zip(tot[n][1:], base))}/{games}")


def main():
    global N_RIVALS, RIVAL_SHIFT
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("engines", nargs="*", default=[os.path.join(HERE, "submission", "team.onnx"),
                                                   os.path.join(HERE, "benchmark.onnx")])
    ap.add_argument("--games", type=int, default=20, help="synthetic games after the example game")
    ap.add_argument("--rivals", type=int, default=N_RIVALS, help="number of rival desks")
    ap.add_argument("--shift", type=float, default=RIVAL_SHIFT * 1e4, help="rival aggressiveness, bp")
    ap.add_argument("--check", action="store_true", help="only check the fee maths")
    args = ap.parse_args()
    if args.check:
        return check()
    N_RIVALS, RIVAL_SHIFT = args.rivals, args.shift / 1e4

    rival = Engine(os.path.join(HERE, "benchmark.onnx"))
    engines = [Engine(p) for p in args.engines]
    sizes = np.abs(np.vstack(example_rfqs(np.random.default_rng(0))))[:, [0, 2, 3, 4, 5, 6]]
    results = {e.name: [] for e in engines}
    for g in range(args.games + 1):
        for e in engines:
            rng = np.random.default_rng(g)                  # same market for every engine
            rounds = example_rfqs(rng) if g == 0 else synthetic_rfqs(rng, sizes)
            seat = example_seat() if g == 0 else synthetic_seat(rng)
            seats = [seat] + [synthetic_seat(rng) for _ in range(N_RIVALS)]
            results[e.name].append(play(e, rival, rounds, seats, rng))
    report(results)


if __name__ == "__main__":
    main()
