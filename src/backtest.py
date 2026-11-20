"""Backtest ONNX quoting engines on the real games plus synthetic games.

    python src/backtest.py                            # submission/team.onnx vs the benchmark
    python src/backtest.py a.onnx b.onnx --games 50   # any engines, more synthetic games
    python src/backtest.py --rivals 10 --shift 3      # bigger field, quoting 3bp more aggressively
    python src/backtest.py --sizes 0.5,1,2            # stress: half- and double-size positions too
    python src/backtest.py --check                    # rebuild each real game's fees from its fills

Every engine sits in the same seat of the same market (same RFQs, private positions and
rivals), one run each, so a difference in P&L comes from the engine alone.

Real:     the games in round_0/ (the organisers' example) and Round_1/ (our training game):
          their RFQs and the trader's private positions, and the fee rules (--check matches
          both games' P&L files). Synthetic games copy their structure: each round, one RFQ per
          sign pattern of the 5 risks (32), sizes resampled from the real RFQs; private
          delta/vega on a 12-round sine wave of 15 / 3, the size both real games had.
Assumed:  rivals run benchmark.onnx on their own books plus RIVAL_NOISE (real teams differ more
          than benchmark copies, which tie a third of the time). The best quote gets TOP_FILL of
          the RFQ; other quotes get up to REST_FILL, shrinking to nothing GAP below the best.
          Calibrated by putting benchmark.onnx in the round 0 trader's seat (its files round
          coupons to 2 decimals): it reproduces the real fill count (141 RFQs), best-quote
          count (69 vs 60) and round-by-round pattern (corr 0.92), but edge comes out ~35% low
          and penalties ~65% high. Trust differences between engines more than absolute P&L.
          Round_1 has full-precision coupons, so it can sharpen this calibration.
"""
import argparse
import csv
import os

import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # the repo
BENCHMARK = "round_0/benchmark.onnx"
ROUNDS = {   # real games: quote history, game history, P&L by round (paths from the repo root)
    "round 0": ("round_0/trader_quote_history.csv", "round_0/trader_game_history.csv",
                "round_0/trader_pnl_by_round.csv"),
    "round 1": ("Round_1/trader_quote_history_2.csv", "Round_1/trader_game_history-2.csv",
                "Round_1/trader_pnl_by_round-2.csv"),
}

# ---- fee rules from the handout           delta   vega  vanna  volga    gap
UNIT_COST   = np.array([0.006, 0.025, 0.006, 0.006, 0.006])   # hedge / capital per $mm
LIMIT       = np.array([12.0,  3.0,   3.0,   3.0,   3.0  ])
BREACH_RATE = np.array([0.05,  0.05,  1.0,   1.0,   1.0  ])   # penalty per $mm over the limit
OFF_MARKET  = 0.02                                             # |quote - mid| above this: no fill

# ---- market assumptions (see docstring), calibrated on the round 0 game
N_RIVALS    = 4
TOP_FILL    = 0.25         # round 0's fills: ~21-26% when best, ~2-10% otherwise
REST_FILL   = 0.10
GAP         = 0.0002
RIVAL_NOISE = 0.0004       # sd of random noise on each rival quote
RIVAL_SHIFT = -0.00015     # added to every rival quote; raise it to test a more aggressive field
SEAT_SIZES  = (1.0,)       # private delta/vega swing 15 x size / 3 x size; both real games: 1

RISKS = ["Delta", "Vega", "Exotic Risk 1-Vanna", "Exotic Risk 2-Volga", "Exotic Risk 3-Gap risk"]
SIGNS = np.array([[1 if k >> (4 - i) & 1 else -1 for i in range(5)] for k in range(32)])


def table(path):
    """CSV as {column: [cells]}, with headers and cells stripped and blank rows dropped."""
    with open(os.path.join(ROOT, path), newline="") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f)]
    body = [r for r in rows[1:] if r and r[0]]
    return {h: [r[i] for r in body] for i, h in enumerate(rows[0]) if h}


def num(cells):
    return np.array([0.0 if c in ("", "-") else float(c) for c in cells])


def real_rfqs(rng, quotes):
    """A real game's RFQs per round: 32 x [notional, mid, 5 risks]. If the file rounds mids to
    2 decimals (round 0 does), each is redrawn inside its rounding interval."""
    q = table(quotes)
    X = np.c_[tuple(num(q[c]) for c in ["RFQ|Notional", "RFQ|MidValue"] + ["RFQ|" + r for r in RISKS])]
    if np.all(np.round(X[:, 1], 2) == X[:, 1]):
        X[:, 1] = rng.uniform(np.maximum(X[:, 1] - 0.005, 0.047), np.minimum(X[:, 1] + 0.005, 0.058))
    rnd = num(q["RFQ|Round"])
    return [X[rnd == t] for t in range(1, 13)]


def real_seat(history):
    """A real game's 12 private positions (its R01..R12 rows) and the trader's name."""
    g = table(history)
    private = np.array([i.startswith("R") for i in g["RFQ ID"]])
    return np.c_[tuple(num(g[c]) for c in RISKS)][private], g["Trader"][0]


def rfq_sizes():
    """Real RFQ sizes, [notional, |5 risks|], from every real game, to resample from."""
    rng = np.random.default_rng(0)
    rounds = [r for quotes, _, _ in ROUNDS.values() for r in real_rfqs(rng, quotes)]
    return np.abs(np.vstack(rounds))[:, [0, 2, 3, 4, 5, 6]]


def synthetic_rfqs(rng, sizes):
    """12 rounds like the real ones: one RFQ per sign pattern, sizes drawn from real RFQs."""
    rounds = []
    for _ in range(12):
        s = sizes[rng.integers(len(sizes), size=32)]
        rounds.append(np.c_[s[:, 0], rng.uniform(0.047, 0.058, 32), SIGNS * s[:, 1:]])
    return rounds


def synthetic_seat(rng):
    """12 private positions shaped like the real games': delta/vega on a sine wave of 15 / 3
    times a size from SEAT_SIZES, an inherited exotic book in round 1, then small moves."""
    size = rng.choice(SEAT_SIZES)
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
    """Rebuild each real game's fees from the trader's own fills; compare with its P&L file."""
    for label, (_, history, pnl_file) in ROUNDS.items():
        g, p = table(history), table(pnl_file)
        rnd, risk = num(g["Round"]), np.c_[tuple(num(g[c]) for c in RISKS)]
        exo, worst = np.zeros(3), 0.0
        print(f"{label} ({g['Trader'][0]})\nround   costs ours/file   penalty ours/file   breaches ours/file   P&L ours/file")
        for t in range(1, 13):
            took = risk[rnd == t].sum(0)                    # private position + fills
            exo += took[2:]
            hedge, capital, penalty, breaches = fees(np.r_[took[:2], exo])
            f = {k: num(p[k])[t - 1] for k in ["Costs", "Aggregate Penalty", "Breaches", "Edge", "PnL"]}
            costs, pnl = hedge + capital + penalty, f["Edge"] - (hedge + capital + penalty)
            worst = max(worst, abs(costs - f["Costs"]), abs(penalty - f["Aggregate Penalty"]), abs(pnl - f["PnL"]))
            print(f"{t:5d}   {costs:6.2f} / {f['Costs']:5.2f}    {penalty:6.2f} / {f['Aggregate Penalty']:5.2f}"
                  f"    {breaches:8d} / {f['Breaches']:2.0f}       {pnl:6.2f} / {f['PnL']:5.2f}")
        print(f"largest difference: {worst:.3g} $mm\n")


def report(results, labels):
    """The first len(labels) games are the real ones; the rest are synthetic."""
    names, k = list(results), len(labels)
    w = max(16, *(len(n) + 2 for n in names))
    tot = {n: [{key: sum(r[key] for r in game) for key in game[0]} for game in results[n]] for n in names}
    line = lambda label, f: print(f"{label:34s}" + "".join(f"{f(n):>{w}}" for n in names))
    print(f"{'':34s}" + "".join(f"{n:>{w}}" for n in names))
    for i, label in enumerate(labels):
        print(label)
        for key in ["pnl", "edge", "hedge", "capital", "penalty", "breaches", "filled", "best"]:
            line(f"  {key}", lambda n: f"{tot[n][i][key]:.2f}" if isinstance(tot[n][i][key], float) else f"{tot[n][i][key]}")
    games = len(tot[names[0]]) - k
    if games:
        print(f"{games} synthetic games, average per game")
        mean = lambda n, key: np.mean([g[key] for g in tot[n][k:]])
        for key in ["pnl", "edge", "hedge", "capital", "penalty", "breaches", "filled", "off_market"]:
            line(f"  {key}", lambda n: f"{mean(n, key):.2f}")
        line("  P&L sd across games", lambda n: f"{np.std([g['pnl'] for g in tot[n][k:]]):.2f}")
        line("  round P&L sd (consistency)", lambda n: f"{np.mean([np.std([r['pnl'] for r in game]) for game in results[n][k:]]):.3f}")
        line("  worst game P&L", lambda n: f"{min(g['pnl'] for g in tot[n][k:]):.2f}")
        if len(names) > 1:
            base = tot[names[-1]][k:]
            line(f"  games beating {names[-1]}", lambda n: "-" if n == names[-1] else
                 f"{sum(a['pnl'] > b['pnl'] for a, b in zip(tot[n][k:], base))}/{games}")


def rounds_table(results, labels):
    """Each real game round by round, one block per engine ($mm)."""
    for name, games in results.items():
        for label, game in zip(labels, games):
            print(f"\n{name}, {label}, $mm")
            print("round    edge   hedge  capital  penalty     P&L  cumulative")
            cum = 0.0
            for t, r in enumerate(game, 1):
                cum += r["pnl"]
                print(f"{t:5d} {r['edge']:7.2f} {r['hedge']:7.2f} {r['capital']:8.2f} {r['penalty']:8.2f}"
                      f" {r['pnl']:7.2f} {cum:11.2f}")


def main():
    global N_RIVALS, RIVAL_SHIFT, SEAT_SIZES
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("engines", nargs="*", default=[os.path.join(ROOT, "submission", "team.onnx"),
                                                   os.path.join(ROOT, BENCHMARK)])
    ap.add_argument("--games", type=int, default=20, help="synthetic games after the real games")
    ap.add_argument("--rivals", type=int, default=N_RIVALS, help="number of rival desks")
    ap.add_argument("--shift", type=float, default=RIVAL_SHIFT * 1e4, help="rival aggressiveness, bp")
    ap.add_argument("--sizes", default="1", help="private position sizes to draw from, e.g. 0.5,1,2")
    ap.add_argument("--rounds", action="store_true", help="also show the real games round by round")
    ap.add_argument("--check", action="store_true", help="only check the fee maths")
    args = ap.parse_args()
    if args.check:
        return check()
    N_RIVALS, RIVAL_SHIFT = args.rivals, args.shift / 1e4
    SEAT_SIZES = tuple(float(s) for s in args.sizes.split(","))

    rival = Engine(os.path.join(ROOT, BENCHMARK))
    engines = [Engine(p) for p in args.engines]
    sizes = rfq_sizes()
    reals = [(files[0], *real_seat(files[1])) for files in ROUNDS.values()]
    labels = [f"{label}: real game ({trader}'s RFQs and positions)"
              for label, (_, _, trader) in zip(ROUNDS, reals)]
    results = {e.name: [] for e in engines}
    for g in range(len(reals) + args.games):
        for e in engines:
            rng = np.random.default_rng(g)                  # same market for every engine
            if g < len(reals):
                rounds, seat = real_rfqs(rng, reals[g][0]), reals[g][1]
            else:
                rounds = synthetic_rfqs(rng, sizes)
                seat = synthetic_seat(rng)
            seats = [seat] + [synthetic_seat(rng) for _ in range(N_RIVALS)]
            results[e.name].append(play(e, rival, rounds, seats, rng))
    report(results, labels)
    if args.rounds:
        rounds_table(results, labels)


if __name__ == "__main__":
    main()
