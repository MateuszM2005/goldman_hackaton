"""Market simulator built from the reverse-engineered RFQ generator (standalone: does not
import backtest.py or engine.py, which are being reworked separately).

    python src/sim_center.py                   # center vs old tuned vs round-1 engine vs benchmark
    python src/sim_center.py a.onnx b.onnx     # any ONNX engines (never-trade is always added)

Each engine plays identical games: the 3 real games (their real RFQs and private positions,
in 10 simulated markets), synthetic games by private-path size (0.5 calm, 1 like games 0-1,
2 like game 2), and a market whose rivals all quote 5bp more aggressively.

What is known (checked on the test sample and games 0-2):
  RFQs      32 a round, binary order: delta sign, vega sign, then the 8 exotic cube corners.
            |delta| = 4*U(0.8,1.2), |vega| = U(0.8,1.2), notional a whole number on 90..109,
            mid ~ U(0.048, 0.058). Exotics = (R . corner) * U(0.9,1.1) per component, R one
            proper rotation per round: identity (games 1, 2), a slow drift (example game, ~2
            deg/round) or a fixed turn (test-sample blocks). Synthetic games draw any of these.
  Seats     private delta/vega = (-15 sin th, 3 cos th), th +30 deg a round from a random start.
            Private exotics: one 12-round path from the catalogue hidden in TestPredict.csv's
            books (36 distinct paths: 12 shapes x sizes 0.5 / 1 / 2; game 2 had a size-2 path
            with shocks up to 5 and a -5 inherited book). The real games' three paths are held
            out of tuning (HOLDOUT) and replayed as the validation set.
  Fees      the handout's, matched to the P&L files to 1e-16: delta/vega hedged at the end of
            each round; exotics stay, 0.006 capital and 100% of any excess over 3, every round.
Assumed (fitted to rounds 1 and 2 by src/calibrate.py; snapshot of its result):
  14 rival desks quoting benchmark.onnx on their own books, plus a lasting style and noise,
  1.5% wild quotes up to +0.02; fills by rank 55% / 24.5% / 14.6% / 4.5% / 1.4% of the RFQ.
"""
import csv
import os
import sys

import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK = os.path.join(ROOT, "round_0", "benchmark.onnx")
RISKS = ["Delta", "Vega", "Exotic Risk 1-Vanna", "Exotic Risk 2-Volga", "Exotic Risk 3-Gap risk"]
REAL = {   # quote history, game history, catalogue run holding that game's private exotic path
    "game 0 (example)": ("round_0/trader_quote_history.csv", "round_0/trader_game_history.csv", 16),
    "game 1": ("Round_1/trader_quote_history_2.csv", "Round_1/trader_game_history-2.csv", 18),
    "game 2": ("submission_game_2/Round_2/trader_quote_history.csv",
               "submission_game_2/Round_2/trader_game_history-2.csv", 32),
}

# ---- fees                                 delta   vega  vanna  volga    gap
UNIT_COST   = np.array([0.006, 0.025, 0.006, 0.006, 0.006])
LIMIT       = np.array([12.0,  3.0,   3.0,   3.0,   3.0  ])
BREACH_RATE = np.array([0.05,  0.05,  1.0,   1.0,   1.0  ])
OFF_MARKET  = 0.02
PRIVATE_EDGE = 0.1         # $mm a round, the same for every engine

# ---- market (calibrate.py on rounds 1 + 2)
MARKET = dict(N_RIVALS=14, RIVAL_SHIFT=-0.00013, TEAM_SPREAD=0.00135, RIVAL_NOISE=0.00053,
              P_WILD=0.015, WILD_LO=0.0032)
FILL_BY_RANK = np.array([0.550, 0.245, 0.146, 0.045, 0.014])

CORNERS = np.array([[1 if k >> (2 - i) & 1 else -1 for i in range(3)] for k in range(8)], float)
DV_SIGNS = np.repeat([[-1, -1], [-1, 1], [1, -1], [1, 1]], 8, axis=0).astype(float)


# ---- data --------------------------------------------------------------------------
def _table(path):
    with open(os.path.join(ROOT, path), newline="") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f)]
    body = [r for r in rows[1:] if r and r[0]]
    return {h: [r[i] for r in body] for i, h in enumerate(rows[0]) if h}


def _num(cells):
    return np.array([0.0 if c in ("", "-") else float(c) for c in cells])


_PATHS = None


def exotic_paths():
    """36 distinct 12-round private exotic paths (36 x 12 x 3) and the catalogue run -> path map."""
    global _PATHS
    if _PATHS is None:
        with open(os.path.join(ROOT, "round_0", "TestPredict.csv"), newline="") as f:
            rows = list(csv.reader(f))
        runs = np.array([[float(c) for c in r[2:5]] for r in rows[1::96]]).reshape(108, 12, 3)
        paths, index = [], []
        for run in runs:
            hit = [i for i, p in enumerate(paths) if np.abs(p - run).max() < 1e-9]
            if not hit:
                paths.append(run)
                hit = [len(paths) - 1]
            index.append(hit[0])
        _PATHS = (np.array(paths), np.array(index))
    return _PATHS


def holdout():
    """Paths of the real games: never used in synthetic games, so they stay a fair test."""
    _, index = exotic_paths()
    return sorted({int(index[run]) for _, _, run in REAL.values()})


def real_game(key, rng):
    """A real game's 12 rounds of RFQs (32 x [notional, mid, 5 risks]) and its private seat."""
    quotes, history, _ = REAL[key]
    q = _table(quotes)
    X = np.c_[tuple(_num(q[c]) for c in ["RFQ|Notional", "RFQ|MidValue"] + ["RFQ|" + r for r in RISKS])]
    if np.all(np.round(X[:, 1], 2) == X[:, 1]):                  # game 0's file rounds mids
        X[:, 1] = rng.uniform(np.maximum(X[:, 1] - 0.005, 0.048), np.minimum(X[:, 1] + 0.005, 0.058))
    rnd = _num(q["RFQ|Round"])
    g = _table(history)
    private = np.array([i.startswith("R") for i in g["RFQ ID"]])
    seat = np.c_[tuple(_num(g[c]) for c in RISKS)][private]
    return [X[rnd == t] for t in range(1, 13)], seat


# ---- synthetic games -----------------------------------------------------------------
def _rot(axis, deg):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    a = np.radians(deg)
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def rotations(rng):
    """12 cube rotations for one game: unturned, drifting or a fixed turn (all seen in the data)."""
    kind = rng.choice(3, p=[0.4, 0.3, 0.3])
    if kind == 0:
        return [np.eye(3)] * 12
    axis = rng.standard_normal(3)
    if kind == 1:
        start, step = rng.uniform(-30, 30), rng.uniform(-4, 4)
        return [_rot(axis, start + step * t) for t in range(1, 13)]
    return [_rot(axis, rng.uniform(-120, 120))] * 12


def synthetic_rfqs(rng):
    out = []
    for R in rotations(rng):
        ex = (np.tile(CORNERS, (4, 1)) @ R.T) * rng.uniform(0.9, 1.1, (32, 3))
        d = DV_SIGNS[:, 0] * 4 * rng.uniform(0.8, 1.2, 32)
        v = DV_SIGNS[:, 1] * rng.uniform(0.8, 1.2, 32)
        out.append(np.c_[rng.integers(90, 110, 32), rng.uniform(0.048, 0.058, 32), d, v, ex])
    return out


def synthetic_seat(rng, allowed=None):
    """Private positions: delta/vega on the 15 / 3 sine, exotics one catalogue path."""
    paths, _ = exotic_paths()
    if allowed is None:
        allowed = [i for i in range(len(paths)) if i not in holdout()]
    th = rng.uniform(0, 2 * np.pi) + np.arange(1, 13) * np.pi / 6
    return np.c_[-15 * np.sin(th), 3 * np.cos(th), paths[rng.choice(allowed)]]


# ---- engines -------------------------------------------------------------------------
class Onnx:
    def __init__(self, path, name=None):
        self.name = name or os.path.basename(path)
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name

    def quote_many(self, books, rnd, rfqs):
        X = np.c_[np.repeat(books, len(rfqs), 0), np.full(len(books) * len(rfqs), rnd),
                  np.tile(rfqs, (len(books), 1))].astype(np.float32)
        return self.sess.run(None, {self.input: X})[0].reshape(len(books), len(rfqs)).astype(float)

    def quote(self, book, rnd, rfqs):
        return self.quote_many(np.atleast_2d(book), rnd, rfqs)[0]


class NeverTrade:
    name = "never trade"

    def quote(self, book, rnd, rfqs):
        return rfqs[:, 1] - 0.0195


def fees(pos):
    excess = np.maximum(np.abs(pos) - LIMIT, 0)
    return (UNIT_COST[:2] @ np.abs(pos[:2]), UNIT_COST[2:] @ np.abs(pos[2:]),
            BREACH_RATE @ excess, int((excess > 0).sum()))


def play(engine, rival, rounds, seats, rng, market=MARKET):
    """One game, `engine` in seat 0, `rival` (benchmark) in the rest. Returns seat 0's rounds."""
    n = len(seats)
    exo = np.zeros((n, 3))
    style = market["RIVAL_SHIFT"] + market["TEAM_SPREAD"] * rng.standard_normal((n - 1, 1))
    out = []
    for t, rfqs in enumerate(rounds, 1):
        mid = rfqs[:, 1]
        exo += np.array([s[t - 1, 2:] for s in seats])
        books = np.c_[np.array([s[t - 1, :2] for s in seats]), exo]
        quotes = np.vstack([engine.quote(books[0], t, rfqs), rival.quote_many(books[1:], t, rfqs)])
        quotes[1:] += style + market["RIVAL_NOISE"] * rng.standard_normal(quotes[1:].shape)
        wild = rng.random(quotes[1:].shape) < market["P_WILD"]
        quotes[1:] = np.where(wild, mid + rng.uniform(market["WILD_LO"], OFF_MARKET - 1e-4, wild.shape), quotes[1:])
        ok = np.isfinite(quotes) & (np.abs(quotes - mid) <= OFF_MARKET)
        q = np.where(ok, quotes, -np.inf)
        rank = 1 + (q[None, :, :] > q[:, None, :]).sum(1)
        top = len(FILL_BY_RANK)
        fill = np.where(ok & (rank <= top), FILL_BY_RANK[np.minimum(rank, top) - 1], 0.0)
        took = np.einsum("sr,rk->sk", fill, rfqs[:, 2:])
        exo += took[:, 2:]
        pos = np.c_[np.array([s[t - 1, :2] for s in seats]) + took[:, :2], exo]
        edge = (fill[0] * rfqs[:, 0] * (mid - np.where(ok[0], quotes[0], 0))).sum() + PRIVATE_EDGE
        hedge, capital, penalty, breaches = fees(pos[0])
        out.append(dict(pnl=edge - hedge - capital - penalty, edge=edge, hedge=hedge, capital=capital,
                        penalty=penalty, breaches=breaches, filled=int((fill[0] > 0).sum()),
                        exo=exo[0].copy()))
    return out


def random_market(rng, aggressive=0.0):
    """A market around the calibrated one; `aggressive` (bp) shifts every rival up."""
    m = dict(MARKET)
    m["N_RIVALS"] = max(3, MARKET["N_RIVALS"] + int(rng.integers(-5, 6)))
    m["RIVAL_SHIFT"] = MARKET["RIVAL_SHIFT"] + rng.uniform(-0.0005, 0.0005) + aggressive / 1e4
    m["TEAM_SPREAD"] = MARKET["TEAM_SPREAD"] * rng.uniform(0.7, 1.3)
    m["RIVAL_NOISE"] = MARKET["RIVAL_NOISE"] * rng.uniform(0.5, 1.5)
    m["P_WILD"] = MARKET["P_WILD"] * rng.uniform(0.5, 1.5)
    return m


def synthetic_game(engine, rival, seed, aggressive=0.0, paths=None):
    rng = np.random.default_rng(seed)
    m = random_market(rng, aggressive)
    rounds = synthetic_rfqs(rng)
    seats = [synthetic_seat(rng, paths)] + [synthetic_seat(rng) for _ in range(m["N_RIVALS"])]
    return play(engine, rival, rounds, seats, rng, m)


def real_replay(engine, rival, key, seed):
    rng = np.random.default_rng(seed)
    rounds, seat = real_game(key, rng)
    seats = [seat] + [synthetic_seat(rng) for _ in range(MARKET["N_RIVALS"])]
    return play(engine, rival, rounds, seats, rng)


def total(rows):
    keys = ["pnl", "edge", "hedge", "capital", "penalty", "breaches", "filled"]
    return {k: sum(r[k] for r in rows) for k in keys} | {"round_sd": float(np.std([r["pnl"] for r in rows]))}


def path_sizes():
    """Size of each catalogue path: 0.5, 1 or 2 (its largest single move is ~1.3, ~2.5 or ~5)."""
    amp = np.abs(exotic_paths()[0]).max(axis=(1, 2))
    return np.where(amp < 1.8, 0.5, np.where(amp < 3.5, 1.0, 2.0))


def report(engines, games=60, markets=10):
    """Every engine on identical games: the real games, synthetic games by private-path size,
    and a market whose rivals all quote 5bp more aggressively."""
    rival = Onnx(BENCHMARK)
    names = [e.name for e in engines]
    w = max(15, *(len(n) + 2 for n in names))
    head = lambda title: print(f"\n{title}\n{'':24s}" + "".join(f"{n:>{w}}" for n in names), flush=True)
    row = lambda label, vals, fmt: print(f"{label:24s}" + "".join(f"{fmt.format(v):>{w}}" for v in vals))

    for key in REAL:
        T = [[total(real_replay(e, rival, key, 7_000 + d)) for d in range(markets)] for e in engines]
        head(f"{key}: its real RFQs and private positions, mean of {markets} simulated markets ($mm)")
        for k in ["pnl", "edge", "hedge", "capital", "penalty", "breaches", "filled"]:
            row(f"  {k}", [np.mean([t[k] for t in Ts]) for Ts in T],
                "{:.1f}" if k in ("breaches", "filled") else "{:+.2f}")

    sizes, hold = path_sizes(), holdout()
    sets = [(f"synthetic, private paths of size {s:g}", 0.0, [i for i in range(len(sizes)) if sizes[i] == s and i not in hold])
            for s in (0.5, 1.0, 2.0)]
    sets.append(("synthetic, all sizes, rivals +5bp more aggressive", 5.0, None))
    for n, (title, aggressive, allowed) in enumerate(sets):
        T = [[total(synthetic_game(e, rival, 90_000 + 1_000 * n + s, aggressive, allowed)) for s in range(games)]
             for e in engines]
        pnl = [np.array([t["pnl"] for t in Ts]) for Ts in T]
        head(f"{title} ({games} games, $mm per game)")
        row("  P&L", [p.mean() for p in pnl], "{:+.2f}")
        row("  standard error", [p.std() / np.sqrt(len(p)) for p in pnl], "{:.2f}")
        row("  worst game", [p.min() for p in pnl], "{:+.2f}")
        row("  games with P&L > 0", [np.mean(p > 0) for p in pnl], "{:.0%}")
        for k in ["edge", "penalty", "breaches", "round_sd"]:
            row(f"  {k}", [np.mean([t[k] for t in Ts]) for Ts in T], "{:.1f}" if k == "breaches" else "{:+.2f}")


if __name__ == "__main__":
    paths = sys.argv[1:] or [os.path.join(ROOT, "submission_center", "Shame Street.onnx"),
                             os.path.join(ROOT, "submission_game_2", "Shame Street.onnx"),
                             os.path.join(ROOT, "models", "Shame Street.onnx"), BENCHMARK]
    labels = {os.path.join(ROOT, "submission_center", "Shame Street.onnx"): "center (new)",
              os.path.join(ROOT, "submission_game_2", "Shame Street.onnx"): "old tuned",
              os.path.join(ROOT, "submission_cube", "team.onnx"): "cube",
              os.path.join(ROOT, "models", "Shame Street.onnx"): "round-1 engine", BENCHMARK: "benchmark"}
    labels = {os.path.normcase(os.path.abspath(k)): v for k, v in labels.items()}
    report([Onnx(p, labels.get(os.path.normcase(os.path.abspath(p)))) for p in paths] + [NeverTrade()])
