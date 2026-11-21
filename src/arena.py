"""Arena: a data-driven market simulator for backtesting quoting engines (.onnx).

    python src/arena.py                                   # submission/team.onnx vs benchmark, synthetic games
    python src/arena.py a.onnx b.onnx --games 200         # any engines; the last one is the reference
    python src/arena.py a.onnx --opponents team           # our team's models (+ benchmark) join the field
    python src/arena.py a.onnx b.onnx --tournament        # all engines in the SAME games, head to head
    python src/arena.py a.onnx --replay                   # real games 0-2 with the competition we observed
    python src/arena.py --check                           # fee maths + replay of our real quotes = official P&L
    python src/arena.py --fit                             # refit the rival field to games 1 and 2

Separate from backtest.py on purpose: nothing here imports it, so the two can change independently.

What is measured from data (round_0/, Round_1/, submission_game_2/Round_2/, the test sample):
  RFQs      the organisers reuse ONE set of RFQ draws: games 1 and 2 are identical in every field,
            game 0 is the same draws with the exotic cube drifting 2.058 deg/round. A round is 32
            RFQs in binary sign order (delta, vega, then the 8 cube corners); exotics are
            (R corner) x u with u ~ U(0.9, 1.1) drawn per component AFTER the turn R.
            Default: those real draws under a random turn (none / drift / fixed). --draws fresh
            draws new ones from the same distributions (notional 90..109, mid U(0.048, 0.058), ...).
  Seats     private delta/vega = size x (-15 sin a, 3 cos a), a +30 deg/round (real games: size 1);
            private exotics = one of the 36 catalogue paths in round_0/TestPredict.csv (sizes
            0.5 / 1 / 2; game 2 drew a size-2 path, moves up to 5 a round). Every desk gets its own:
            rivals do NOT share our book (game 2: on trades that helped our book the best rival
            quote had median +5bp, never above +50bp).
  Fills     rank-based, resampled from our 212 real fills: first place 37-74% (mean 55%), second
            19-26% (mean 25%), lower places 2-15%. Ranks below K get nothing (K = 5 makes the
            average shares add up to ~100%).
  Field     rival desks with their own books, quoting  mid + lean - RISK_W x cost(25% of the
            trade) / notional + noise,  capped at mid + cap. Some desks are "offloaders": when a
            trade cuts their exotic excess over a trigger they bid far above mid, as seen in game 1
            (one desk bid +106..+126bp on every vanna-positive RFQ of round 7). One field is fitted
            per real game to what happened to our 384 real quotes there (--fit; arena_field.json):
            game 1's rivals offloaded a lot (27% of our fills faced a rival above +40bp), game 2's
            hardly (2%). Synthetic games draw either regime, jittered (--jitter).
  Replay    games 1 and 2 at full precision: the rival quotes are drawn from the fitted field but
            CONDITIONED on what we observed (best and second coupons where we filled, "at least K
            rivals above us" where we did not). Replaying our real quotes reproduces the official P&L
            exactly; another engine is scored against the same competition.
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIELD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arena_field.json")
BENCHMARK = os.path.join(ROOT, "round_0", "benchmark.onnx")
GAMES = {   # quote history, game history, P&L by round
    0: ("round_0/trader_quote_history.csv", "round_0/trader_game_history.csv", "round_0/trader_pnl_by_round.csv"),
    1: ("Round_1/trader_quote_history_2.csv", "Round_1/trader_game_history-2.csv", "Round_1/trader_pnl_by_round-2.csv"),
    2: ("submission_game_2/Round_2/trader_quote_history.csv", "submission_game_2/Round_2/trader_game_history-2.csv",
        "submission_game_2/Round_2/trader_pnl_by_round-2.csv"),
}
RISKS = ["Delta", "Vega", "Exotic Risk 1-Vanna", "Exotic Risk 2-Volga", "Exotic Risk 3-Gap risk"]
SIGNS = np.array([[1 if k >> (4 - i) & 1 else -1 for i in range(5)] for k in range(32)])   # RFQ k's signs

# ---- fee rules (handout; reproduce all three official P&L files)   delta  vega  vanna  volga  gap
UNIT_COST    = np.array([0.006, 0.025, 0.006, 0.006, 0.006])
LIMIT        = np.array([12.0, 3.0, 3.0, 3.0, 3.0])
BREACH_RATE  = np.array([0.05, 0.05, 1.0, 1.0, 1.0])
OFF_MARKET   = 0.02          # |quote - mid| above this: off-market, no fill
PRIVATE_EDGE = 0.1           # the private position is booked at mid - 10bp on $100mm: $0.1mm a round
DRIFT_AXIS, DRIFT_RATE = (-0.0799, 0.8227, -0.5629), -2.058   # game 0's cube drift (deg per round)

# ---- the rival field. Overwritten by arena_field.json when it exists (python src/arena.py --fit)
FIELD = dict(
    n_desks=14,          # rival desks besides the seats under test
    lean_mu=-0.0004,     # each desk's lasting lean vs mid (coupon units; 0.0001 = 1bp)...
    lean_sd=0.0008,      # ...and its spread across desks
    cap_lo=0.0003,       # each desk never pays more than mid + cap, cap ~ U(cap_lo, cap_hi)...
    cap_hi=0.0025,       # ...unless it is offloading
    noise=0.0004,        # per-quote noise
    risk_w=0.5,          # how much of the trade's fee cost a desk prices in
    p_off=0.3,           # share of desks that offload when over their trigger...
    trigger=2.5,         # ...exotic level that makes them offload
    off_lo=0.005,        # offload bids: mid + U(off_lo, off_hi), fixed per desk
    off_hi=0.0199,
    p_wild=0.005,        # stray quotes anywhere in mid + (0, 0.0199)
    K=5,                 # ranks below K get nothing
)
FIELDS = {}          # one fitted field per real game: {"game 1": {...}, "game 2": {...}}
if os.path.exists(FIELD_FILE):
    with open(FIELD_FILE) as f:
        _saved = json.load(f)
    if "fields" in _saved:
        FIELDS = {k: dict(FIELD, **{kk: vv for kk, vv in v.items() if kk in FIELD}) for k, v in _saved["fields"].items()}
        FIELD = dict(FIELDS[sorted(FIELDS)[-1]])          # default: the latest game's field
    else:
        FIELD.update({k: v for k, v in _saved.items() if k in FIELD})


def field_for(k):
    """The field fitted to real game k (or the default field)."""
    return FIELDS.get(f"game {k}", FIELD)


def regimes():
    """Rival fields synthetic games draw from: one per fitted real game."""
    return (list(FIELDS), list(FIELDS.values())) if FIELDS else (["default"], [FIELD])
PHI, HORIZON = 0.25, 3       # rivals price a 25% fill; a cycle-driven breach lasts ~3 rounds


# =====================================================================================  data
def table(path):
    with open(os.path.join(ROOT, path), newline="") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f)]
    body = [r for r in rows[1:] if r and r[0]]
    return {h: [r[i] for r in body] for i, h in enumerate(rows[0]) if h}


def num(cells):
    return np.array([0.0 if c in ("", "-") else float(c) for c in cells])


def rod(axis, deg):
    """Rotation by deg about axis (Rodrigues)."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    r = np.radians(deg)
    return np.eye(3) + np.sin(r) * K + (1 - np.cos(r)) * K @ K


_CACHE = {}


def base_draws():
    """The organisers' fixed RFQ draws, from game 1 (unturned cube, full precision):
    notional, mid, |delta|, |vega| and the exotic multipliers u, each (12, 32)."""
    if "draws" not in _CACHE:
        q = table(GAMES[1][0])
        X = np.c_[tuple(num(q["RFQ|" + c]) for c in ["Notional", "MidValue"] + RISKS)].reshape(12, 32, 7)
        _CACHE["draws"] = dict(N=X[..., 0], mid=X[..., 1], d=np.abs(X[..., 2]), v=np.abs(X[..., 3]),
                               u=X[..., 4:] * SIGNS[None, :, 2:])
    return _CACHE["draws"]


def catalogue():
    """The 36 private exotic paths (12 rounds x vanna/volga/gap moves): the distinct books of
    round_0/TestPredict.csv in order are 108 runs of 12 rounds, each path at 3 start angles."""
    if "cat" not in _CACHE:
        X = np.loadtxt(os.path.join(ROOT, "round_0", "TestPredict.csv"), delimiter=",", skiprows=1, usecols=range(5))
        _, first = np.unique(X, axis=0, return_index=True)
        runs = X[np.sort(first)].reshape(-1, 12, 5)[:, :, 2:]
        _CACHE["cat"] = np.unique(np.round(runs, 6), axis=0)
    return _CACHE["cat"]


def rfq_rounds(draws, rotations):
    """12 x (32 x 7) RFQ arrays [notional, mid, delta, vega, vanna, volga, gap]."""
    out = []
    for t in range(12):
        ex = (SIGNS[:, 2:] @ rotations[t].T) * draws["u"][t]
        out.append(np.c_[draws["N"][t], draws["mid"][t], SIGNS[:, 0] * draws["d"][t], SIGNS[:, 1] * draws["v"][t], ex])
    return out


def random_rotation(rng):
    q = rng.standard_normal(4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def make_rfqs(rng, draws="shared", turn="mix"):
    """A game's RFQs. turn: none (games 1, 2), drift (game 0: a few deg/round about a random axis),
    fixed (one random turn, like the test sample's 60/120 deg blocks), or mix of the three."""
    if turn == "mix":
        turn = rng.choice(["none", "drift", "fixed"], p=[0.45, 0.3, 0.25])
    if turn == "none":
        rots = [np.eye(3)] * 12
    elif turn == "drift":
        axis, rate = rng.standard_normal(3), rng.uniform(1.0, 3.5) * rng.choice([-1, 1])
        rots = [rod(axis, rate * t) for t in range(1, 13)]
    else:
        R = random_rotation(rng)
        rots = [R] * 12
    if draws == "shared":
        d = base_draws()
    else:
        d = dict(N=rng.integers(90, 110, (12, 32)).astype(float), mid=rng.uniform(0.048, 0.058, (12, 32)),
                 d=4 * rng.uniform(0.8, 1.2, (12, 32)), v=rng.uniform(0.8, 1.2, (12, 32)),
                 u=rng.uniform(0.9, 1.1, (12, 32, 3)))
    return rfq_rounds(d, rots), turn


DV_SIZES = {1.0: 0.6, 0.5: 0.2, 2.0: 0.2}    # real games: 1, 1, 1; the test sample uses all three


def make_seat(rng, path=None, dv_size=None):
    """12 private moves [delta, vega, vanna, volga, gap]; returns (seat, path index)."""
    cat = catalogue()
    path = int(rng.integers(len(cat))) if path is None else path
    size = rng.choice(list(DV_SIZES), p=list(DV_SIZES.values())) if dv_size is None else dv_size
    a = rng.uniform(0, 2 * np.pi) + np.arange(12) * np.pi / 6
    return np.c_[-15 * size * np.sin(a), 3 * size * np.cos(a), cat[path]], path


def path_size(path):
    m = np.abs(catalogue()[path]).max()
    return 0.5 if m < 2 else (1.0 if m < 4 else 2.0)


class RealGame:
    """One real game: RFQs (full precision), our seat's private moves, our quotes and outcomes."""
    def __init__(self, k):
        qf, gf, pf = GAMES[k]
        q, g = table(qf), table(gf)
        self.k = k
        self.trader = g["Trader"][0]
        self.book = np.c_[tuple(num(q["Book|" + r]) for r in RISKS)].reshape(12, 32, 5)[:, 0]
        self.mine = num(q["Quoted_coupon"]).reshape(12, 32)
        self.precise = not np.all(np.round(self.mine, 2) == self.mine)
        if k == 0:     # 2-decimal file: rebuild exact RFQs from the shared draws + game 0's drift
            self.rfqs = rfq_rounds(base_draws(), [rod(DRIFT_AXIS, DRIFT_RATE * t) for t in range(1, 13)])
        else:
            X = np.c_[tuple(num(q["RFQ|" + c]) for c in ["Notional", "MidValue"] + RISKS)].reshape(12, 32, 7)
            self.rfqs = list(X)
        ids = q["RFQ ID"]
        at = {rid: i for i, rid in enumerate(ids)}
        self.fill = np.zeros(384)
        self.best, self.second = np.full(384, np.nan), np.full(384, np.nan)
        seat = []
        for i, rid in enumerate(g["RFQ ID"]):
            if rid in at:
                j = at[rid]
                self.fill[j] = float(g["Notional"][i]) / float(q["RFQ|Notional"][j])
                self.best[j], self.second[j] = float(g["Best_Quoted_coupon"][i]), float(g["Second_Quoted_coupon"][i])
            else:
                seat.append([float(g[c][i]) for c in RISKS])
        self.seat = np.array(seat)
        self.fill, self.best, self.second = (a.reshape(12, 32) for a in (self.fill, self.best, self.second))
        self.pnl = table(pf)


# =====================================================================================  fills
class Shares:
    """Fill share by rank, resampled from our real fills (games 1 and 2): rank 1 from the 110
    first places, rank 2 from the 20 second places, ranks 3..K from the 82 lower places
    (K-2 draws, largest to rank 3). Ranks below K get nothing."""
    def __init__(self, field):
        if "shares" not in _CACHE:
            r1, r2, low = [], [], []
            for k in (1, 2):
                g = RealGame(k)
                F = g.fill > 0
                first = F & np.isclose(g.mine, g.best)
                second = F & ~first & np.isclose(g.mine, g.second)
                r1 += list(g.fill[first]); r2 += list(g.fill[second]); low += list(g.fill[F & ~first & ~second])
            _CACHE["shares"] = (np.array(r1), np.array(r2), np.array(low))
        self.r1, self.r2, self.low = _CACHE["shares"]
        self.K = int(field["K"])

    def draw(self, rng, shape):
        """shape + (K,) shares for ranks 1..K."""
        out = np.empty(shape + (self.K,))
        out[..., 0] = rng.choice(self.r1, shape)
        out[..., 1] = rng.choice(self.r2, shape)
        if self.K > 2:
            out[..., 2:] = -np.sort(-rng.choice(self.low, shape + (self.K - 2,)), -1)
        return out


# =====================================================================================  engines
class Onnx:
    def __init__(self, path):
        import onnxruntime as ort
        o = ort.SessionOptions()
        o.intra_op_num_threads, o.inter_op_num_threads = 1, 1
        self.path, self.name = path, os.path.splitext(os.path.basename(path))[0]
        self.sess = ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name

    def quote_many(self, books, t, rfqs):
        """(seats, 32) quotes for several books in one call."""
        X = np.c_[np.repeat(books, len(rfqs), 0), np.full(len(books) * len(rfqs), t),
                  np.tile(rfqs, (len(books), 1))].astype(np.float32)
        return self.sess.run(None, {self.input: X})[0].reshape(len(books), len(rfqs)).astype(float)


class Function:
    """Wrap a numpy pricing function f(X: n x 13) -> n quotes (e.g. engine.price before export)."""
    def __init__(self, fn, name="python"):
        self.fn, self.name = fn, name

    def quote_many(self, books, t, rfqs):
        X = np.c_[np.repeat(books, len(rfqs), 0), np.full(len(books) * len(rfqs), t), np.tile(rfqs, (len(books), 1))]
        return np.asarray(self.fn(X), float).reshape(len(books), len(rfqs))


class Recorded:
    """What a real trader actually quoted (only meaningful in its own real game)."""
    def __init__(self, game):
        self.mine, self.name = game.mine, f"actual quotes (game {game.k})"

    def quote_many(self, books, t, rfqs):
        return np.tile(self.mine[t - 1], (len(books), 1))


class NeverTrade:
    name = "never trade"

    def quote_many(self, books, t, rfqs):
        return np.tile(rfqs[:, 1] - 0.0195, (len(books), 1))


# =====================================================================================  the field
def desk_styles(rng, n, field):
    is_off = rng.random(n) < field["p_off"]
    return dict(lean=field["lean_mu"] + field["lean_sd"] * rng.standard_normal(n),
                cap=rng.uniform(field["cap_lo"], max(field["cap_hi"], field["cap_lo"]), n),
                is_off=is_off, off_bid=rng.uniform(field["off_lo"], max(field["off_hi"], field["off_lo"]), n),
                trigger=np.full(n, field["trigger"]))


def fee_cost(pos, t):
    """Forward-looking fee cost of holding pos (..., 5) from round t: delta/vega once, exotics
    capital for every round left and breach penalties for ~HORIZON rounds."""
    left = 13 - t
    w_cap = UNIT_COST * np.r_[1, 1, left, left, left]
    w_br = BREACH_RATE * np.r_[1, 1, [min(left, HORIZON)] * 3]
    return (np.abs(pos) * w_cap).sum(-1) + (np.maximum(np.abs(pos) - LIMIT, 0) * w_br).sum(-1)


def field_quotes(books, t, rfqs, st, eps, wild, field):
    """Rival desks' quotes, (desks, 32). books: (desks, 5); eps, wild: (desks, 32) draws."""
    N, mid, r = rfqs[:, 0], rfqs[:, 1], rfqs[:, 2:]
    pos0, pos1 = books[:, None, :], books[:, None, :] + PHI * r[None]
    dcost = fee_cost(pos1, t) - fee_cost(pos0, t)
    q = mid + st["lean"][:, None] - field["risk_w"] * dcost / (PHI * N) + field["noise"] * eps
    q = np.minimum(q, mid + st["cap"][:, None])
    T = st["trigger"][:, None, None]
    relief = (np.maximum(np.abs(pos0[..., 2:]) - T, 0).sum(-1) - np.maximum(np.abs(pos1[..., 2:]) - T, 0).sum(-1)) > 0.02
    q = np.where(st["is_off"][:, None] & relief, np.maximum(q, mid + st["off_bid"][:, None]), q)
    q = np.where(wild < field["p_wild"], mid + 0.0199 * wild / max(field["p_wild"], 1e-12), q)   # stray bids
    return np.clip(q, mid - 0.0199, mid + 0.0199)


class Market:
    """Everything random about one game, drawn once so every engine meets the same market."""
    def __init__(self, rng, field, n_test, rfqs=None, seats=None, draws="shared", turn="mix", turn_label=None):
        self.field = field
        n = int(field["n_desks"])
        self.rfqs, self.turn = (rfqs, turn_label) if rfqs is not None else make_rfqs(rng, draws, turn)
        made = [make_seat(rng) for _ in range(n_test + n)]
        self.seats = np.array([s for s, _ in made])
        self.paths = [p for _, p in made]
        if seats is not None:                       # fixed seats for the test engines (real games)
            for i, s in enumerate(seats):
                self.seats[i] = s
        self.styles = desk_styles(rng, n, field)
        self.eps = rng.standard_normal((12, n, 32))
        self.wild = rng.random((12, n, 32))
        self.shares = Shares(field).draw(rng, (12, 32))
        self.n_test = n_test


def play(market, engines, opponents=(), record=False):
    """One game. engines sit in seats 0..len-1 (one each); opponents (engines) take the next
    seats as rival desks with lean 0; the rest of the field is numpy desks.
    Returns per seat per round dicts (test seats only), plus detail arrays if record."""
    m, F = market, market.field
    n_e, n_o = len(engines), len(opponents)
    n_d = len(m.styles["lean"]) - n_o           # numpy desks left after opponents take seats
    S = n_e + n_o + n_d
    seats = m.seats[:S]
    exo = np.zeros((S, 3))
    out = [[] for _ in range(n_e + n_o)]
    det = dict(q=[], fill=[]) if record else None
    st = {k: v[n_o:] for k, v in m.styles.items()}
    for t in range(1, 13):
        R = m.rfqs[t - 1]
        mid = R[:, 1]
        exo += seats[:, t - 1, 2:]
        books = np.c_[seats[:, t - 1, :2], exo]
        Q = np.empty((S, 32))
        for i, e in enumerate(list(engines) + list(opponents)):
            Q[i] = e.quote_many(books[i:i + 1], t, R)[0]
        Q[n_e + n_o:] = field_quotes(books[n_e + n_o:], t, R, st, m.eps[t - 1, n_o:], m.wild[t - 1, n_o:], F)
        ok = np.isfinite(Q) & (np.abs(Q - mid) <= OFF_MARKET)
        q = np.where(ok, Q, -np.inf)
        rank = 1 + (q[None, :, :] > q[:, None, :]).sum(1)
        K = m.shares.shape[-1]
        share = np.take_along_axis(np.broadcast_to(m.shares[t - 1], (S, 32, K)),
                                   np.minimum(rank, K)[..., None] - 1, -1)[..., 0]
        fill = np.where(ok & (rank <= K), share, 0.0)
        took = fill @ R[:, 2:]
        exo += took[:, 2:]
        pos = np.c_[seats[:, t - 1, :2] + took[:, :2], exo]
        edge = (fill * R[:, 0] * (mid - np.where(ok, Q, 0))).sum(1) + PRIVATE_EDGE
        for i in range(n_e + n_o):
            hedge, capital, penalty, breaches = fees(pos[i])
            out[i].append(dict(pnl=edge[i] - hedge - capital - penalty, edge=edge[i], hedge=hedge, capital=capital,
                               penalty=penalty, breaches=breaches, filled=int((fill[i] > 0).sum()),
                               first=int((ok[i] & (rank[i] == 1)).sum()), off=int((~ok[i]).sum()),
                               maxexo=float(np.abs(exo[i]).max())))
        if record:
            det["q"].append(Q)
            det["fill"].append(fill)
    return (out, det) if record else out


def fees(pos):
    excess = np.maximum(np.abs(pos) - LIMIT, 0)
    return (float(UNIT_COST[:2] @ np.abs(pos[:2])), float(UNIT_COST[2:] @ np.abs(pos[2:])),
            float(BREACH_RATE @ excess), int((excess > 0).sum()))


# =====================================================================================  fit
BUCKETS = [(-200, -10), (-10, -3), (-3, 3), (3, 15), (15, 60), (60, 201)]
PCT = [10, 25, 50, 75, 90]


def outcome_stats(mid, mine, fill, best, second):
    """What happened to a set of our quotes: the numbers --fit matches (real or simulated)."""
    F = fill > 0
    first = F & (mine >= best - 1e-12)
    sec = F & ~first & (mine >= second - 1e-12)
    low = F & ~first & ~sec
    rb = np.where(first, second, best)[F]          # best RIVAL quote, seen whenever we filled
    qm = (mine - mid) * 1e4
    rate = [F[(qm >= lo) & (qm < hi)].mean() if ((qm >= lo) & (qm < hi)).any() else np.nan for lo, hi in BUCKETS]
    pct = lambda x: np.percentile(x, PCT) if len(x) else np.full(len(PCT), np.nan)
    return dict(counts=np.array([first.sum(), sec.sum(), low.sum(), (~F).sum()], float), rate=np.array(rate),
                rbest=pct((rb - mid[F]) * 1e4), wild=np.array([np.mean((rb - mid[F]) > 0.004) if F.any() else np.nan]),
                rsecond=pct((second - mid)[low] * 1e4), low=np.array([fill[low].mean() if low.any() else np.nan]))


def real_stats(g):
    mid = np.array([r[:, 1] for r in g.rfqs])
    return outcome_stats(mid.ravel(), g.mine.ravel(), g.fill.ravel(), g.best.ravel(), g.second.ravel())


def sim_outcomes(g, field, rng, reps):
    """Our real quotes in game g against `reps` simulated fields: pooled outcome arrays,
    plus the 10 best rival quotes, sorted, (reps, 384, 10) for the conditional replay."""
    rec = Recorded(g)
    mids, mines, fills, bests, seconds, rivals = [], [], [], [], [], []
    for _ in range(reps):
        mk = Market(rng, field, 1, rfqs=g.rfqs, seats=[g.seat], turn_label="real")
        _, det = play(mk, [rec], record=True)
        Q = np.stack(det["q"], 1)                  # (seats, 12, 32)
        fill = np.stack(det["fill"], 1)[0]
        mid = np.array([r[:, 1] for r in g.rfqs])
        allq = np.where(np.abs(Q - mid) <= OFF_MARKET, Q, -np.inf)
        top = -np.sort(-allq, 0)
        mids.append(mid.ravel()); mines.append(g.mine.ravel()); fills.append(fill.ravel())
        bests.append(top[0].ravel()); seconds.append(top[1].ravel())
        rv = -np.sort(-allq[1:], 0)[:10]
        rv = np.concatenate([rv, np.full((10 - len(rv),) + rv.shape[1:], -np.inf)]) if len(rv) < 10 else rv
        rivals.append(rv.reshape(10, -1).T)
    cat = np.concatenate
    return cat(mids), cat(mines), cat(fills), cat(bests), cat(seconds), np.stack(rivals)


WEIGHTS = dict(counts=2.0, rate=1.0, rbest=1.0, wild=1.0, rsecond=1.0, low=1.0)   # fills drive P&L most
SCALE = dict(counts=None, rate=0.05, rbest=8.0, wild=0.05, rsecond=4.0, low=0.015)


def stats_loss(sim, obs, reps):
    l = 0.0
    for k in WEIGHTS:
        a, b = sim[k], obs[k]
        if k == "counts":
            a = a / reps
            d = (a - b) ** 2 / (b + 5)
        else:
            d = ((a - b) / SCALE[k]) ** 2
        l += WEIGHTS[k] * np.nansum(d)
    return l


SPACE = dict(n_desks=(6, 24), lean_mu=(-0.0015, 0.001), lean_sd=(0.0001, 0.002), cap_lo=(0.0, 0.0015),
             cap_hi=(0.0005, 0.004), noise=(0.00005, 0.0015), risk_w=(0.0, 1.5), p_off=(0.0, 0.6),
             trigger=(1.5, 3.5), off_lo=(0.002, 0.012), off_hi=(0.008, 0.0199), p_wild=(0.0, 0.03),
             K=(4, 8))
INT = ("n_desks", "K")


_FIT = {}


def _fit_init(keys=(1, 2)):
    _FIT["games"] = [RealGame(k) for k in keys]
    _FIT["obs"] = [real_stats(g) for g in _FIT["games"]]
    catalogue()
    Shares(FIELD)


def _fit_score(task):
    c, reps, seed = task
    rng = np.random.default_rng(seed)
    tot = 0.0
    for g, o in zip(_FIT["games"], _FIT["obs"]):
        m, q, f, b, s, _ = sim_outcomes(g, c, rng, reps)
        tot += stats_loss(outcome_stats(m, q, f, b, s), o, reps)
    return tot


def fit(seed=0, keys=(1, 2), n_search=500, reps_search=4, workers=6):
    """Fit the rival field to what happened to our real quotes in the given games."""
    import multiprocessing as mp
    _fit_init(keys)
    games, obs = _FIT["games"], _FIT["obs"]
    rng = np.random.default_rng(seed)
    clean = lambda c: {k: (int(round(v)) if k in INT else float(v)) for k, v in c.items()}
    draw = lambda: clean({k: rng.uniform(lo, hi) for k, (lo, hi) in SPACE.items()})
    vary = lambda c, s: clean({k: np.clip(c[k] + rng.normal(0, s * (hi - lo)), lo, hi) for k, (lo, hi) in SPACE.items()})
    t0 = time.time()
    with mp.Pool(workers, initializer=_fit_init, initargs=(keys,)) as pool:
        def rank(cands, reps, stage):   # within a stage every candidate meets the same markets;
            losses = pool.map(_fit_score, [(c, reps, seed + 1000 * stage) for c in cands])   # new ones each stage
            return sorted(zip(losses, range(len(cands)), cands), key=lambda x: x[0])

        cands = [clean({k: FIELD[k] for k in SPACE})] + [draw() for _ in range(n_search)]
        scored = rank(cands, reps_search, 1)
        print(f"search: {len(cands)} fields, best loss {scored[0][0]:.1f} ({time.time() - t0:.0f}s)", flush=True)
        best_pool = [c for _, _, c in scored[:12]]
        for step, s in enumerate([0.08, 0.04, 0.02]):
            scored = rank(best_pool + [vary(c, s) for c in best_pool for _ in range(8)], 10, 2 + step)
            best_pool = [c for _, _, c in scored[:12]]
            print(f"refine {step + 1}: best loss {scored[0][0]:.1f} ({time.time() - t0:.0f}s)", flush=True)
        final = rank(best_pool, 30, 9)
    best = final[0][2]
    print(f"final: loss {final[0][0]:.1f} on 30 markets per game\n")
    show_fit(best, games, obs)
    return best


def show_fit(field, games=None, obs=None):
    games = games or [RealGame(1), RealGame(2)]
    obs = obs or [real_stats(g) for g in games]
    rng = np.random.default_rng(99)
    names = dict(counts="first/second/lower/none", rate="fill rate by our q-mid bucket", rbest="rival best - mid bp p10..p90",
                 wild="share of rival best > 40bp", rsecond="rival 2nd - mid bp (we lower)", low="our lower-rank share")
    for g, o in zip(games, obs):
        m, q, f, b, s, _ = sim_outcomes(g, field, rng, 40)
        sim = outcome_stats(m, q, f, b, s)
        sim["counts"] = sim["counts"] / 40
        print(f"game {g.k}: our {len(g.mine.ravel())} real quotes, real vs simulated field")
        for k in names:
            fmt = (lambda v: " ".join(f"{x:5.1f}" for x in v)) if k in ("counts", "rbest", "rsecond") else \
                  (lambda v: " ".join(f"{x:5.2f}" for x in v))
            print(f"  {names[k]:32s} real {fmt(o[k]):38s} sim {fmt(sim[k])}")
    print("  fill-rate buckets (bp):", ", ".join(f"[{a},{b})" for a, b in BUCKETS))
    print("field:", json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in field.items()}))


# =====================================================================================  replay
class Replay:
    """Real game k with competition consistent with what we observed there."""
    def __init__(self, k, field, rng, prior_reps=150):
        self.g = g = RealGame(k)
        self.field = field
        if g.precise:
            *_, riv = sim_outcomes(g, field, rng, prior_reps)            # (reps, 384, 10) sorted rival quotes
            self.prior = riv
        self.K = int(field["K"])

    def competition(self, rng):
        """(384, 10) sorted rival quotes for one replay, consistent with the observed outcomes."""
        g, K, P = self.g, self.K, self.prior
        mine, fill, best, second = (a.ravel() for a in (g.mine, g.fill, g.best, g.second))
        mid = np.array([r[:, 1] for r in g.rfqs]).ravel()
        on = np.abs(mine - mid) <= OFF_MARKET
        F = fill > 0
        first = F & (mine >= best - 1e-12)
        sec = F & ~first & (mine >= second - 1e-12)
        low = F & ~first & ~sec
        out = np.empty((384, P.shape[2]))
        for j in range(384):
            c = P[:, j]                                                  # (reps, 10)
            if F[j]:
                c = c.copy()
                if first[j]:
                    c[:, 0] = second[j]
                    okc = c[:, 1] <= second[j]
                elif sec[j]:
                    c[:, 0] = best[j]
                    okc = c[:, 1] < mine[j]
                else:
                    c[:, 0], c[:, 1] = best[j], second[j]
                    above = (c > mine[j]).sum(1)
                    okc = (c[:, 2] <= second[j]) & (above >= 2) & (above <= K - 1)
            elif on[j]:
                okc = c[:, K - 1] > mine[j]
            else:
                okc = np.ones(len(c), bool)
            if okc.any():
                pick = c[rng.choice(np.flatnonzero(okc))]
            else:                                                        # repair the nearest draw
                pick = c[rng.integers(len(c))].copy()
                if F[j] and first[j]:
                    pick[1:] = np.minimum(pick[1:], second[j])
                elif F[j] and sec[j]:
                    pick[1:] = np.minimum(pick[1:], mine[j] - 1e-9)
                elif F[j]:
                    pick[2:] = np.minimum(pick[2:], second[j])
                    pick[K - 1:] = np.minimum(pick[K - 1:], mine[j] - 1e-9)    # keep us inside the K filled places
                else:
                    pick[:K] = np.maximum(pick[:K], mine[j] + 1e-9)
            out[j] = -np.sort(-pick)
        return out

    def play(self, engine, rng, comp=None, shares=None):
        g, K = self.g, self.K
        comp = self.competition(rng) if comp is None else comp
        shares = Shares(self.field).draw(rng, (12, 32)) if shares is None else shares
        obs_rank = np.full(384, 0)
        mine = g.mine.ravel()
        F = g.fill.ravel() > 0
        obs_rank[F] = 1 + (comp[F] > mine[F, None]).sum(1)
        exo, rows = np.zeros(3), []
        for t in range(1, 13):
            R = g.rfqs[t - 1]
            mid = R[:, 1]
            exo = exo + g.seat[t - 1, 2:]
            book = np.r_[g.seat[t - 1, :2], exo]
            q = engine.quote_many(book[None], t, R)[0]
            ok = np.isfinite(q) & (np.abs(q - mid) <= OFF_MARKET)
            sl = slice(32 * (t - 1), 32 * t)
            rank = 1 + (comp[sl] > q[:, None]).sum(1)
            share = shares[t - 1][np.arange(32), np.minimum(rank, K) - 1]
            same = F[sl] & (rank == obs_rank[sl])                      # same place as we really got
            share = np.where(same, g.fill.ravel()[sl], share)
            fill = np.where(ok & (rank <= K), share, 0.0)
            took = fill @ R[:, 2:]
            exo = exo + took[2:]
            pos = np.r_[g.seat[t - 1, :2] + took[:2], exo]
            edge = float((fill * R[:, 0] * (mid - np.where(ok, q, 0))).sum()) + PRIVATE_EDGE
            hedge, capital, penalty, breaches = fees(pos)
            rows.append(dict(pnl=edge - hedge - capital - penalty, edge=edge, hedge=hedge, capital=capital,
                             penalty=penalty, breaches=breaches, filled=int((fill > 0).sum()),
                             first=int((ok & (rank == 1)).sum()), off=int((~ok).sum()), maxexo=float(np.abs(exo).max())))
        return rows


# =====================================================================================  checks
def check():
    """1) Rebuild each real game's edge and fees from its game history: must match its P&L file.
    2) Replay our actual quotes in the conditional replay: must give the official P&L."""
    for k in GAMES:
        qf, gf, pf = GAMES[k]
        g, p = table(gf), table(pf)
        rnd = num(g["Round"])
        risk = np.c_[tuple(num(g[c]) for c in RISKS)]
        edge_rows = (num(g["MidValue"]) - num(g["Quoted_coupon"])) * num(g["Notional"])
        exo, worst = np.zeros(3), 0.0
        for t in range(1, 13):
            took = risk[rnd == t].sum(0)
            exo += took[2:]
            hedge, capital, penalty, _ = fees(np.r_[took[:2], exo])
            edge = edge_rows[rnd == t].sum()
            pnl = edge - hedge - capital - penalty
            worst = max(worst, abs(pnl - num(p["PnL"])[t - 1]), abs(edge - num(p["Edge"])[t - 1]))
        print(f"game {k} ({g['Trader'][0]}): edge and P&L rebuilt from fills, largest difference {worst:.2g} $mm"
              f"  (official total {num(p['PnL']).sum():+.2f})")
    rng = np.random.default_rng(0)
    for k in (1, 2):
        rp = Replay(k, field_for(k), rng, prior_reps=40)
        rows = rp.play(Recorded(rp.g), rng)
        off = num(rp.g.pnl["PnL"])
        print(f"game {k}: replaying our actual quotes in the conditional replay: P&L {sum(r['pnl'] for r in rows):+.4f}"
              f" vs official {off.sum():+.4f}; worst round diff {max(abs(r['pnl'] - o) for r, o in zip(rows, off)):.2g}")
    g0 = RealGame(0)
    q = table(GAMES[0][0])
    X0 = np.c_[tuple(num(q["RFQ|" + c]) for c in ["Notional", "MidValue"] + RISKS)].reshape(12, 32, 7)
    print(f"game 0 rebuilt from the shared draws + drift: max diff vs its 2-decimal file {np.abs(np.round(np.array(g0.rfqs), 2) - X0).max():.3f}")


# =====================================================================================  runs
KEYS = ["pnl", "edge", "hedge", "capital", "penalty", "breaches", "filled", "first", "off"]


def totals(rows):
    d = {k: sum(r[k] for r in rows) for k in KEYS}
    d["sd"] = float(np.std([r["pnl"] for r in rows]))
    d["maxexo"] = max(r["maxexo"] for r in rows)
    return d


def jitter_field(rng, field, amount):
    """A market around the fitted one, so engines can't over-fit one exact field."""
    if amount <= 0:
        return dict(field)
    f = dict(field)
    f["n_desks"] = int(max(4, field["n_desks"] + rng.integers(-4, 5)))
    f["lean_mu"] = field["lean_mu"] + rng.uniform(-amount, amount) * 0.001
    for k in ("lean_sd", "noise", "cap_lo", "cap_hi", "p_wild"):
        f[k] = field[k] * rng.uniform(1 - 0.4 * amount, 1 + 0.4 * amount)
    f["p_off"] = float(np.clip(field["p_off"] * rng.uniform(1 - 0.5 * amount, 1 + 0.5 * amount), 0, 1))
    return f


def run_solo(engines, opponents, n_games, seed, draws, turn, jitter, shift):
    """Each engine in seat 0 of the SAME market (common random numbers): paired comparison."""
    res = {e.name: [] for e in engines}
    meta = []
    for gi in range(n_games):
        rng = np.random.default_rng([seed, gi])
        names, fields = regimes()
        regime = int(rng.integers(len(fields)))
        field = jitter_field(rng, fields[regime], jitter)
        field["lean_mu"] += shift * 1e-4
        mk = Market(rng, field, 1 + len(opponents), draws=draws, turn=turn)
        meta.append(dict(turn=mk.turn, size=path_size(mk.paths[0]), field=names[regime]))
        for e in engines:
            res[e.name].append(totals(play(mk, [e], opponents)[0]))
    return res, meta


def run_tournament(engines, opponents, n_games, seed, draws, turn, jitter, shift):
    """All engines in the same games as separate seats; seats rotate so each engine meets every book."""
    res = {e.name: [] for e in engines}
    ranks = {e.name: [] for e in engines}
    for gi in range(n_games):
        rng = np.random.default_rng([seed, gi])
        fields = regimes()[1]
        field = jitter_field(rng, fields[int(rng.integers(len(fields)))], jitter)
        field["lean_mu"] += shift * 1e-4
        mk = Market(rng, field, len(engines) + len(opponents), draws=draws, turn=turn)
        order = np.roll(np.arange(len(engines)), gi)
        mk.seats[:len(engines)] = mk.seats[:len(engines)][order]
        rows = play(mk, engines, opponents)
        pn = [sum(r["pnl"] for r in rows[i]) for i in range(len(engines))]
        rk = np.argsort(np.argsort(-np.array(pn))) + 1
        for i, e in enumerate(engines):
            res[e.name].append(totals(rows[i]))
            ranks[e.name].append(rk[i])
    return res, ranks


def report(res, meta=None, title=""):
    names = list(res)
    w = max(14, *(len(n) + 2 for n in names))
    line = lambda label, f: print(f"{label:30s}" + "".join(f"{f(n):>{w}}" for n in names))
    n = len(res[names[0]])
    print(f"\n{title}: {n} games, $mm per game")
    print(f"{'':30s}" + "".join(f"{x:>{w}}" for x in names))
    arr = {nm: {k: np.array([g[k] for g in res[nm]]) for k in res[nm][0]} for nm in names}
    line("P&L mean +- se", lambda nm: f"{arr[nm]['pnl'].mean():+.2f}+-{arr[nm]['pnl'].std() / np.sqrt(n):.2f}")
    line("P&L median", lambda nm: f"{np.median(arr[nm]['pnl']):+.2f}")
    line("P&L p10 / worst", lambda nm: f"{np.percentile(arr[nm]['pnl'], 10):+.2f}/{arr[nm]['pnl'].min():+.1f}")
    line("games > 0", lambda nm: f"{np.mean(arr[nm]['pnl'] > 0):.0%}")
    for k in ["edge", "hedge", "capital", "penalty"]:
        line(f"  {k}", lambda nm: f"{arr[nm][k].mean():.2f}")
    line("breaches / game", lambda nm: f"{arr[nm]['breaches'].mean():.2f}")
    line("games with a breach", lambda nm: f"{np.mean(arr[nm]['breaches'] > 0):.0%}")
    line("max |exotic| (mean)", lambda nm: f"{arr[nm]['maxexo'].mean():.2f}")
    line("round P&L sd (consistency)", lambda nm: f"{arr[nm]['sd'].mean():.3f}")
    line("fills / first places", lambda nm: f"{arr[nm]['filled'].mean():.0f}/{arr[nm]['first'].mean():.0f}")
    line("off-market quotes", lambda nm: f"{arr[nm]['off'].mean():.1f}")
    if len(names) > 1:
        ref = arr[names[-1]]["pnl"]
        line(f"vs {names[-1][:20]} (paired)", lambda nm: "-" if nm == names[-1] else
             f"{(arr[nm]['pnl'] - ref).mean():+.2f}+-{(arr[nm]['pnl'] - ref).std() / np.sqrt(n):.2f}")
        line("  games ahead", lambda nm: "-" if nm == names[-1] else f"{np.mean(arr[nm]['pnl'] > ref):.0%}")
    if meta:
        for label, key, vals in [("private exotic path size", "size", (0.5, 1.0, 2.0)),
                                 ("cube turn", "turn", ("none", "drift", "fixed")),
                                 ("rivals like", "field", tuple(regimes()[0]))]:
            for v in vals:
                sel = np.array([m[key] == v for m in meta])
                if sel.any():
                    line(f"  {label} {v} ({sel.sum()})", lambda nm: f"{arr[nm]['pnl'][sel].mean():+.2f}")


def team_models():
    """Every distinct .onnx our team has produced (by content), plus the organisers' benchmark."""
    import hashlib
    seen, out = set(), []
    for p in [BENCHMARK] + sorted(glob.glob(os.path.join(ROOT, "**", "*.onnx"), recursive=True)):
        h = hashlib.md5(open(p, "rb").read()).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(p)
    return out


def evaluate(engine, games=60, seed=7, replay=True, reps=10, jitter=1.0, shift=0.0, draws="shared", turn="mix"):
    """For tuners: one engine (Onnx / Function / anything with quote_many) -> summary dict.
    Synthetic games use common random numbers per seed, so two engines with the same seed
    meet exactly the same markets. With replay, adds games 1 and 2 against observed competition."""
    res, meta = run_solo([engine], [], games, seed, draws, turn, jitter, shift)
    p = np.array([g["pnl"] for g in res[engine.name]])
    out = dict(pnl=p.mean(), se=p.std() / np.sqrt(len(p)), p10=np.percentile(p, 10), worst=p.min(),
               breaches=np.mean([g["breaches"] for g in res[engine.name]]),
               round_sd=np.mean([g["sd"] for g in res[engine.name]]),
               by_size={s: p[[m["size"] == s for m in meta]].mean() for s in (0.5, 1.0, 2.0)
                        if any(m["size"] == s for m in meta)})
    if replay:
        if "replays" not in _CACHE:
            rng = np.random.default_rng(seed)
            _CACHE["replays"] = {k: Replay(k, field_for(k), rng) for k in (1, 2)}
        for k, rp in _CACHE["replays"].items():
            rng = np.random.default_rng([seed, k])
            out[f"game{k}"] = np.mean([sum(r["pnl"] for r in rp.play(engine, rng)) for _ in range(reps)])
    return out


def main():
    global FIELD
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("engines", nargs="*", help="engines to test (.onnx); the last is the reference")
    ap.add_argument("--opponents", nargs="*", default=[], help=".onnx engines that join the field as rival desks; 'team' = every distinct team model + benchmark")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--draws", choices=["shared", "fresh"], default="shared", help="real RFQ draws (default) or new ones")
    ap.add_argument("--turn", choices=["mix", "none", "drift", "fixed"], default="mix", help="exotic cube turn")
    ap.add_argument("--jitter", type=float, default=1.0, help="randomise the field around the fit (0 = exact fit)")
    ap.add_argument("--shift", type=float, default=0.0, help="make every rival desk this many bp more aggressive")
    ap.add_argument("--tournament", action="store_true", help="all engines in the same games, head to head")
    ap.add_argument("--replay", action="store_true", help="real games 0-2 instead of synthetic games")
    ap.add_argument("--reps", type=int, default=30, help="replays per real game")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--show-fit", action="store_true")
    a = ap.parse_args()
    if a.check:
        return check()
    if a.fit:
        fields = {}
        for k in (1, 2):
            print(f"==== fitting the rival field of game {k}", flush=True)
            fields[f"game {k}"] = fit(seed=k, keys=(k,))
        with open(FIELD_FILE, "w") as f:
            json.dump({"fields": fields}, f, indent=1)
        print()
        print(f"saved {FIELD_FILE}")
        return
    if a.show_fit:
        for k in (1, 2):
            show_fit(field_for(k), [RealGame(k)])
        return

    paths = a.engines or [os.path.join(ROOT, "submission", "team.onnx"), BENCHMARK]
    paths = [p for x in paths for p in (sorted(glob.glob(x)) or [x])]
    engines = [Onnx(p) for p in paths]
    stems = [e.name for e in engines]
    for e in engines:                                  # same-named files: label them by folder
        if stems.count(e.name) > 1:
            e.name = f"{os.path.basename(os.path.dirname(os.path.abspath(e.path)))}/{e.name}"
    opp_paths = [p for x in a.opponents for p in (team_models() if x == "team" else (sorted(glob.glob(x)) or [x]))]
    opponents = [Onnx(p) for p in opp_paths]
    if opponents:
        print("rival desks running our models:", ", ".join(os.path.relpath(p, ROOT) for p in opp_paths))
    t0 = time.time()
    if a.replay:
        rng = np.random.default_rng(a.seed)
        for k in (1, 2, 0):
            if k == 0:                                 # 2-decimal coupons: no conditioning possible
                g = RealGame(0)
                res = {e.name: [] for e in engines}
                for r in range(a.reps):
                    r_rng = np.random.default_rng([a.seed, 100, r])
                    mk = Market(r_rng, regimes()[1][r % len(regimes()[1])], 1 + len(opponents), rfqs=g.rfqs,
                                seats=[g.seat], turn_label="real")
                    for e in engines:
                        res[e.name].append(totals(play(mk, [e], opponents)[0]))
                report(res, title=f"game 0 ({g.trader}'s seat, exact RFQs rebuilt), simulated field")
                print(f"  official P&L of {g.trader} in that game: {num(g.pnl['PnL']).sum():+.2f}")
                continue
            rp = Replay(k, field_for(k), rng)
            res = {e.name: [] for e in list(engines) + [Recorded(rp.g)]}
            for r in range(a.reps):
                comp = rp.competition(rng)
                shares = Shares(rp.field).draw(rng, (12, 32))
                for e in list(engines) + [Recorded(rp.g)]:
                    res[e.name].append(totals(rp.play(e, rng, comp, shares)))
            res = {n: res[n] for n in [e.name for e in engines] + [f"actual quotes (game {k})"]}
            report(res, title=f"game {k} replay ({rp.g.trader}'s seat, competition conditioned on what we saw)")
            print(f"  official P&L in game {k}: {num(rp.g.pnl['PnL']).sum():+.2f}")
    elif a.tournament:
        res, ranks = run_tournament(engines, opponents, a.games, a.seed, a.draws, a.turn, a.jitter, a.shift)
        report(res, title="tournament (all engines in the same games)")
        print(f"{'average rank among engines':30s}" + "".join(f"{np.mean(ranks[n]):>14.2f}" for n in res))
    else:
        res, meta = run_solo(engines, opponents, a.games, a.seed, a.draws, a.turn, a.jitter, a.shift)
        opp = f", {len(opponents)} team models in the field" if opponents else ""
        report(res, meta, title=f"synthetic games ({a.draws} draws, turn {a.turn}, shift {a.shift:+g}bp{opp})")
    print(f"\n({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
