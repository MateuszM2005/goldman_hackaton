"""Tests for engine_center.py and sim_center.py.

    python src/testing/test_engine_center.py          # plain python, no pytest needed
    python -m pytest src/testing/test_engine_center.py

Engine: the exported ONNX equals price() on the organisers' test sample and on 200k random
inputs (including nonsense ones); quotes are finite and inside the +-0.02 band; price()
equals an independent float64 re-implementation of the formula in the docstring; a won
trade never loses money by the engine's own accounting; flipping the sign of book and RFQ
together changes nothing; a bigger offsettable position never lowers the quote.
Simulator: its fee rules rebuild every round of the organisers' P&L files from the real
fills; the catalogue holds the three real games' private exotic paths; synthetic RFQs and
seats have the distributions found in the data.
"""
import csv
import os
import sys
import tempfile

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")))   # engine_center.py
import engine_center as ec   # noqa: E402
import sim_center as sc      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ONNX = {}


def onnx_quotes(X, p=None):
    key = tuple(sorted((p or {}).items()))
    if key not in _ONNX:
        path = os.path.join(tempfile.mkdtemp(), "center.onnx")
        ec.export(path, p)
        _ONNX[key] = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return _ONNX[key].run(None, {"X": np.asarray(X, np.float32)})[0][:, 0]


def sample_rows():
    with open(os.path.join(ROOT, "data", "game_0", "TestPredict.csv"), newline="") as f:
        return np.array([r[:13] for r in list(csv.reader(f))[1:]], np.float32)


def random_inputs(rng, n):
    """Plausible and nonsense rows: huge books, round 0 / 13 / fractional, notional 0."""
    X = np.zeros((n, 13), np.float32)
    X[:, 0] = rng.uniform(-40, 40, n)
    X[:, 1:5] = rng.uniform(-10, 10, (n, 4))
    X[:, 5] = rng.choice([-1, 0, 1, 2, 3, 6, 9, 11, 12, 13, 20, 5.5], n)
    X[:, 6] = rng.choice([0, 1, 50, 90, 100, 109, 150], n)
    X[:, 7] = rng.uniform(0.03, 0.08, n)
    X[:, 8] = rng.uniform(-6, 6, n)
    X[:, 9:13] = rng.uniform(-2, 2, (n, 4))
    return X


# ---- engine -------------------------------------------------------------------------
def test_onnx_matches_python_on_test_sample():
    X = sample_rows()
    q = onnx_quotes(X)
    assert np.all(np.isfinite(q))
    assert np.abs(q - ec.price(X, ec.NUMPY)[:, 0]).max() < 1e-6
    assert np.all(np.abs(q - X[:, 7]) < 0.02)


def test_onnx_matches_python_on_random_and_nonsense_inputs():
    X = random_inputs(np.random.default_rng(1), 200_000)
    q = onnx_quotes(X)
    assert np.all(np.isfinite(q)), "NaN or inf quote"
    assert np.abs(q - ec.price(X, ec.NUMPY)[:, 0]).max() < 1e-6
    assert np.all(np.abs(q - X[:, 7]) <= ec.BOUND + 1e-6), "quote outside mid +- BOUND"


def test_onnx_matches_python_with_other_settings():
    rng = np.random.default_rng(2)
    X = random_inputs(rng, 20_000)
    p = dict(MARGIN=0.0004, PASS=0.6, MAX_DISCOUNT=0.003, CEILING=0.0003, RELIEF_PASS=0.4,
             MAX_RELIEF=0.004, BREACH_H=2.0, CAPITAL_H=5.0, MOVE_DV=2.5, MOVE_EX=1.7, SKEW=1.2,
             WL_DELTA=9.0, WL_VEGA=2.1, WL_EXOTIC=1.8)
    assert np.abs(onnx_quotes(X, p) - ec.price(X, ec.NUMPY, p)[:, 0]).max() < 1e-6


def reference_quote(x, p):
    """The docstring's formula, written out again in float64, one row at a time."""
    book, rnd, N, mid, rfq = x[0:5], min(max(x[5], 1), 12), max(x[6], 1), x[7], x[8:13]
    wl = np.array([p["WL_DELTA"], p["WL_VEGA"]] + [p["WL_EXOTIC"]] * 3)
    move = np.array([p["MOVE_DV"]] * 2 + [p["MOVE_EX"]] * 3)
    fee, rate = np.array([0.006, 0.025, 0.006, 0.006, 0.006]), np.array([0.05, 0.05, 1, 1, 1])
    left = 13 - rnd
    cap_r = np.array([1, 1] + [min(left, p["CAPITAL_H"])] * 3)
    br_r = np.array([1, 1] + [min(left, p["BREACH_H"])] * 3)
    breach = lambda pos: rate * np.maximum(np.abs(pos) - wl, 0) * br_r
    dist = lambda pos: fee * np.abs(pos) * (1 + 0.5 * p["SKEW"] * np.abs(pos) / wl) * cap_r + breach(pos)
    after = book + move * rfq
    change = (dist(after) - dist(book)) / move
    added, removed = change[change > 0].sum() / N, -change[change < 0].sum() / N
    relief = (np.maximum(breach(book) - breach(after), 0) / move).sum() / N
    q = mid - p["MARGIN"] - added + min(p["PASS"] * removed, p["MAX_DISCOUNT"])
    q = min(q, mid + p["CEILING"]) + min(p["RELIEF_PASS"] * relief, p["MAX_RELIEF"])
    return min(max(q, mid - ec.BOUND), mid + ec.BOUND), added - removed


def test_old_settings_reproduce_the_old_engine():
    """With engine.py's tuned settings (as submitted for game 2), the center engine IS the old one."""
    import tune_center
    old = ort.InferenceSession(os.path.join(ROOT, "submissions", "game_2", "Shame Street.onnx"),
                               providers=["CPUExecutionProvider"])
    for X in (sample_rows(), random_inputs(np.random.default_rng(11), 50_000)):
        q_old = old.run(None, {old.get_inputs()[0].name: X})[0][:, 0]
        assert np.abs(ec.price(X, ec.NUMPY, tune_center.OLD)[:, 0] - q_old).max() < 1e-6


def test_price_equals_independent_reimplementation():
    rng = np.random.default_rng(3)
    X = random_inputs(rng, 3_000).astype(np.float64)
    q = ec.price(X.astype(np.float32), ec.NUMPY)[:, 0]
    ref = np.array([reference_quote(x, ec.P)[0] for x in X])
    assert np.abs(q - ref).max() < 2e-6


def test_won_trade_never_loses_by_own_accounting():
    """mid - quote >= the net distance the trade adds (unless the quote sits on the floor)."""
    rng = np.random.default_rng(4)
    X = random_inputs(rng, 5_000).astype(np.float64)
    X[:, 6] = np.maximum(X[:, 6], 50)
    for x in X:
        q, net = reference_quote(x, ec.P)
        if q > x[7] - ec.BOUND + 1e-12:
            assert x[7] - q >= net - 1e-9, (x, q, net)
    assert ec.P["PASS"] + ec.P["RELIEF_PASS"] <= 1 + 1e-12


def test_sign_flip_symmetry():
    rng = np.random.default_rng(5)
    X = random_inputs(rng, 20_000)
    Y = X.copy()
    Y[:, 0:5] *= -1
    Y[:, 8:13] *= -1
    assert np.abs(ec.price(X, ec.NUMPY) - ec.price(Y, ec.NUMPY)).max() < 1e-7


def test_bigger_offsettable_position_never_lowers_the_quote():
    for axis, rfq in [(0, [-4, 1, 1, 1, 1]), (1, [4, -1, 1, 1, 1]), (2, [4, 1, -1, 1, 1]), (4, [4, 1, 1, 1, -1])]:
        for rnd in (1, 4, 8, 12):
            sizes = np.linspace(0, 8 if axis else 30, 200)
            X = np.zeros((len(sizes), 13), np.float32)
            X[:, axis] = sizes
            X[:, 5], X[:, 6], X[:, 7], X[:, 8:13] = rnd, 100, 0.053, rfq
            q = ec.price(X, ec.NUMPY)[:, 0]
            assert np.all(np.diff(q) >= -1e-7), (axis, rnd)


def test_quote_band():
    X = random_inputs(np.random.default_rng(6), 50_000)
    s = ec.price(X, ec.NUMPY)[:, 0] - X[:, 7]
    assert s.max() <= ec.P["CEILING"] + ec.P["MAX_RELIEF"] + 1e-7
    assert s.min() >= -ec.BOUND - 1e-7


# ---- simulator ----------------------------------------------------------------------
def test_fees_rebuild_official_pnl_files():
    files = {"Round_1": ("data/game_1/trader_game_history.csv", "data/game_1/trader_pnl_by_round.csv", 1e-9),
             "Round_2": ("data/game_2/trader_game_history.csv",
                         "data/game_2/trader_pnl_by_round.csv", 1e-9),
             "round_0": ("data/game_0/trader_game_history.csv", "data/game_0/trader_pnl_by_round.csv", 0.03)}
    for name, (history, pnl, tol) in files.items():
        g, p = sc._table(history), sc._table(pnl)
        rnd, risk = sc._num(g["Round"]), np.c_[tuple(sc._num(g[c]) for c in sc.RISKS)]
        exo = np.zeros(3)
        for t in range(1, 13):
            took = risk[rnd == t].sum(0)
            exo += took[2:]
            hedge, capital, penalty, breaches = sc.fees(np.r_[took[:2], exo])
            assert abs(hedge + capital + penalty - sc._num(p["Costs"])[t - 1]) < tol, (name, t)
            assert breaches == int(sc._num(p["Breaches"])[t - 1]), (name, t)


def test_catalogue_holds_real_games_paths():
    paths, index = sc.exotic_paths()
    assert len(paths) == 36
    for key, (_, history, run) in sc.REAL.items():
        _, seat = sc.real_game(key, np.random.default_rng(0))
        assert np.abs(seat[:, 2:] - paths[index[run]]).max() < 0.006, key   # game 0's file: 2dp
    assert len(sc.holdout()) == 3


def test_synthetic_rfqs_follow_the_generator():
    rng = np.random.default_rng(7)
    R = np.vstack([r for _ in range(20) for r in sc.synthetic_rfqs(rng)])
    assert R.shape[1] == 7 and len(R) == 20 * 12 * 32
    assert np.all(R[:, 0] == np.round(R[:, 0])) and R[:, 0].min() >= 90 and R[:, 0].max() <= 109
    assert R[:, 1].min() >= 0.048 and R[:, 1].max() <= 0.058
    assert np.abs(R[:, 2]).min() >= 3.2 and np.abs(R[:, 2]).max() <= 4.8
    assert np.abs(R[:, 3]).min() >= 0.8 and np.abs(R[:, 3]).max() <= 1.2
    norm = np.linalg.norm(R[:, 4:7], axis=1)
    assert norm.min() >= 0.9 * np.sqrt(3) - 1e-9 and norm.max() <= 1.1 * np.sqrt(3) + 1e-9
    one = sc.synthetic_rfqs(np.random.default_rng(8))[0]
    assert np.array_equal(np.sign(one[:, 2:4]), sc.DV_SIGNS)          # delta first, then vega


def test_synthetic_seats():
    rng = np.random.default_rng(9)
    for _ in range(50):
        s = sc.synthetic_seat(rng)
        assert s.shape == (12, 5)
        assert np.allclose(np.hypot(s[:, 0] / 15, s[:, 1] / 3), 1)
        paths, _ = sc.exotic_paths()
        hit = [i for i in range(len(paths)) if np.abs(paths[i] - s[:, 2:]).max() < 1e-12]
        assert hit and hit[0] not in sc.holdout()


def test_play_books_add_up():
    rng = np.random.default_rng(10)
    rival = sc.Onnx(sc.BENCHMARK)
    rounds = sc.synthetic_rfqs(rng)
    seats = [sc.synthetic_seat(rng) for _ in range(6)]
    rows = sc.play(ec.Engine(), rival, rounds, seats, rng)
    for r in rows:
        assert abs(r["pnl"] - (r["edge"] - r["hedge"] - r["capital"] - r["penalty"])) < 1e-12
    never = sc.play(sc.NeverTrade(), rival, rounds, seats, np.random.default_rng(10))
    assert all(r["filled"] == 0 for r in never)
    assert np.allclose(never[-1]["exo"], seats[0][:, 2:].sum(0))      # only private moves


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, f in tests:
        try:
            f()
            print(f"PASS  {name}")
        except Exception as e:                       # report every failure, not just the first
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__} {str(e)[:300]}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
