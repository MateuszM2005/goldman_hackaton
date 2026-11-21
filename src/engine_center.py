"""Quoting engine "center": the old cost-based engine, refined with what games 0-2 and the
RFQ generator taught us.

    python src/engine_center.py   ->  submissions/center/Shame Street.onnx  and  TestPredict.csv

We PAY the coupon we quote (higher = better for the client, less edge for us). The margin
we leave, mid - quote, follows the distance from the center of risk space:

    margin = MARGIN + added - min(PASS x removed, MAX_DISCOUNT)    (floor: -CEILING)
             - min(RELIEF_PASS x breach removed, MAX_RELIEF)        (may go above mid)

  distance  the fee-weighted distance of a book from the center (all five risks at 0), one
            convex term per risk: hedge / capital cost x |position|, growing by SKEW as the
            position fills its working limit, plus the breach penalty beyond that limit.
  added     distance the trade adds, per $ of notional, charged in full (a won trade never
            loses money, even when it pushes us over a limit).
  removed   distance the trade takes away: the further from the center we are, the more a
            trade back towards it is worth, and the thinner the margin we leave on it.
  breach    a trade that takes us back under a limit may be bid ABOVE mid, up to MAX_RELIEF.

What changed from the old engine (src/engine.py as of commit 0fcf42b), and why. With the
old settings (tune_center.OLD) this engine reproduces the old one exactly (tested):
  * Breaches are priced over BREACH_H rounds, not every round left. The private exotic
    cycle swings back by itself, and each round's shock can be traded away in the round it
    arrives, so in round 1 an excess is not worth 12 rounds of penalty (tuned: ~1.2 rounds).
  * Exotics are kept under the REAL limit (WL_EXOTIC = 3), not a buffer below it. With a
    2.4-2.6 buffer the engine paid above mid to fix "breaches" that cost nothing.
  * Each trade is valued as part of its group (MOVE_DV / MOVE_EX). All 32 copies of the
    engine see the same book, and 16 RFQs push each risk the same way. Valuing a trade as a
    slice of that joint move (tuned: 2.8x the RFQ's delta/vega, 1.4x its exotics) stops all
    copies from chasing the same offset and overshooting the opposite limit (game 2's
    engine: vanna -4.7, gap +4.7).
  * Game 2 handed out exotic shocks of up to 5 a round and a -5 inherited book. The old
    engine never paid above mid - 2.4bp, so it sat in breach (simulated -27 $mm). This one
    pays up to mid + 4.8bp for any trade back towards the center and up to ~47bp more to
    get back under a limit, never the +199bp the submitted game-2 engine paid.
  * It passes 32% of the distance removed to the client (old: 100%), so it wins fewer
    small trades in calm games and keeps its firepower for the trades that matter.
Settings: tuned by src/tune_center.py (random search, 3 stages) on games from
src/sim_center.py, then WL_EXOTIC moved to 3, then checked by a coordinate search (no
single setting change improved it) and in src/arena.py, an independently fitted market.
"""
import csv
import os
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

TEAM = "Shame Street"

# ---- settings, coupon units (0.001 = $100k on $100mm); tuned with src/tune_center.py
P = dict(
    MARGIN=0.0,            # profit we want on a trade that leaves the distance unchanged
    PASS=0.3175,           # share of the distance removed that we hand to the client...
    MAX_DISCOUNT=0.001512, # ...at most this much
    CEILING=0.0004815,     # an ordinary trade never pays more than mid + CEILING
    RELIEF_PASS=0.1134,    # extra share of a breach penalty removed, above the ceiling...
    MAX_RELIEF=0.004265,   # ...at most this much (PASS + RELIEF_PASS <= 1)
    BREACH_H=1.189,        # rounds of penalty an exotic excess is expected to last
    CAPITAL_H=10.66,       # rounds of capital a unit of exotic is expected to stay
    MOVE_DV=2.799,         # share of an RFQ's delta/vega that lands, counting the clones
    MOVE_EX=1.433,         # the same for exotics
    SKEW=0.02286,          # extra cost of a unit at a full working limit, x its fee
    WL_DELTA=9.31, WL_VEGA=2.004, WL_EXOTIC=3.0,   # working limits (official 12 / 3 / 3)
)
BOUND = 0.0195             # |quote - mid| > 0.02 counts as an off-market quote

UNIT_COST   = np.array([[0.006, 0.025, 0.006, 0.006, 0.006]], np.float32)
BREACH_RATE = np.array([[0.05,  0.05,  1.00,  1.00,  1.00 ]], np.float32)
IS_EXOTIC   = np.array([[0,     0,     1,     1,     1    ]], np.float32)
ONES        = np.ones((5, 1), np.float32)


def cols(*idx):
    m = np.zeros((13, len(idx)), np.float32)
    m[list(idx), range(len(idx))] = 1
    return m


def price(X, F, p=None):
    """Quote for each row of X (n x 13, the handout's columns). F is NUMPY or GRAPH."""
    p = dict(P, **(p or {}))
    wl = np.array([[p["WL_DELTA"], p["WL_VEGA"]] + [p["WL_EXOTIC"]] * 3], np.float32)
    move = np.array([[p["MOVE_DV"]] * 2 + [p["MOVE_EX"]] * 3], np.float32)
    book = X @ cols(0, 1, 2, 3, 4)
    rnd  = F.min(F.max(X @ cols(5), 1.0), 12.0)
    N    = F.max(X @ cols(6), 1.0)
    mid  = X @ cols(7)
    rfq  = X @ cols(8, 9, 10, 11, 12)

    left = 13.0 - rnd                                   # rounds charged, this one included
    capital_rounds = 1 + IS_EXOTIC * (F.min(left, p["CAPITAL_H"]) - 1)
    breach_rounds  = 1 + IS_EXOTIC * (F.min(left, p["BREACH_H"]) - 1)

    def breach(pos):
        return BREACH_RATE * F.relu(F.abs(pos) - wl) * breach_rounds

    def distance(pos):                                  # per risk, $mm
        size = F.abs(pos)
        return UNIT_COST * size * (1 + 0.5 * p["SKEW"] * size / wl) * capital_rounds + breach(pos)

    after   = book + move * rfq                         # the joint move this trade is part of
    change  = (distance(after) - distance(book)) / move # per unit of this RFQ's risk
    added   = (F.relu(change) @ ONES) / N
    removed = (F.relu(0.0 - change) @ ONES) / N
    relief  = ((F.relu(breach(book) - breach(after)) / move) @ ONES) / N
    quote = mid - p["MARGIN"] - added + F.min(p["PASS"] * removed, p["MAX_DISCOUNT"])
    quote = F.min(quote, mid + p["CEILING"]) + F.min(p["RELIEF_PASS"] * relief, p["MAX_RELIEF"])
    return F.min(F.max(quote, mid - BOUND), mid + BOUND)


# ---- numpy / ONNX ---------------------------------------------------------------------
NUMPY = SimpleNamespace(abs=np.abs, relu=lambda a: np.maximum(a, 0), min=np.minimum, max=np.maximum)
NODES, CONSTS = [], []


class Sym:
    __array_ufunc__ = None

    def __init__(self, name): self.name = name
    def __add__(s, o): return op("Add", s, o)
    def __radd__(s, o): return op("Add", o, s)
    def __sub__(s, o): return op("Sub", s, o)
    def __rsub__(s, o): return op("Sub", o, s)
    def __mul__(s, o): return op("Mul", s, o)
    def __rmul__(s, o): return op("Mul", o, s)
    def __truediv__(s, o): return op("Div", s, o)
    def __rtruediv__(s, o): return op("Div", o, s)
    def __matmul__(s, o): return op("MatMul", s, o)


def op(kind, *args):
    names = []
    for a in args:
        if not isinstance(a, Sym):
            name = f"c{len(CONSTS)}"
            CONSTS.append(numpy_helper.from_array(np.asarray(a, np.float32), name))
            a = Sym(name)
        names.append(a.name)
    out = f"t{len(NODES)}"
    NODES.append(helper.make_node(kind, names, [out]))
    return Sym(out)


GRAPH = SimpleNamespace(abs=lambda a: op("Abs", a), relu=lambda a: op("Relu", a),
                        min=lambda a, b: op("Min", a, b), max=lambda a, b: op("Max", a, b))


def export(path, p=None):
    """Same interface as benchmark.onnx: X (n x 13 float) -> variable (n x 1 float)."""
    NODES.clear()
    CONSTS.clear()
    out = price(Sym("X"), GRAPH, p)
    NODES.append(helper.make_node("Identity", [out.name], ["variable"]))
    graph = helper.make_graph(NODES, "quoting_engine_center",
                              [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 13])],
                              [helper.make_tensor_value_info("variable", TensorProto.FLOAT, [None, 1])],
                              CONSTS)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    onnx.save(model, path)


class Engine:
    """price() in numpy with given settings, for the simulator (same numbers as the ONNX)."""
    def __init__(self, p=None, name="center"):
        self.p, self.name = dict(P, **(p or {})), name

    def quote(self, book, rnd, rfqs):
        X = np.c_[np.tile(book, (len(rfqs), 1)), np.full(len(rfqs), rnd), rfqs].astype(np.float32)
        return price(X, NUMPY, self.p)[:, 0].astype(float)


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, "submissions", "center")
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, f"{TEAM}.onnx")
    export(onnx_path)

    with open(os.path.join(root, "round_0", "TestPredict.csv"), newline="") as f:
        rows = list(csv.reader(f))
    X = np.array([r[:13] for r in rows[1:]], np.float32)
    q = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]).run(None, {"X": X})[0][:, 0]
    assert np.all(np.isfinite(q)), "a quote is not a number"
    assert np.abs(q - price(X, NUMPY)[:, 0]).max() < 1e-6, "ONNX disagrees with price()"
    assert np.all(np.abs(q - X[:, 7]) < 0.02), "a quote is off-market"
    for r, v in zip(rows[1:], q):
        r[13:] = [f"{v:.8f}"]
    with open(os.path.join(out_dir, "TestPredict.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)
    s = (q - X[:, 7]) * 1e4
    print(f"wrote {onnx_path} and TestPredict.csv ({len(q)} quotes)")
    print("quote - mid, bp, percentiles 1/10/50/90/99:", np.round(np.percentile(s, [1, 10, 50, 90, 99]), 1))
