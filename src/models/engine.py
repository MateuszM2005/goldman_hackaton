"""Quoting engine v1 for the FX market-making challenge.

    python src/models/engine.py   ->  submissions/tuned/<TEAM>.onnx  and  submissions/tuned/TestPredict.csv

We PAY the coupon we quote, so a higher quote is better for the client and earns us
less. For every RFQ:

    quote = min(mid - cost - MARGIN + discount, mid + CEILING) + relief

  cost      what the trade adds to our bill, per $ of notional: delta/vega hedging,
            exotic capital for every round left in the game, limit-breach penalties,
            and an inventory charge that grows with how full each limit already is
            (SKEW). Charged in full, so a trade we win never loses money, even when
            it pushes us over a limit.
  discount  if the trade offsets risk we already hold, it shrinks that bill, and by
            more the bigger our position is. We hand PASS_THROUGH of that saving to
            the client so we win the trade, never more than MAX_DISCOUNT, and keep
            the rest.
  CEILING   bid shading: rivals rarely pay more than about mid + CEILING, so for an
            ordinary trade quoting higher only gives edge away.
  relief    a trade that takes us back under a limit gets extra: BREACH_PASS of the
            penalty it saves, at most MAX_RELIEF, on top of the ceiling. This is the
            part of the skew that only switches on when the book is over a limit.
            PASS_THROUGH + BREACH_PASS <= 1 keeps every won trade profitable.

Fee rules are from the handout, checked against trader_pnl_by_round.csv: delta and
vega are hedged once at the end of the round; exotics stay on the book and are
charged capital AND breach penalty (100% of the excess) every round until the end.
"""
import csv
import os
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

TEAM = "team"            # the submission must be named <team_name>.onnx

# ---- knobs, in coupon units (0.001 = $100k on a $100mm trade); tuned with tune.py
MARGIN       = 0.00001   # profit we want on every trade (the ceiling already keeps >= 2.4bp)
PASS_THROUGH = 1.0       # share of an offsetting trade's saving we give to the client
MAX_DISCOUNT = 0.00174   # ...but never more than this
SKEW         = 0.05      # inventory aversion: holding a unit of risk costs (1 + SKEW) x its
                         # hedge/capital cost at a full working limit, 1 + SKEW/2 at half-full
CEILING      = -0.00024  # never pay more than mid - 2.4bp for an ordinary trade (0.02 = off)
BREACH_PASS  = 0.0       # extra share of a saved breach penalty we give, above the ceiling
MAX_RELIEF   = 0.0       # ...but never more than this (off: PASS_THROUGH is already 1)
FILL         = 0.30      # the most of an RFQ we can get: price as if we get 30% of it
BOUND        = 0.0195    # |quote - mid| > 0.02 counts as an off-market quote

# ---- desk rules, positions in $mm               delta   vega  vanna  volga    gap
UNIT_COST     = np.array([[0.006, 0.025, 0.006, 0.006, 0.006]], np.float32)  # hedge / capital per $mm
BREACH_RATE   = np.array([[0.05,  0.05,  1.00,  1.00,  1.00 ]], np.float32)  # penalty per $mm over limit
WORKING_LIMIT = np.array([[11.1,  2.25,  2.6,   2.6,   2.6  ]], np.float32)  # official: 12, 3, 3, 3, 3.
IS_EXOTIC     = np.array([[0,     0,     1,     1,     1    ]], np.float32)  # Buffer: all 32 RFQs fill at once
ONES          = np.ones((5, 1), np.float32)


def cols(*idx):
    """13 x k matrix that picks input columns, so `X @ cols(...)` works in numpy and in ONNX."""
    m = np.zeros((13, len(idx)), np.float32)
    m[list(idx), range(len(idx))] = 1
    return m


def price(X, F):
    """Quote for each row of X (n x 13, columns in the handout's order). F is NUMPY or GRAPH."""
    book = X @ cols(0, 1, 2, 3, 4)            # what we hold now
    rnd  = X @ cols(5)
    N    = F.max(X @ cols(6), 1.0)            # notional, $mm
    mid  = X @ cols(7)
    rfq  = X @ cols(8, 9, 10, 11, 12)         # risk of the whole RFQ

    # delta/vega are charged once; exotics this round and every round after it
    rounds_charged = 1 + IS_EXOTIC * (12 - F.min(F.max(rnd, 1.0), 12.0))

    def breach(pos):                          # $mm of limit penalties for holding `pos`
        return BREACH_RATE * F.relu(F.abs(pos) - WORKING_LIMIT) * rounds_charged

    def bill(pos):                            # $mm we pay for holding `pos`
        size = F.abs(pos)
        carry = UNIT_COST * size * (1 + 0.5 * SKEW * size / WORKING_LIMIT)   # dearer the fuller
        return carry * rounds_charged + breach(pos)

    after  = book + FILL * rfq                          # our book if we get our piece
    change = bill(after) - bill(book)                   # per risk, for that piece
    piece  = FILL * N                                   # $mm notional of that piece
    cost   = (F.relu(change) @ ONES) / piece            # risk it adds: charged in full
    saving = (F.relu(0.0 - change) @ ONES) / piece      # risk it removes: partly passed on
    relief = (F.relu(breach(book) - breach(after)) @ ONES) / piece   # of which, breach penalty
    quote  = mid - cost - MARGIN + F.min(PASS_THROUGH * saving, MAX_DISCOUNT)
    quote  = F.min(quote, mid + CEILING) + F.min(BREACH_PASS * relief, MAX_RELIEF)
    return F.min(F.max(quote, mid - BOUND), mid + BOUND)


# ---- run price() either on numpy arrays or as an ONNX graph --------------------
NUMPY = SimpleNamespace(abs=np.abs, relu=lambda a: np.maximum(a, 0), min=np.minimum, max=np.maximum,
                        exp=np.exp)
NODES, CONSTS = [], []


class Sym:
    """A tensor in the ONNX graph being built; + - * / @ add graph nodes."""
    __array_ufunc__ = None                    # make numpy hand `array * Sym` over to Sym

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
        if not isinstance(a, Sym):            # numbers / arrays become float32 constants
            name = f"c{len(CONSTS)}"
            CONSTS.append(numpy_helper.from_array(np.asarray(a, np.float32), name))
            a = Sym(name)
        names.append(a.name)
    out = f"t{len(NODES)}"
    NODES.append(helper.make_node(kind, names, [out]))
    return Sym(out)


GRAPH = SimpleNamespace(abs=lambda a: op("Abs", a), relu=lambda a: op("Relu", a),
                        min=lambda a, b: op("Min", a, b), max=lambda a, b: op("Max", a, b),
                        exp=lambda a: op("Exp", a))


def export(path, fn=None):
    """fn (default price) as ONNX, same input/output as benchmark.onnx: X (n x 13) -> variable (n x 1)."""
    NODES.clear()
    CONSTS.clear()
    out = (fn or price)(Sym("X"), GRAPH)
    NODES.append(helper.make_node("Identity", [out.name], ["variable"]))
    graph = helper.make_graph(NODES, "quoting_engine",
                              [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 13])],
                              [helper.make_tensor_value_info("variable", TensorProto.FLOAT, [None, 1])],
                              CONSTS)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 7                      # as benchmark.onnx, so older onnxruntime loads it
    onnx.checker.check_model(model)
    onnx.save(model, path)


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))     # the repo


def submit(out_dir, team, fn=None):
    """Export fn (default price) to out_dir/<team>.onnx, quote the organisers' test sample with
    it into out_dir/TestPredict.csv, and check it against the Python and the 0.02 bound."""
    fn = fn or price
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, f"{team}.onnx")
    export(onnx_path, fn)
    with open(os.path.join(ROOT, "data", "game_0", "TestPredict.csv"), newline="") as f:
        rows = list(csv.reader(f))
    X = np.array([r[:13] for r in rows[1:]], np.float32)
    q = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]).run(None, {"X": X})[0][:, 0]
    assert np.abs(q - fn(X, NUMPY)[:, 0]).max() < 1e-6, "ONNX disagrees with the Python"
    assert np.all(np.abs(q - X[:, 7]) < 0.02), "a quote is off-market"
    for r, v in zip(rows[1:], q):
        r[13:] = [f"{v:.8f}"]
    with open(os.path.join(out_dir, "TestPredict.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)

    s = q - X[:, 7]
    print(f"wrote {onnx_path} and TestPredict.csv ({len(q)} quotes)")
    print("quote - mid, percentiles 1/10/50/90/99:", np.round(np.percentile(s, [1, 10, 50, 90, 99]), 5))


if __name__ == "__main__":
    submit(os.path.join(ROOT, "submissions", "tuned"), TEAM)
