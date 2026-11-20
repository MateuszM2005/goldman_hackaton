"""Quoting engine v1 for the FX market-making challenge.

    python engine.py   ->  submission/<TEAM>.onnx  and  submission/TestPredict.csv

We PAY the coupon we quote, so a higher quote is better for the client and earns us
less. For every RFQ:

    quote = mid - cost - MARGIN + discount

  cost      what the trade adds to our bill, per $ of notional: delta/vega hedging,
            exotic capital for every round left in the game, and limit-breach
            penalties. Charged in full, so a trade we win never loses money, even
            when it pushes us over a limit.
  discount  if the trade offsets risk we already hold, it shrinks our bill. We hand
            PASS_THROUGH of that saving to the client so we win the trade, never
            more than MAX_DISCOUNT, and keep the rest.

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

# ---- knobs, in coupon units (0.001 = $100k on a $100mm trade) -----------------
MARGIN       = 0.001     # profit we want on every trade
PASS_THROUGH = 0.5       # share of an offsetting trade's saving we give to the client
MAX_DISCOUNT = 0.002     # ...but never more than this, so we never quote above mid + 0.001
FILL         = 0.25      # share of an RFQ we get when we win it (example game: 2.5-25%)
BOUND        = 0.0195    # |quote - mid| > 0.02 counts as an off-market quote

# ---- desk rules, positions in $mm               delta   vega  vanna  volga    gap
UNIT_COST     = np.array([[0.006, 0.025, 0.006, 0.006, 0.006]], np.float32)  # hedge / capital per $mm
BREACH_RATE   = np.array([[0.05,  0.05,  1.00,  1.00,  1.00 ]], np.float32)  # penalty per $mm over limit
WORKING_LIMIT = np.array([[10.0,  2.5,   2.0,   2.0,   2.0  ]], np.float32)  # official: 12, 3, 3, 3, 3.
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

    def bill(pos):                            # $mm we pay for holding `pos`
        size = F.abs(pos)
        return (UNIT_COST * size + BREACH_RATE * F.relu(size - WORKING_LIMIT)) * rounds_charged

    change = bill(book + FILL * rfq) - bill(book)      # per risk, for the piece we would get
    piece  = FILL * N                                   # $mm notional of that piece
    cost   = (F.relu(change) @ ONES) / piece            # risk it adds: charged in full
    saving = (F.relu(0.0 - change) @ ONES) / piece      # risk it removes: partly passed on
    quote  = mid - cost - MARGIN + F.min(PASS_THROUGH * saving, MAX_DISCOUNT)
    return F.min(F.max(quote, mid - BOUND), mid + BOUND)


# ---- run price() either on numpy arrays or as an ONNX graph --------------------
NUMPY = SimpleNamespace(abs=np.abs, relu=lambda a: np.maximum(a, 0), min=np.minimum, max=np.maximum)
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
                        min=lambda a, b: op("Min", a, b), max=lambda a, b: op("Max", a, b))


def export(path):
    """Same input/output as benchmark.onnx: X (n x 13 float) -> variable (n x 1 float)."""
    NODES.clear()
    CONSTS.clear()
    out = price(Sym("X"), GRAPH)
    NODES.append(helper.make_node("Identity", [out.name], ["variable"]))
    graph = helper.make_graph(NODES, "quoting_engine",
                              [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 13])],
                              [helper.make_tensor_value_info("variable", TensorProto.FLOAT, [None, 1])],
                              CONSTS)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 7                      # as benchmark.onnx, so older onnxruntime loads it
    onnx.checker.check_model(model)
    onnx.save(model, path)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "submission")
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, f"{TEAM}.onnx")
    export(onnx_path)

    # quote the organisers' test sample with the exported file, and check it
    with open(os.path.join(here, "TestPredict.csv"), newline="") as f:
        rows = list(csv.reader(f))
    X = np.array([r[:13] for r in rows[1:]], np.float32)
    q = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]).run(None, {"X": X})[0][:, 0]
    assert np.abs(q - price(X, NUMPY)[:, 0]).max() < 1e-6, "ONNX disagrees with price()"
    assert np.all(np.abs(q - X[:, 7]) < 0.02), "a quote is off-market"
    for r, v in zip(rows[1:], q):
        r[13:] = [f"{v:.8f}"]
    with open(os.path.join(out_dir, "TestPredict.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)

    s = q - X[:, 7]
    print(f"wrote {onnx_path} and TestPredict.csv ({len(q)} quotes)")
    print("quote - mid, percentiles 1/10/50/90/99:", np.round(np.percentile(s, [1, 10, 50, 90, 99]), 5))
