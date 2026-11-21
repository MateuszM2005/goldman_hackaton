"""Need-vector engine: every RFQ priced by its projection on the book's need vector.

    python src/models/need_engine.py   ->  submissions/need/Shame Street.onnx  and  TestPredict.csv

The round as one optimisation problem, solved exactly under its assumptions:
  cost   the end-of-round fee bill with the OFFICIAL limits and the penalty counted once:
             C(B) = sum_i w_i |B_i| + pi_i max(|B_i| - L_i, 0)
         w = (0.006, 0.025, 0.006 h, 0.006 h, 0.006 h), h = 13 - t rounds of exotic capital,
         pi = (0.05, 0.05, 1, 1, 1), L = (12, 3, 3, 3, 3). Fills are random, so the engine uses
         E[C] with the end book normal(B, sigma^2), sigma from the fill noise of the round's bids.
  need   n(B) = -grad E[C](B): what one more unit moved in each risk saves.
             n_i = -w_i (2 Phi(B_i/s_i) - 1) - pi_i (Phi((B_i - L_i)/s_i) - Phi((-B_i - L_i)/s_i))
  bid    the optimum (KKT) bids each RFQ as its own sealed auction with value
             v = n*.r / N        n* = n(B*), B* = the planned end-of-round book
             x = argmax_x  phi(x) (n*.r - N x) - kappa Var_f(x)        phi: expected fill at mid + x
         The second term is the exact second-order cost of the fill's randomness: it widens
         the end book, kappa = 1/2 sum_i Gamma_i r_i^2 with Gamma_i = d2 E[C] / d mu_i^2 at B*
         (largest near zero and near the limits). VARIANCE = False drops it.
         RFQs parallel to the need get bid up and contra-parallel ones down, in proportion to
         how big the need is. x*(v), phi*(v) and var*(v) come from src/testing/fill_curve.py
         (the arena's fitted rivals; first place ~55% of the RFQ, second ~25%).
  plan   all 32 copies find the same n* from (book, round) alone. The round's RFQs are the
         known cube (sign patterns of rho = (4, 1, 1, 1, 1)), so the expected move in risk i
         for a need n is M_i(n) = 16 rho_i (phi*(n rho_i / N) - phi*(-n rho_i / N)). The game's
         rivals are like game 1's or game 2's (the same all game, so a shortfall does not
         average out), so the expected cost averages both end books and
             n*_i = (n_i(b_i + M1_i(n*_i)) + n_i(b_i + M2_i(n*_i))) / 2
         is one increasing equation per risk, solved by bisection (STEPS halvings).
Only abs, relu, min, max, sign, sqrt, erf, reshape and matrix products, so the ONNX graph
computes exactly this.
"""
import csv
import json
import os
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

TEAM = "Shame Street"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "need_curve.json")) as f:
    CURVE = json.load(f)


def f32(a):
    return np.asarray(a, np.float32)


# ---- the handout's fee rules: official limits, penalty counted once     delta  vega  vanna  volga  gap
W   = f32([[0.006, 0.025, 0.006, 0.006, 0.006]])     # hedge (delta, vega) or capital (exotics) per unit
PEN = f32([[0.05,  0.05,  1.0,   1.0,   1.0  ]])     # penalty per unit over the limit
LIM = f32([[12.0,  3.0,   3.0,   3.0,   3.0  ]])
EXO = f32([[0,     0,     1,     1,     1    ]])

# ---- the round's cube of RFQs, the same in every copy
RHO     = f32([[4.0, 1.0, 1.0, 1.0, 1.0]])            # typical size of each risk on one RFQ
NBAR    = 99.5                                         # typical notional (90-109)
CORNERS = f32([[1 if k >> (4 - i) & 1 else -1 for i in range(5)] for k in range(32)]) * RHO   # 32 x 5
SIGMA0  = f32([[3.0, 0.75, 0.75, 0.75, 0.75]])        # fill noise before the round's plan is known
STEPS   = 26                                           # bisection halvings (float32 precision)
VARIANCE = True                                        # charge each bid for the noise its fill adds
REGIME   = True                                        # plan for both fitted rival fields (game 1's, game 2's)
BOUND   = 0.0195                                       # |quote - mid| > 0.02 is off-market
ONES    = np.ones((5, 1), np.float32)
CAP     = 16 * RHO * f32(max(CURVE["phi"])) + 1e-3      # largest expected one-sided move in a round


def _table(key):
    """CURVE[key] as y(v) = y0 + sum_k d_k relu(v - v_k), flat outside the knots."""
    v, y = np.asarray(CURVE["v"], float), np.asarray(CURVE[key], float)
    s = np.diff(y) / np.diff(v)
    return dict(lo=float(v[0]), hi=float(v[-1]), k=f32(v[:-1])[None, :], y0=float(y[0]),
                d=f32(np.r_[s[0], np.diff(s)])[:, None])


TAB = {key: _table(key) for key in ("x", "phi", "var", "phi1", "phi2")}

# candidate bids for the per-RFQ argmax: every 0.5bp from -40 to +40bp, then a few far ones
_c = CURVE["curve"]
_xs = np.r_[-195.0, -120.0, -80.0, -60.0, np.arange(-40.0, 40.01, 0.5), 50.0, 60.0, 80.0, 100.0, 150.0, 199.0] * 1e-4
XG = f32(_xs)[None, :]                                                             # 1 x G bids
PG = f32(np.interp(_xs, _c["x"], _c["phi"]))[None, :]                             # expected fill
VG = f32(np.maximum(np.interp(_xs, _c["x"], _c["ef2"]) - PG[0].astype(float) ** 2, 0))[None, :]   # fill variance


def cols(*idx):
    m = np.zeros((13, len(idx)), np.float32)
    m[list(idx), range(len(idx))] = 1
    return m


def _erf_np(z):
    """Abramowitz-Stegun 7.1.26, |error| < 1.5e-7 (the ONNX graph uses the exact Erf)."""
    a = np.abs(z)
    t = 1 / (1 + 0.3275911 * a)
    y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return np.sign(z) * y


def _pwl_np(v, key, ncols):
    t = TAB[key]
    vc = np.clip(v, t["lo"], t["hi"])
    return t["y0"] + (np.maximum(vc[..., None] - t["k"][0], 0) @ t["d"])[..., 0]


NUMPY = SimpleNamespace(abs=np.abs, relu=lambda a: np.maximum(a, 0), min=np.minimum, max=np.maximum,
                        sign=np.sign, sqrt=np.sqrt, erf=_erf_np, exp=np.exp, pwl=_pwl_np,
                        best=lambda obj: XG[0][np.argmax(obj, axis=1)][:, None])


def Phi(z, F):
    return 0.5 + 0.5 * F.erf(z * float(1 / np.sqrt(2.0)))       # python floats keep float32


def pdf(z, F):
    return F.exp(z * z * -0.5) * float(1 / np.sqrt(2 * np.pi))


def need(B, sig, wv, F):
    """n = -grad E[C] at the expected end book B with fill noise sig: what a unit moved saves."""
    hold = 2.0 * Phi(B / sig, F) - 1.0
    over = Phi((B - LIM) / sig, F) - Phi((0.0 - B - LIM) / sig, F)
    return 0.0 - (wv * hold + PEN * over)


def move(n, F, key="phi"):
    """Expected move of each risk over the round's cube when every copy bids for need n."""
    v = n * (RHO / NBAR)
    return (16 * RHO) * (F.pwl(v, key, 5) - F.pwl(0.0 - v, key, 5))


def noise(n0, F):
    """Std of each risk's end-of-round position from the fills' randomness, floored at 0.1 rho."""
    var = F.pwl(n0 @ (CORNERS.T / NBAR), "var", 32) @ (CORNERS * CORNERS)
    return F.sqrt(var + (0.1 * RHO) ** 2)


def plan(book, wv, F):
    """The need n* and the planned end-of-round book(s): the same in all 32 copies.
    REGIME: the game's rivals are like game 1's or like game 2's, half each and the same all game,
    so a shortfall does not average out. The expected cost averages the two end books
    b + M1(n) and b + M2(n), and the need solves
        n = (n(b + M1(n)) + n(b + M2(n))) / 2
    one increasing equation per risk, by bisection on n. Without REGIME: one end book with the
    average fill curve, B = b + M(n(B)), by bisection on B."""
    sig = noise(need(book, SIGMA0, wv, F), F)
    if REGIME:
        lo, hi = 0.0 - (wv + PEN) - 1e-3, (wv + PEN) + 1e-3          # the need is never bigger
        for _ in range(STEPS):
            nm = (lo + hi) * 0.5
            h = nm - 0.5 * (need(book + move(nm, F, "phi1"), sig, wv, F) + need(book + move(nm, F, "phi2"), sig, wv, F))
            s = (F.sign(h) + 1.0) * 0.5                               # 1 where the root is below nm
            hi = s * nm + (1.0 - s) * hi
            lo = s * lo + (1.0 - s) * nm
        n_star = (lo + hi) * 0.5
        return [book + move(n_star, F, "phi1"), book + move(n_star, F, "phi2")], n_star, sig
    lo, hi = book - CAP, book + CAP
    for _ in range(STEPS):
        mb = (lo + hi) * 0.5
        h = mb - book - move(need(mb, sig, wv, F), F)                 # increasing in mb
        s = (F.sign(h) + 1.0) * 0.5                                   # 1 where the root is below mb
        hi = s * mb + (1.0 - s) * hi
        lo = s * lo + (1.0 - s) * mb
    end = (lo + hi) * 0.5
    return [end], need(end, sig, wv, F), sig


def price(X, F):
    """Quote for each row of X (n x 13, the handout's columns). F is NUMPY or GRAPH."""
    book = X @ cols(0, 1, 2, 3, 4)
    rnd  = F.min(F.max(X @ cols(5), 1.0), 12.0)
    N    = F.max(X @ cols(6), 1.0)
    mid  = X @ cols(7)
    rfq  = X @ cols(8, 9, 10, 11, 12)
    wv = W * (1.0 + EXO * (12.0 - rnd))                               # exotics: 13 - t rounds of capital
    ends, n_star, sig = plan(book, wv, F)
    val = (n_star * rfq) @ ONES                                       # what a full fill saves, $mm
    if VARIANCE:
        gs = [(2.0 * wv * pdf(e / sig, F) + PEN * (pdf((e - LIM) / sig, F) + pdf((0.0 - e - LIM) / sig, F))) / sig
              for e in ends]
        gamma = gs[0] if len(gs) == 1 else (gs[0] + gs[1]) * 0.5      # curvature of E[C] at the plan
        kappa = ((gamma * rfq * rfq) @ ONES) * 0.5                     # $mm per unit of fill variance
        obj = (val - N * XG) * PG - kappa * VG                        # n x G: each candidate bid
        quote = mid + F.best(obj)
    else:
        quote = mid + F.pwl(val / N, "x", 1)                          # best bid against the fill curve
    return F.min(F.max(quote, mid - BOUND), mid + BOUND)


# ---- ONNX ---------------------------------------------------------------------------------
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


def op(kind, *args, **attrs):
    names = []
    for a in args:
        if not isinstance(a, Sym):
            a = np.asarray(a)
            name = f"c{len(CONSTS)}"
            CONSTS.append(numpy_helper.from_array(a if a.dtype == np.int64 else a.astype(np.float32), name))
            a = Sym(name)
        names.append(a.name)
    out = f"t{len(NODES)}"
    NODES.append(helper.make_node(kind, names, [out], **attrs))
    return Sym(out)


def _pwl_graph(v, key, ncols):
    t = TAB[key]
    vc = op("Min", op("Max", v, t["lo"]), t["hi"])
    z = op("Relu", op("Sub", op("Reshape", vc, np.array([-1, ncols, 1], np.int64)), t["k"]))
    return op("Add", op("Reshape", op("MatMul", z, t["d"]), np.array([-1, ncols], np.int64)), t["y0"])


GRAPH = SimpleNamespace(abs=lambda a: op("Abs", a), relu=lambda a: op("Relu", a),
                        min=lambda a, b: op("Min", a, b), max=lambda a, b: op("Max", a, b),
                        sign=lambda a: op("Sign", a), sqrt=lambda a: op("Sqrt", a),
                        erf=lambda a: op("Erf", a), exp=lambda a: op("Exp", a), pwl=_pwl_graph,
                        best=lambda obj: op("Gather", XG[0], op("ArgMax", obj, axis=1, keepdims=1)))


def export(path):
    """Same interface as benchmark.onnx: X (n x 13 float) -> variable (n x 1 float)."""
    NODES.clear()
    CONSTS.clear()
    out = price(Sym("X"), GRAPH)
    NODES.append(helper.make_node("Identity", [out.name], ["variable"]))
    graph = helper.make_graph(NODES, "quoting_engine_need",
                              [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 13])],
                              [helper.make_tensor_value_info("variable", TensorProto.FLOAT, [None, 1])],
                              CONSTS)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    onnx.save(model, path)


def submit(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{TEAM}.onnx")
    export(path)
    with open(os.path.join(ROOT, "data", "game_0", "TestPredict.csv"), newline="") as f:
        rows = list(csv.reader(f))
    X = np.array([r[:13] for r in rows[1:]], np.float32)
    q = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"X": X})[0][:, 0]
    assert np.all(np.isfinite(q)), "a quote is not a number"
    p = np.concatenate([price(X[i:i + 8192].astype(np.float64), NUMPY)[:, 0] for i in range(0, len(X), 8192)])   # exact maths
    d = np.abs(q - p)    # float32 may break an exact tie between two candidate bids the other way: one 0.5bp step
    assert np.mean(d > 1e-6) < 1e-4 and d.max() < 0.5e-4 + 1e-6, "ONNX disagrees with price()"
    assert np.all(np.abs(q - X[:, 7]) < 0.02), "a quote is off-market"
    for r, v in zip(rows[1:], q):
        r[13:] = [f"{v:.8f}"]
    with open(os.path.join(out_dir, "TestPredict.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)
    s = (q - X[:, 7]) * 1e4
    print(f"wrote {path} and TestPredict.csv ({len(q)} quotes)")
    print("quote - mid, bp, percentiles 1/10/50/90/99:", np.round(np.percentile(s, [1, 10, 50, 90, 99]), 1))


if __name__ == "__main__":
    submit(os.path.join(ROOT, "submissions", "need"))
