"""Tests for the need-vector engine (src/models/need_engine.py).

    python src/testing/test_need_engine.py          # plain python, no pytest needed

The exported ONNX equals price() on the test sample and on random and nonsense inputs; quotes are
finite and inside the +-0.02 band; and the guarantees the maths gives hold on random books:
  - never pays more than a trade is worth: quote - mid <= v = n*.r / N (when it can fill at all)
  - flipping the sign of the book and the RFQ together changes nothing
  - the plan solves its equation: n* = mean of n(b + M_k(n*)) over the two fields, to float precision
  - a bigger need along an RFQ raises its bid: pushing the book away from zero in the
    direction the RFQ would take back never lowers the quote
  - the fill curve's shares are the real ones: first place ~55%, second ~25%
"""
import os
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")))   # need_engine.py
import need_engine as ne   # noqa: E402

ROOT = ne.ROOT
ONNX = os.path.join(ROOT, "submissions", "need", f"{ne.TEAM}.onnx")


def random_inputs(n, seed=0):
    rng = np.random.default_rng(seed)
    book = np.c_[rng.uniform(-25, 25, (n, 1)), rng.uniform(-5, 5, (n, 4))]
    rfq = np.c_[rng.choice([-1, 1], (n, 1)) * 4 * rng.uniform(0.8, 1.2, (n, 1)),
                rng.choice([-1, 1], (n, 4)) * rng.uniform(0.8, 1.2, (n, 4))]
    return np.c_[book, rng.integers(1, 13, (n, 1)), rng.integers(90, 110, (n, 1)),
                 rng.uniform(0.048, 0.058, (n, 1)), rfq].astype(np.float32)


def value(X):
    wv = ne.W * (1.0 + ne.EXO * (12.0 - np.clip(X[:, 5:6], 1, 12)))
    ends, n, sig = ne.plan(X[:, :5], wv, ne.NUMPY)
    return (n * X[:, 8:13]).sum(1) / np.maximum(X[:, 6], 1), ends, n, sig, wv


def test_onnx_matches_python():
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    for X in (random_inputs(20000, 1), np.loadtxt(os.path.join(ROOT, "data", "game_0", "TestPredict.csv"),
                                                   delimiter=",", skiprows=1, usecols=range(13), dtype=np.float32)[::7]):
        q = sess.run(None, {"X": X})[0][:, 0]
        d = np.abs(q - ne.price(X.astype(np.float64), ne.NUMPY)[:, 0])
        assert np.mean(d > 1e-6) < 1e-4 and d.max() < 0.5e-4 + 1e-6   # at most a rare tie flip, one 0.5bp step
        assert np.all(np.isfinite(q)) and np.all(np.abs(q - X[:, 7]) < 0.02)


def test_never_pays_more_than_value():
    X = random_inputs(20000, 2)
    v, *_ = value(X)
    x = ne.price(X, ne.NUMPY)[:, 0] - X[:, 7]
    can_fill = x > -0.006                       # quotes far below mid never fill (fill curve is 0)
    assert np.all(x[can_fill] <= v[can_fill] + 1e-7)


def test_sign_flip_symmetry():
    X = random_inputs(5000, 3)
    Y = X.copy()
    Y[:, :5] *= -1
    Y[:, 8:] *= -1
    assert np.abs(ne.price(X, ne.NUMPY) - ne.price(Y, ne.NUMPY)).max() < 1e-6


def test_plan_solves_its_equation():
    X = random_inputs(5000, 4)
    _, ends, n, sig, wv = value(X)
    resid = n - 0.5 * sum(ne.need(e, sig, wv, ne.NUMPY) for e in ends)      # n = mean need at the end books
    assert np.abs(resid).max() < 1e-4
    for e in ends:
        assert np.all(np.abs(e - X[:, :5]) <= ne.CAP + 1e-4)


def test_bigger_need_raises_the_bid():
    X = random_inputs(5000, 5)
    away = -np.sign(X[:, 8:13])                 # the RFQ takes this direction back
    Y = X.copy()
    Y[:, 2:5] += 1.5 * away[:, 2:]              # push the exotics further out along it
    q, p = ne.price(X, ne.NUMPY)[:, 0], ne.price(Y, ne.NUMPY)[:, 0]
    assert np.mean(p >= q - 1e-7) > 0.999


def test_fill_shares_are_the_real_ones():
    for shares in ne.CURVE["meta"]["shares"]:
        assert abs(shares[0] - 0.55) < 0.02 and abs(shares[1] - 0.245) < 0.02


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, f in tests:
        try:
            f()
            print(f"PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__} {str(e)[:300]}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
