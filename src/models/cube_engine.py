"""Cube engine: the 32 copies of the model plan each round together.

    python src/models/cube_engine.py   ->  submissions/cube/<TEAM>.onnx  and  submissions/cube/TestPredict.csv

A round's 32 RFQs are the corners of a cube: 2 delta signs x 2 vega signs x 8 exotic corners (the
exotic part possibly turned). Every copy of the engine sees the same book and round, so each one
works out the same plan and plays its own part, without seeing the other 31 RFQs:

  1. target  where the book should end the round. Delta and vega flat (they reset anyway).
             Exotics a little ahead of next round's private move, -PRE x that move, inside the
             limits. The move is read off the organisers' catalogue of 36 exotic paths (the
             test sample) by finding the path that matches the book.
  2. share   the cube is symmetric (r and -r are both in it) with E[r r'] = diag(16, 1, 1, 1, 1),
             so if every copy plans the share f = relu(<r, w>), w = (target - book) / (16 E[r^2]),
             the 32 expected fills add up to exactly target - book, however the cube is turned.
             All shares shrink together if the biggest would exceed F_CAP.
  3. bid     the price that buys that expected share (fill curve of the calibrated market:
             first place ~55% of the RFQ, second ~25%, lower ~5%), but never more per $ than
             this RFQ's part of the plan saves (its marginal value at the planned end), less
             MARGIN. A copy whose RFQ would move us away from the target only bids at a
             profit: breakeven minus MARGIN_AWAY.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "testing")))   # backtest.py
import backtest as bt
import engine as en
from engine import cols

TEAM = "team"

# ---- knobs (coupon units: 0.0001 = 1bp); tuned with `python src/testing/tune.py --cube`
MARGIN      = 0.00068   # edge kept on a planned trade, below what the plan is worth
MARGIN_AWAY = 0.00136   # edge demanded on a trade that moves us away from the target
F_CAP       = 0.45      # biggest planned share of one RFQ (first place ~55%; past ~44% gets very dear)
PRE         = 0.32      # how far ahead of next round's private exotic move to position (0 = flat)
TARGET_LIM  = 2.03      # exotic targets stay within +-this
NEXT_COST   = 0.12      # $mm per unit to unwind an exotic position next round
H           = 1.0       # rounds an exotic breach is priced for (tuning: fix it within the round)
F0          = 0.39      # share of an RFQ used to price moving away from the target
SIGMA       = 0.30      # how closely the book must match a catalogue path
BOUND       = 0.0195    # |quote - mid| > 0.02 is off-market

# ---- desk rules, positions in $mm            delta   vega  vanna  volga    gap
UNIT_COST = np.array([[0.006, 0.025, 0.006, 0.006, 0.006]], np.float32)   # hedge / capital per $mm
BREACH    = np.array([[0.05,  0.05,  1.0,   1.0,   1.0  ]], np.float32)   # penalty per $mm over limit
LIMIT     = np.array([[11.0,  2.8,   2.8,   2.8,   2.8  ]], np.float32)   # official 12 / 3, less a buffer
EXOTIC    = np.array([[0,     0,     1,     1,     1    ]], np.float32)
MU2       = np.array([[16.2,  1.01,  1.0,   1.0,   1.0  ]], np.float32)   # E[r_k^2] over a round's cube
REACH     = np.array([[4.0], [1.0], [1.0], [1.0], [1.0]], np.float32)     # max over the cube of <r, w>
EMBED     = np.eye(5, dtype=np.float32)[2:]                               # 3 x 5: exotics into 5 risks
ONES = {k: np.ones((k, 1), np.float32) for k in (3, 5, 36)}

# expected share of an RFQ vs bid - mid in the calibrated market (rounds 1-2), inverted below
CURVE = np.array([[0.000, -0.0020], [0.008, -0.0010], [0.035, -0.0005], [0.068, -0.0002], [0.098, 0.0],
                  [0.133, 0.0002], [0.192, 0.0005], [0.297, 0.0010], [0.440, 0.0020]])

# ---- the catalogue, laid out for matrix lookups: path p, component c sits in column 3p + c
PATHS  = bt.exotic_paths().astype(np.float32)                                     # 36 x 12 x 3
NOW    = PATHS.transpose(1, 0, 2).reshape(12, -1)                                 # row k: moves in round k+1
NEXT   = np.vstack([NOW[1:], np.zeros_like(NOW[:1])])                             # row k: moves in round k+2
SPREAD = np.kron(np.ones((1, len(PATHS))), np.eye(3)).astype(np.float32)          # 3 x 108: copy to all paths
GROUP  = np.kron(np.eye(len(PATHS)), np.ones((3, 1))).astype(np.float32)          # 108 x 36: sum each path
ROUNDS = np.arange(1, 13, dtype=np.float32)[None, :]


def bid_for(f, F):
    """bid - mid that buys an expected share f: piecewise-linear through CURVE, as a sum of relus."""
    x, y = CURVE[:, 0], CURVE[:, 1]
    s = np.diff(y) / np.diff(x)
    out = float(y[0]) + float(s[0]) * f
    for k in range(1, len(s)):
        out = out + float(s[k] - s[k - 1]) * F.relu(f - float(x[k]))
    return out


def parts(X, F):
    """The round's plan (the same in all 32 copies) and this copy's part in it."""
    book, t = X @ cols(0, 1, 2, 3, 4), F.min(F.max(X @ cols(5), 1.0), 12.0)
    N, mid, r = F.max(X @ cols(6), 1.0), X @ cols(7), X @ cols(8, 9, 10, 11, 12)
    left = 13.0 - t                                   # rounds an exotic is still charged, this one too
    held = 1.0 + EXOTIC * (left - 1.0)
    hurt = 1.0 + EXOTIC * (F.min(left, H) - 1.0)

    # 1. next round's private exotic move, from the catalogue path that matches the book
    now = F.relu(1.0 - F.abs(t - ROUNDS))                            # n x 12: picks this round
    seen = 1.0 - PRE * F.min(F.relu(t - 1.0), 1.0)                   # after round 1 we sat at -PRE x it
    exo = X @ cols(2, 3, 4)
    look = (now @ NOW) * seen                                        # n x 108: book each path predicts
    dist = (exo * exo) @ ONES[3] - 2.0 * ((look * (exo @ SPREAD)) @ GROUP) + (look * look) @ GROUP
    like = F.exp(dist * (-0.5 / SIGMA ** 2))                         # n x 36
    total = like @ ONES[36]
    nxt = ((((like / (total + 1e-12)) @ GROUP.T) * (now @ NEXT)) @ SPREAD.T) * F.min(total, 1.0)

    # 2. target, and this copy's share of the move there
    target = F.min(F.max(nxt * (-PRE), -TARGET_LIM), TARGET_LIM) @ EMBED
    gap = target - book
    w = gap / (16.0 * MU2)
    scale = F.min(F_CAP / F.max(F.abs(w) @ REACH, 1e-9), 1.0)
    share = F.relu((r * w) @ ONES[5]) * scale
    end = book + gap * scale                                         # where the expected fills land us

    # 3. bids
    def bill(pos):                                                   # $mm: this round, plus unwinding
        size = F.abs(pos)
        fees = (UNIT_COST * size * held + BREACH * F.relu(size - LIMIT) * hurt) @ ONES[5]
        return fees + (F.abs(pos @ EMBED.T + nxt) @ ONES[3]) * NEXT_COST

    piece = F.max(share, 0.02)
    worth = (bill(end - r * piece) - bill(end)) / (N * piece)        # what this RFQ's part saves per $
    harm = (bill(end + r * F0) - bill(end)) / (N * F0)               # cost per $ of moving away
    on = F.min(share * 1e4, 1.0)                                     # 1 if this copy has a part
    bid = on * F.min(bid_for(share, F), worth - MARGIN) + (1.0 - on) * (0.0 - harm - MARGIN_AWAY)
    return dict(quote=F.min(F.max(mid + bid, mid - BOUND), mid + BOUND), share=share, end=end,
                target=target, nxt=nxt, scale=scale, gap=gap)


def price(X, F):
    return parts(X, F)["quote"]


if __name__ == "__main__":
    en.submit(os.path.join(en.ROOT, "submissions", "cube"), TEAM, price)
