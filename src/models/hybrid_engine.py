"""Hybrid engine: the tuned engine's pricing while exotics are small, the cube plan when they are big.

    python src/models/hybrid_engine.py   ->  submissions/hybrid/<TEAM>.onnx  and  TestPredict.csv

Calm (every exotic, now and after next round's expected private move, within LO): engine.py's tuned
pricing, which earns edge. Storm (any of them past HI): cube_engine.py's plan, which moves the whole
book where it should be and stays out of breach. In between, a straight-line blend of the two quotes.
The size check uses the cube's catalogue lookup of next round's private exotic move.

Tuned in arena (src/testing/arena.py) on fresh markets at every stage. On 300 unseen arena games it
made -4.29 a game against -5.65 for the cube alone and -10.27 for the calm engine alone, with +0.44 in
the game 1 replay and -3.64 in the game 2 replay (real: +0.13 and -86.24).
"""
import os

import numpy as np

import cube_engine as ce
import engine as en
from engine import cols

TEAM = "team"

LO, HI = 2.2, 3.3      # largest exotic (now or next round) where the cube starts / fully takes over
STORM = dict(MARGIN=0.00061, MARGIN_AWAY=0.00142, F_CAP=0.45, PRE=0.25, NEXT_COST=0.078, H=1.03)
PICK = [np.eye(3, dtype=np.float32)[:, [k]] for k in range(3)]                       # column picks


def price(X, F):
    for k, v in STORM.items():                                            # the cube's storm settings
        setattr(ce, k, v)
    calm = en.price(X, F)
    cube = ce.parts(X, F)
    exo = X @ cols(2, 3, 4)
    size = F.max(F.abs(exo), F.abs(exo + cube["nxt"]))                    # n x 3: now, or next round
    big = F.max(F.max(size @ PICK[0], size @ PICK[1]), size @ PICK[2])     # n x 1: the largest
    storm = F.min(F.max((big - LO) / (HI - LO), 0.0), 1.0)
    return calm + storm * (cube["quote"] - calm)


if __name__ == "__main__":
    en.submit(os.path.join(en.ROOT, "submissions", "hybrid"), TEAM, price)
