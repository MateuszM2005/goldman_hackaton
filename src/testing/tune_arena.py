"""Fine-tune the center engine (src/models/engine_center.py) in the arena backtester.

    python src/testing/tune_arena.py          # search, refine, validate; ~3 min on 8 cores
    python src/testing/tune_arena.py 1234     # repeat a run: the seed it printed

A fine-tune, not a fresh search: it starts from the engine's current settings
(engine_center.P) and only looks nearby. Every stage plays NEW arena games: the organisers'
shared RFQ draws under random cube turns, every private-path size, both fitted rival fields,
jittered. Within a stage every candidate meets exactly the same games, so they are compared
like for like:
  1. search    the current settings + 300 nearby ones, 96 games each
  2. refine    the best 15 + 8 smaller variations of each, 200 new games
  3. validate  the best 8 and the current settings on 1000 new games, plus the real games 1
               and 2 replayed 20 times against the competition we observed (arena --replay)
A candidate replaces the current settings only if it beats them by more than two standard
errors on the validation games, is no worse on either real game and adds no breaches. Of
those, the best lower confidence bound (gain - 2 se) wins, so a finalist that was merely
lucky on the validation games is not picked over a steadier one. The winner is written to
submissions/center_tuned/: Shame Street.onnx, TestPredict.csv and settings.json.
engine_center.py itself is not changed.
"""
import csv
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")))   # engine_center.py
import arena as A              # noqa: E402
import engine_center as ec     # noqa: E402

SPACE = {                      # the same knobs and ranges as tune_center.py
    "MARGIN":       (0.0, 0.0015),
    "PASS":         (0.0, 1.0),
    "MAX_DISCOUNT": (0.0, 0.004),
    "CEILING":      (-0.001, 0.0005),
    "RELIEF_PASS":  (0.0, 1.0),       # kept <= 1 - PASS, so a won trade never loses
    "MAX_RELIEF":   (0.0, 0.005),
    "BREACH_H":     (1.0, 12.0),
    "CAPITAL_H":    (1.0, 12.0),
    "MOVE_DV":      (0.1, 4.0),
    "MOVE_EX":      (0.1, 4.0),
    "SKEW":         (0.0, 2.0),
    "WL_DELTA":     (8.0, 12.0),
    "WL_VEGA":      (2.0, 3.0),
    "WL_EXOTIC":    (1.5, 3.5),
}
WORKERS = min(8, os.cpu_count() or 4)
REPLAY_REPS = 20
TOL = {1: 0.05, 2: 0.5}        # $mm a winner may lose vs the current settings on real games 1 / 2
OUT = os.path.join(A.ROOT, "submissions", "center_tuned")


def fix(s):
    s["RELIEF_PASS"] = min(s["RELIEF_PASS"], 1.0 - s["PASS"])
    return s


def engine(p, name="candidate"):
    return A.Function(lambda X: ec.price(X.astype(np.float32), ec.NUMPY, p)[:, 0], name)


_W = {}


def init():
    """Per worker: load the data once and draw the real-game replays (same for every candidate)."""
    A.catalogue()
    A.Shares(A.FIELD)
    _W["replays"] = {}
    for k in (1, 2):
        rng = np.random.default_rng(100 + k)
        rp = A.Replay(k, A.field_for(k), rng)
        _W["replays"][k] = (rp, [(rp.competition(rng), A.Shares(rp.field).draw(rng, (12, 32)))
                                 for _ in range(REPLAY_REPS)])


def synth(task):
    """One candidate on a list of arena games: rows of (P&L, breaches, round sd, path size)."""
    p, seed, games = task
    e = engine(p)
    _, fields = A.regimes()
    out = []
    for gi in games:
        rng = np.random.default_rng([seed, gi])                        # same game for every candidate
        field = A.jitter_field(rng, fields[int(rng.integers(len(fields)))], 1.0)
        mk = A.Market(rng, field, 1, draws="shared", turn="mix")
        t = A.totals(A.play(mk, [e])[0])
        out.append((t["pnl"], t["breaches"], t["sd"], A.path_size(mk.paths[0])))
    return np.array(out)


def replay(task):
    """One candidate in real game k, mean P&L over the fixed replays."""
    p, k = task
    rp, draws = _W["replays"][k]
    e = engine(p)
    return float(np.mean([sum(r["pnl"] for r in rp.play(e, None, comp, sh)) for comp, sh in draws]))


def evaluate(pool, cands, seed, n_games, chunks=1):
    """(candidates, games) results; games split into chunks so few candidates still use every core."""
    idx = np.array_split(np.arange(n_games), chunks)
    res = pool.map(synth, [(c, seed, list(g)) for c in cands for g in idx])
    return [np.vstack(res[i * chunks:(i + 1) * chunks]) for i in range(len(cands))]


def show(label, cands, res, base):
    order = np.argsort([-r[:, 0].mean() for r in res])
    b = res[base][:, 0]
    print(f"{label}: best {res[order[0]][:, 0].mean():+.3f} per game, current {b.mean():+.3f}"
          f" ({len(res[0])} games x {len(cands)} settings)", flush=True)
    return order


def export(p, metrics):
    """Write the tuned engine like engine_center.py's main does, plus its settings."""
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{ec.TEAM}.onnx")
    ec.export(path, p)
    with open(os.path.join(A.ROOT, "data", "game_0", "TestPredict.csv"), newline="") as f:
        rows = list(csv.reader(f))
    X = np.array([r[:13] for r in rows[1:]], np.float32)
    q = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"X": X})[0][:, 0]
    assert np.all(np.isfinite(q)), "a quote is not a number"
    assert np.abs(q - ec.price(X, ec.NUMPY, p)[:, 0]).max() < 1e-6, "ONNX disagrees with price()"
    assert np.all(np.abs(q - X[:, 7]) < 0.02), "a quote is off-market"
    for r, v in zip(rows[1:], q):
        r[13:] = [f"{v:.8f}"]
    with open(os.path.join(OUT, "TestPredict.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)
    with open(os.path.join(OUT, "settings.json"), "w") as f:
        json.dump(dict(settings=p, base="engine_center.P", validation=metrics), f, indent=1)
    print(f"wrote {path}, TestPredict.csv ({len(q)} quotes) and settings.json")


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else int(time.time()) % 100_000
    print("seed", seed, flush=True)
    rng = np.random.default_rng(seed)
    now = fix({k: float(ec.P[k]) for k in SPACE})

    def near(s, scale):
        """A nearby setting: all knobs a little, or one to three knobs more."""
        s = dict(s)
        keys = list(SPACE) if rng.random() < 0.5 else list(rng.choice(list(SPACE), rng.integers(1, 4), replace=False))
        step = scale if len(keys) == len(SPACE) else 2.5 * scale
        for k in keys:
            lo, hi = SPACE[k]
            s[k] = float(np.clip(s[k] + rng.normal(0, step * (hi - lo)), lo, hi))
        return fix(s)

    t0 = time.time()
    with mp.Pool(WORKERS, initializer=init) as pool:
        c1 = [now] + [near(now, 0.06) for _ in range(300)]
        r1 = evaluate(pool, c1, seed * 10 + 1, 96)
        o1 = show(f"1 search ({time.time() - t0:.0f}s)", c1, r1, 0)

        top = [c1[i] for i in o1[:15]]
        c2 = top + [near(c, 0.03) for c in top for _ in range(8)]
        if now not in c2:
            c2 = [now] + c2
        r2 = evaluate(pool, c2, seed * 10 + 2, 200)
        o2 = show(f"2 refine ({time.time() - t0:.0f}s)", c2, r2, c2.index(now))

        c3 = [now] + [c2[i] for i in o2 if c2[i] is not now][:8]
        r3 = evaluate(pool, c3, seed * 10 + 3, 1000, chunks=8)
        g = pool.map(replay, [(c, k) for c in c3 for k in (1, 2)])
        g = np.array(g).reshape(len(c3), 2)
        show(f"3 validate ({time.time() - t0:.0f}s)", c3, r3, 0)

    base = r3[0][:, 0]
    sizes = r3[0][:, 3]
    print("\nvalidation: 1000 new arena games and the real games replayed; $mm per game")
    print(f"{'':10s}{'P&L':>8s}{'vs current':>16s}{'breaches':>10s}{'round sd':>10s}"
          f"{'size .5':>9s}{'size 1':>8s}{'size 2':>8s}{'game 1':>8s}{'game 2':>8s}   changed settings")
    best, best_i, finalists = None, 0, []
    for i, (c, r) in enumerate(zip(c3, r3)):
        d = r[:, 0] - base
        se = d.std() / np.sqrt(len(d))
        by = [r[sizes == s, 0].mean() for s in (0.5, 1.0, 2.0)]
        ok = (i > 0 and d.mean() > 2 * se and g[i, 0] >= g[0, 0] - TOL[1] and g[i, 1] >= g[0, 1] - TOL[2]
              and r[:, 1].mean() <= r3[0][:, 1].mean())
        diff = ", ".join(f"{k} {now[k]:.4g}->{c[k]:.4g}" for k in SPACE if abs(c[k] - now[k]) > 1e-9 * max(1, abs(now[k])))
        tag = "current" if i == 0 else ("passes" if ok else "")
        print(f"{tag:10s}{r[:, 0].mean():+8.3f}{d.mean():+10.3f}+-{se:.3f}{r[:, 1].mean():10.2f}{r[:, 2].mean():10.3f}"
              f"{by[0]:+9.2f}{by[1]:+8.2f}{by[2]:+8.2f}{g[i, 0]:+8.2f}{g[i, 1]:+8.2f}   {diff if i else ''}")
        finalists.append(dict(settings=c, pnl=float(r[:, 0].mean()), vs_current=float(d.mean()), se=float(se),
                              breaches=float(r[:, 1].mean()), round_sd=float(r[:, 2].mean()),
                              game1_replay=float(g[i, 0]), game2_replay=float(g[i, 1]), passes=bool(ok)))
        if ok and (best is None or d.mean() - 2 * se > best):      # best lower confidence bound
            best, best_i = d.mean() - 2 * se, i
    if best is None:
        print("\nno candidate beat the current settings clearly; nothing written")
        return
    p = c3[best_i]
    r = r3[best_i]
    gain = float((r[:, 0] - base).mean())
    metrics = dict(seed=seed, games=len(r), pnl=float(r[:, 0].mean()), vs_current=gain,
                   breaches=float(r[:, 1].mean()), round_sd=float(r[:, 2].mean()),
                   game1_replay=float(g[best_i, 0]), game2_replay=float(g[best_i, 1]),
                   current=dict(pnl=float(base.mean()), game1_replay=float(g[0, 0]), game2_replay=float(g[0, 1])),
                   finalists=finalists)
    print(f"\nwinner: {gain:+.3f} $mm per game better than the current settings")
    print("settings:", json.dumps({k: round(v, 6) for k, v in p.items()}))
    export(p, metrics)


if __name__ == "__main__":
    main()
