"""
Symulator gry do testowania strategii przed oddaniem.
 
- RFQ: te same 384 zapytania co w dotychczasowych grach (plik z kwotowaniami)
- prywatne pozycje: z prawdziwych gier albo syntetyczne (cykle z danych)
- konkurencja: losowana z rozkładów oszacowanych na grach 1-2
- koszty: pricing_v2.state_cost, zwalidowany co do 1e-16 z P&L organizatorów
 
Strategia to dowolna funkcja X (n x 13) -> kupony (n,).
"""
import math
import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")))   # pricing_v2.py
from pricing_v2 import state_cost, PEN, LIM
 
BP = 1e-4
EXO = ["Exotic Risk 1-Vanna", "Exotic Risk 2-Volga", "Exotic Risk 3-Gap risk"]
 
# "Prawda" o konkurencji w symulatorze (zmieniaj, żeby sprawdzać odporność)
TRUTH = dict(mu_b=8 * BP, sd_b=7 * BP, p_out=0.10, out_spread=195 * BP,
             mu_s=3 * BP, sd_s=7 * BP, mu_any=-6.5 * BP, sd_any=3 * BP,
             f1=0.55, f2=0.25, f_rest=0.06)
_erf = np.vectorize(math.erf)
 
 
def load_rfqs(quote_history_csv):
    """Kolumny 5..12 pliku z kwotowaniami: runda, nominał, Mid, 5 Greków RFQ."""
    return pd.read_csv(quote_history_csv).iloc[:, 5:13].to_numpy(float)
 
 
def privates_from_game(game_history_csv):
    """Prywatne pozycje (12 x 5) z wierszy R.. historii gry."""
    g = pd.read_csv(game_history_csv)
    r = g[g["RFQ ID"].str.startswith("R")]
    return r[["Delta", "Vega"] + EXO].to_numpy(float)
 
 
def synthetic_privates(rng, amp):
    """Delta/vega: okrąg o okresie 12 rund. Egzotyka: okres 8.6 rundy, fazy co 60 st."""
    t = np.arange(1, 13)
    psi = np.radians(rng.uniform(0, 360) - 30 * (t - 1))
    dv = np.c_[-15 * np.cos(psi), 3 * np.sin(psi)]
    phi = rng.uniform(0, 2 * np.pi)
    exo = np.array([amp * np.cos(2 * np.pi * t / 8.6 + phi + k * np.pi / 3) for k in range(3)]).T
    exo[0] += rng.normal(0, 1.5, 3)                     # szok w rundzie 1
    return np.c_[dv, exo]
 
 
def play(strategy, rfqs, priv, rng, truth=TRUTH):
    """Rozgrywa 12 rund. Zwraca słownik z P&L, odchyleniem P&L i karami."""
    cum_exo = np.zeros(3)
    pnl, pen_total = [], 0.0
    for t in range(1, 13):
        rf = rfqs[rfqs[:, 0] == t]
        n = len(rf)
        book = np.r_[priv[t - 1, :2], cum_exo + priv[t - 1, 2:]]
        X = np.c_[np.tile(book, (n, 1)), rf]
        quote = strategy(X)
        spread = quote - X[:, 7]
        b = np.where(rng.random(n) < truth["p_out"], truth["out_spread"],
                     rng.normal(truth["mu_b"], truth["sd_b"], n))
        s = np.minimum(rng.normal(truth["mu_s"], truth["sd_s"], n), b)
        p_any = 0.5 * (1 + _erf((spread - truth["mu_any"]) / truth["sd_any"] / math.sqrt(2)))
        fill = np.where(spread > b, truth["f1"],
               np.where(spread > s, truth["f2"],
               np.where(rng.random(n) < p_any, truth["f_rest"], 0.0)))
        trades = fill[:, None] * X[:, 8:13]
        dv = priv[t - 1, :2] + trades[:, :2].sum(0)
        cum_exo = cum_exo + priv[t - 1, 2:] + trades[:, 2:].sum(0)
        S = np.r_[dv, cum_exo][None, :]
        edge = 0.1 + (fill * X[:, 6] * (X[:, 7] - quote)).sum()   # 0.1 = prywatna pozycja
        pnl.append(edge - state_cost(S)[0])
        pen_total += (PEN * np.maximum(np.abs(S) - LIM, 0)).sum()
    pnl = np.array(pnl)
    return dict(pnl=pnl.sum(), sd=pnl.std(), penalties=pen_total)
 
 
if __name__ == "__main__":
    import pricing_v2
    QUOTES = "data/game_1/trader_quote_history.csv"      # dowolny plik z kwotowaniami (RFQ są te same)
    GAMES = {"gra 1": "data/game_1/trader_game_history.csv"}  # dopisz ścieżkę do gry 2
    rfqs = load_rfqs(QUOTES)
    strategies = {"v2": pricing_v2.optimal_quote, "nie handluj": lambda X: X[:, 7] - 0.0199}
    for sname, strat in strategies.items():
        for gname, path in GAMES.items():
            r = [play(strat, rfqs, privates_from_game(path), np.random.default_rng(s)) for s in range(20)]
            print(f"{sname:12s} {gname}: P&L {np.mean([x['pnl'] for x in r]):7.2f}  "
                  f"SD rundy {np.mean([x['sd'] for x in r]):.2f}  kary {np.mean([x['penalties'] for x in r]):.2f}")
        for amp in [0.3, 3.5]:
            r = [play(strat, rfqs, synthetic_privates(np.random.default_rng(100 + s), amp), np.random.default_rng(s)) for s in range(20)]
            print(f"{sname:12s} syntetyczna, amplituda {amp}: P&L {np.mean([x['pnl'] for x in r]):7.2f}")
