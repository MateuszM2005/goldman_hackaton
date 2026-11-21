"""
Silnik kwotujący v2 (numpy).
 
Używa wyłącznie operacji dostępnych w ONNX: abs, maximum/minimum (Relu),
+ - * /, mnożenie macierzy i argmax. Wszystko, co dotyczy konkurencji,
jest stałą (wektory policzone raz, poza silnikiem).
 
Struktura:
  state_cost     prawdziwy koszt stanu księgi (zwalidowany na grach 1-2)
  delta_cost     zmiana kosztu przy wypełnieniu f, z korektami wyceny
  quote_tables   stałe z modelu konkurencji (siatka kuponów, wagi scenariuszy)
  optimal_quote  wybór kuponu maksymalizującego oczekiwany zysk
"""
import numpy as np
 
BP = 1e-4
 
# ---- Regulamin (kolejność: delta, vega, vanna, volga, gap) ----
LIM = np.array([12.0, 3.0, 3.0, 3.0, 3.0])          # limity
HOLD = np.array([0.006, 0.025, 0.006, 0.006, 0.006])  # koszt utrzymania za rundę
PEN = np.array([0.05, 0.05, 1.00, 1.00, 1.00])       # kara: ułamek nadwyżki za rundę
IS_EXO = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
 
# ---- Parametry strategii (dobrane w symulatorze) ----
H_PEN = 1.0      # przez ile rund liczymy karę za nadwyżkę (1 = tylko bieżąca runda)
K_SHARE = 8.0    # na ile wygranych RFQ rozkłada się jedna nadwyżka ponad limit
                 # (12 najlepsze przy łagodnej konkurencji, 6 przy ostrej; 8 = kompromis)
FILLS = (0.55, 0.25, 0.06)           # wypełnienie za 1. miejsce, 2. miejsce, dalsze
GRID = np.arange(-50, 31) * BP       # kandydaci: Mid - 50 pb ... Mid + 30 pb (limit premii)
 
# ---- Konkurencja (spread = kupon - Mid), pula gier 1 i 2 ----
COMP = dict(mu_best=8 * BP, sd_best=7 * BP, p_out=0.10,
            mu_second=3 * BP, sd_second=7 * BP,
            mu_any=-6.5 * BP, sd_any=3 * BP)
 
 
def state_cost(S, h_hold=1.0, h_pen=1.0):
    """
    Koszt stanu księgi S (n x 5) w mln USD.
    Przy h_hold = h_pen = 1 to dokładnie koszt jednej rundy u organizatorów:
    hedging delty/vegi, kapitał egzotyki i kary (5% / 100% nadwyżki).
    h_hold, h_pen: mnożniki horyzontu (skalar albo n x 5).
    """
    hold = (h_hold * HOLD * np.abs(S)).sum(axis=1)
    pen = (h_pen * PEN * np.maximum(np.abs(S) - LIM, 0.0)).sum(axis=1)
    return hold + pen
 
 
def horizons(t):
    """
    Mnożniki horyzontu (n x 5) dla rundy t:
      utrzymanie: 1 dla delty/vegi (reset co rundę), 13 - t dla egzotyki
      kara:       1 dla delty/vegi, min(13 - t, H_PEN) dla egzotyki
    """
    t = t[:, None]
    h_hold = 1.0 + IS_EXO * (12.0 - t)
    h_pen = 1.0 + IS_EXO * (np.minimum(13.0 - t, H_PEN) - 1.0)
    return h_hold, h_pen
 
 
def share_excess(B):
    """
    Księga, w której nadwyżka ponad limit jest podzielona przez K_SHARE.
    Jedno RFQ może 'zarobić' tylko na swojej części nadwyżki, bo tę samą
    nadwyżkę zmniejsza wiele wygranych RFQ naraz (lekcja z gry 2).
    Dodawanie ryzyka ponad limit jest nadal wyceniane w pełni.
    """
    excess = np.maximum(B - LIM, 0.0) - np.maximum(-B - LIM, 0.0)
    return B - (1.0 - 1.0 / K_SHARE) * excess
 
 
def delta_cost(X, f):
    """
    Zmiana kosztu księgi (mln USD), jeśli dostaniemy ułamek f danego RFQ.
    X: (n, 13) wejście silnika. Zwraca (n,).
    """
    B, g, t = X[:, 0:5], X[:, 8:13], X[:, 5]
    h_hold, h_pen = horizons(t)
    hold = (h_hold * HOLD * (np.abs(B + f * g) - np.abs(B))).sum(axis=1)
    Be = share_excess(B)
    pen = (h_pen * PEN * (np.maximum(np.abs(Be + f * g) - LIM, 0.0)
                          - np.maximum(np.abs(Be) - LIM, 0.0))).sum(axis=1)
    return hold + pen
 
 
def _norm_cdf(z):
    from math import erf, sqrt
    return 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))
 
 
def quote_tables(comp=COMP, grid=GRID, fills=FILLS):
    """
    Stałe niezależne od RFQ (liczone raz, poza silnikiem):
      P: (3, k) szansa na 1. miejsce, 2. miejsce i dalsze dla każdego kandydata
      A: (k,)   oczekiwana marża na jednostkę nominału: -c * sum_j P_j f_j
    """
    p1 = (1 - comp["p_out"]) * _norm_cdf((grid - comp["mu_best"]) / comp["sd_best"])
    top2 = _norm_cdf((grid - comp["mu_second"]) / comp["sd_second"])
    p2 = np.maximum(top2 - p1, 0.0)
    p_any = np.maximum(_norm_cdf((grid - comp["mu_any"]) / comp["sd_any"]), top2)
    pr = np.maximum(p_any - p1 - p2, 0.0)
    P = np.vstack([p1, p2, pr])
    A = -grid * (np.array(fills)[:, None] * P).sum(axis=0)
    return P, A
 
 
P_TAB, A_TAB = quote_tables()
 
 
def optimal_quote(X):
    """
    Kupon maksymalizujący oczekiwany zysk:
      E[zysk](c) = sum_j P_j(c) * [ f_j * N * (Mid - c) - delta_cost(f_j) ]
                 = N * A(c) - sum_j P_j(c) * delta_cost(f_j)
    """
    dC = np.stack([delta_cost(X, f) for f in FILLS], axis=1)     # (n, 3)
    ev = X[:, 6:7] * A_TAB[None, :] - dC @ P_TAB                # (n, k)
    return X[:, 7] + GRID[np.argmax(ev, axis=1)]
 
Claude is responding
