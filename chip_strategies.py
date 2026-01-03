#!/usr/bin/env python3
"""
PropSetzstrategien.py - All-in-One Strategy Optimizer for Two-Player Chip-Abräum Game


Ein All-in-One-Skript zur Analyse von Zwei-Spieler "Chip-Abräum"-Strategien.

Funktionen:
  - generate_strategies_with_limits: Erzeugt Strategien innerhalb von Limits
  - prepare_p_list: Optional exakte Bruchrechnung mit Fraction
  - P_A: Memoisierte Rekursion für P_A(S,T)
  - compute_outcomes: Liefert (P_A, P_B, P_U)
  - find_better_strategies: Vollsuche oder Hill-Climb
  - parse_limits: Hilfsfunktion für CLI
  - main: Kommandozeilen-Interface

Usage:
  python PropSetzstrategien.py [--mode MODE] [--chips N] [--min LIMITS] [--max LIMITS] [--seed SEED]

Parameter-Übergabe (funktionaler Stil) garantiert klare Signaturen und einfache Tests.
"""

import argparse
import itertools
import random
from functools import lru_cache
import sys
from typing import Tuple

try:
    from fractions import Fraction
except ImportError:
    Fraction = None

# ─────────────────────────────────────────────────────────────────────────────
# Einstellungen / Konstanten
# ─────────────────────────────────────────────────────────────────────────────
# Default-Wahrscheinlichkeitsverteilung p für 6-Felder-Beispiel (Differenzen beim Würfeln)
DEFAULT_P_LIST = [6/36, 10/36, 8/36, 6/36, 4/36, 2/36]
# Beispiel-Basis-Strategie
DEFAULT_BASE_STRAT = (3, 5, 4, 3, 2, 1)

# ─────────────────────────────────────────────────────────────────────────────
# Kernfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def generate_strategies_with_limits(total_chips, num_fields, min_limits, max_limits):
    """
    Erzeuge alle Strategien von `total_chips` über `num_fields` Felder,
    die in [min_limits[i], max_limits[i]] liegen.

    Returns:
        Iterator[tuple[int]]
    """
    for combo in itertools.combinations_with_replacement(range(num_fields), total_chips):
        strat = [0] * num_fields
        for idx in combo:
            strat[idx] += 1
        if all(min_limits[i] <= strat[i] <= max_limits[i] for i in range(num_fields)):
            yield tuple(strat)


def prepare_p_list(p_values, exact=False):
    """
    Konvertiere Liste von float-Wahrscheinlichkeiten in Fraction,
    falls exact=True und Fraction verfügbar.

    Returns:
        list[float] oder list[Fraction]
    """
    if exact and Fraction is not None:
        return [Fraction(x).limit_denominator() for x in p_values]
    return list(p_values)

# Module-level Wahrscheinlichkeitsliste (Standard: float)
p_list = prepare_p_list(DEFAULT_P_LIST, exact=False)


#================
from functools import lru_cache

# p_list muss schon existieren, z.B.:
# p_list = prepare_p_list(DEFAULT_P_LIST, exact=False)

@lru_cache(None)
def P_A(V, W):
    V, W = list(V), list(W)
    sum_V, sum_W = sum(V), sum(W)
    if   sum_V == 0 and sum_W == 0: return 0      # Unentschieden
    elif sum_V == 0:             return 1      # A gewinnt
    elif sum_W == 0:             return 0      # B gewinnt

    s = sum(p for p, v_i, w_i in zip(p_list, V, W) if v_i or w_i)

    prob = 0
    for i, p in enumerate(p_list):
        if V[i] or W[i]:
            v_new, w_new = V.copy(), W.copy()
            if v_new[i]: v_new[i] -= 1
            if w_new[i]: w_new[i] -= 1
            prob += (p / s) * P_A(tuple(v_new), tuple(w_new))
    return prob

def compute_outcomes(V, W):
    PA = P_A(tuple(V), tuple(W))
    PB = P_A(tuple(W), tuple(V))
    PU = 1 - PA - PB
    return PA, PB, PU

def baue_P_gewinn(p: Tuple[Fraction, ...]):
    """
    Baut eine exakte Gewinnwahrscheinlichkeitsfunktion für gegebene p-Werte als Fractions.

    Args:
        p (Tuple[Fraction,...]): Exakte Wahrscheinlichkeiten p_i.
    Returns:
        Callable[[Tuple[int,...], Tuple[int,...]], Tuple[Fraction, Fraction, Fraction]]
    """
    # Lokale exakte p_list anlegen
    exact_p = prepare_p_list(list(p), exact=True)
    def f(S, T):
        global p_list
        old_p = p_list
        p_list = exact_p
        PA, PB, PU = compute_outcomes(S, T)
        p_list = old_p
        return PA, PB, PU
    return f


def simulate_game(p_vals, S, T, N=100000, seed=None):
    """
    Monte-Carlo-Simulation des Spiels für N Durchläufe.

    Args:
        p_vals (list[float] or list[Fraction]): Treffer-Wahrscheinlichkeiten p_i.
        S (tuple[int]): Strategie von A.
        T (tuple[int]): Strategie von B.
        N (int): Anzahl Simulationen (default 100000).
        seed (int): Zufalls-Seed (optional).

    Returns:
        dict: {
            'P_A': P_A_schätzung,
            'P_B': P_B_schätzung,
            'P_U': P_U_schätzung
        }
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    m = len(p_vals)
    counts = {'A':0, 'B':0, 'U':0}
    for _ in range(N):
        s = list(S)
        t = list(T)
        while True:
            i = rng.choice(m, p=p_vals)
            if s[i] > 0:
                s[i] -= 1
            if t[i] > 0:
                t[i] -= 1
            if sum(s)==0 and sum(t)==0:
                counts['U'] += 1
                break
            if sum(s)==0:
                counts['A'] += 1
                break
            if sum(t)==0:
                counts['B'] += 1
                break
    return {k: counts[k]/N for k in counts}


def simulate_with_wald(p_vals, S, T, N=100000, alpha=0.05, seed=None):
    """
    Monte-Carlo-Simulation mit Wald-Konfidenzintervallen.

    Args:
        p_vals (list[float] or list[Fraction]): Treffer-Wahrscheinlichkeiten p_i.
        S (tuple[int]): Strategie von A.
        T (tuple[int]): Strategie von B.
        N (int): Anzahl Simulationen.
        alpha (float): Signifikanzniveau (default 0.05 für 95% CI).
        seed (int): Zufalls-Seed.

    Returns:
        dict: {
            'P_A': (schätzung, (low,high)),
            'P_B': (schätzung, (low,high)),
            'P_U': (schätzung, (low,high))
        }
    """
    import numpy as np
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    m = len(p_vals)
    counts = {'A':0, 'B':0, 'U':0}
    for _ in range(N):
        s = list(S)
        t = list(T)
        while True:
            i = rng.choice(m, p=p_vals)
            if s[i] > 0: s[i] -= 1
            if t[i] > 0: t[i] -= 1
            if sum(s)==0 and sum(t)==0:
                counts['U'] += 1; break
            if sum(s)==0:
                counts['A'] += 1; break
            if sum(t)==0:
                counts['B'] += 1; break
    # Schätzwerte
    phat_A = counts['A']/N
    phat_B = counts['B']/N
    phat_U = counts['U']/N
    # Wald-CI
    z = norm.ppf(1 - alpha/2)
    def wald_ci(p):
        delta = z * (p*(1-p)/N)**0.5
        return (max(0, p - delta), min(1, p + delta))
    return {
        'P_A': (phat_A, wald_ci(phat_A)),
        'P_B': (phat_B, wald_ci(phat_B)),
        'P_U': (phat_U, wald_ci(phat_U)),
    }


def simuliere_duell_1_wuerfel(S, T, p_vals, N=100000, alpha=0.05, seed=None):
    """
    Alias für Ein-Würfel-Simulation mit Wald-CI.
    """
    return simulate_with_wald(p_vals, S, T, N, alpha, seed)


def simuliere_duell_2_wuerfel(S, T, p1_vals, p2_vals, N=100000, alpha=0.05, seed=None):
    """
    Zwei-Würfel-Simulation mit Wald-Konfidenzintervallen.
    Jede Runde: A wirft Index i ~ p1, B wirft Index j ~ p2 (unabhängig).
    Entfernen jeweils 1 Chip, falls am geworfenen Feld > 0 vorhanden.
    """
    import numpy as np
    try:
        from scipy.stats import norm
        z = norm.ppf(1 - alpha/2)
    except Exception:
        z = 1.959963984540054  # 95%-Fallback

    rng = np.random.default_rng(seed)
    p1 = np.asarray(p1_vals, dtype=float); p1 /= p1.sum()
    p2 = np.asarray(p2_vals, dtype=float); p2 /= p2.sum()
    m1, m2 = len(p1), len(p2)
    assert len(S) == m1 and len(T) == m2, "Strategielängen müssen zu p1/p2 passen"

    wins = {'A': 0, 'B': 0, 'U': 0}
    for _ in range(N):
        V = list(S); W = list(T)
        while True:
            i = rng.choice(m1, p=p1)  # A's Wurf
            j = rng.choice(m2, p=p2)  # B's Wurf
            if V[i] > 0: V[i] -= 1
            if W[j] > 0: W[j] -= 1
            sv, sw = sum(V), sum(W)
            if sv == 0 and sw == 0:
                wins['U'] += 1; break
            if sv == 0:
                wins['A'] += 1; break
            if sw == 0:
                wins['B'] += 1; break

    phA = wins['A']/N; phB = wins['B']/N; phU = wins['U']/N
    def ci(p):
        delta = z * (p*(1-p)/N) ** 0.5
        return (max(0.0, p - delta), min(1.0, p + delta))
    return {'P_A': (phA, ci(phA)), 'P_B': (phB, ci(phB)), 'P_U': (phU, ci(phU))}




def find_better_strategies(base_strategy, total_chips, min_limits, max_limits,
                           p_vals, exhaustive=True, seed=None):
    """
    Finde Strategien W, für die P_B(W, base) > P_A(base, W).

    Args:
        base_strategy (tuple[int]): zu schlagende Strategie
        total_chips (int): Gesamtzahl der Chips
        min_limits (tuple[int]): Minimale Chips pro Feld
        max_limits (tuple[int]): Maximale Chips pro Feld
        p_vals (list[float] or list[Fraction]): Wahrscheinlichkeiten p_i
        exhaustive (bool): True=Vollsuche, False=Hill-Climb
        seed (int): Seed für Zufall (bei Hill-Climb)

    Returns:
        list[tuple[tuple[int], float]]: (Strategie W, P_B(W,base)-P_A(base,W))
    """
    global p_list
    # Setze Wahrscheinlichkeitsliste im Modul
    p_list = prepare_p_list(p_vals, exact=isinstance(p_vals[0], Fraction))
    better = []
    num_fields = len(p_list)

    if exhaustive:
        # Vollständige Aufzählung
        for W in generate_strategies_with_limits(total_chips, num_fields, min_limits, max_limits):
            PA_base, PB_alt, _ = compute_outcomes(base_strategy, W)
            # Wenn B mit W gegen A besser ist
            if PB_alt > PA_base:
                better.append((W, PB_alt - PA_base))
    else:
        # Hill-Climbing zur Suche einer einzigen besseren Strategie
        random.seed(seed)
        current = base_strategy
        best_diff = 0
        for _ in range(1000):
            i, j = random.sample(range(num_fields), 2)
            cand = list(current)
            if cand[i] > min_limits[i] and cand[j] < max_limits[j]:
                cand[i] -= 1
                cand[j] += 1
                PA_base, PB_cand, _ = compute_outcomes(base_strategy, cand)
                # Differenz P_B - P_A
                diff = PB_cand - PA_base
                if diff > best_diff:
                    best_diff = diff
                    better = [(tuple(cand), diff)]
                    current = tuple(cand)
    # Sortiere nach Vorteil absteigend
    return sorted(better, key=lambda x: -x[1])


def P_fixiert(S, p_vals, xlabel="", ylabel="", title="", width=0.8, sort_by='PB',
                save_pdf: str=None, breite=8, hoehe=4, csv_path: str=None, plot: bool=True):
    """
    Visualisiert gestapelte Balkendiagramme für P_A(S,T) und P_B(S,T)
    für eine feste Strategie S über alle Gegner-Strategien T.

    Args:
        S (tuple[int]): Fixe Strategie von A.
        p_vals (list[float] or list[Fraction]): Wahrscheinlichkeiten p_i.
        xlabel, ylabel, title (str): Achsenbeschriftungen und Titel.
        width (float): Breite der Balken (Empfehlung: 0.6 bis 1.0).
        sort_by (str): 'PA' oder 'PB'; sortiert nach absteigenden Werten.
        save_pdf (str): Dateiname, um die Grafik als PDF zu speichern.
        csv_path (str): Dateiname, um Ergebnisse in CSV zu speichern.
        plot (bool): Wenn False, wird die Grafik nicht angezeigt.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Setup globale p_list
    global p_list
    p_list = prepare_p_list(p_vals, exact=isinstance(p_vals[0], Fraction))

    # Erstelle alle möglichen Gegner-Strategien T
    total = sum(S)
    num_fields = len(p_list)
    Ts = list(generate_strategies_with_limits(
        total, num_fields,
        min_limits=(0,)*num_fields,
        max_limits=(total,)*num_fields
    ))
    Ts = [T for T in Ts if T != S]

    # Berechnung
    results = []
    for T in Ts:
        pa, pb, _ = compute_outcomes(S, T)
        results.append((T, float(pa), float(pb)))

    # Sortieren
    key_idx = 1 if sort_by=='PA' else 2
    results.sort(key=lambda x: x[key_idx], reverse=True)

    Ts_sorted = [r[0] for r in results]
    PA_vals = np.array([r[1] for r in results])
    PB_vals = np.array([r[2] for r in results])

    # CSV export
    if csv_path:
        import csv
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['T_strategy','P_A','P_B'])
            for T, pa, pb in results:
                writer.writerow([T, pa, pb])

    x = np.arange(len(Ts_sorted))

    # Plot
    #fig, ax = plt.subplots(figsize=(max(8, len(Ts_sorted)*0.2), 4))
    fig, ax = plt.subplots(figsize=(breite,hoehe))
    ax.bar(x, PA_vals, width, label='P_A', color='gray', edgecolor='black')
    ax.bar(x, PB_vals, width, bottom=PA_vals, label='P_B', color='white', edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels([str(T) for T in Ts_sorted], rotation=45, ha='right')
    ax.set_xlabel(xlabel,fontsize=12)
    ax.set_ylabel(ylabel,fontsize=14)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    # PDF speichern
    if save_pdf:
        fig.savefig(save_pdf, format='pdf')

    # Plot anzeigen
    if plot:
        plt.show()
    else:
        plt.close(fig)


# parse_limits: CSV-String → tuple[int]
def parse_limits(arg):
    """Wandelt CSV-String in tuple[int] um (z.B. "2,4,3,2,1,0")."""
    return tuple(int(x) for x in arg.split(','))



from functools import lru_cache
from fractions import Fraction

def make_two_dice_recursion(p1_list, p2_list):
    @lru_cache(maxsize=None)
    def P_A2(V, W):
        S, T = list(V), list(W)
        # Absorptionsfälle
        if   sum(S)==0 and sum(T)==0: return Fraction(0)
        if   sum(S)==0:               return Fraction(1)
        if   sum(T)==0:               return Fraction(0)

        X, p_stay = Fraction(0), Fraction(0)
        m = len(p1_list)
        for i in range(m):
            for j in range(m):
                pij = p1_list[i] * p2_list[j]
                a_rem = S[i]>0
                b_rem = T[j]>0
                if not (a_rem or b_rem):
                    p_stay += pij
                else:
                    S2, T2 = S.copy(), T.copy()
                    if a_rem: S2[i] -= 1
                    if b_rem: T2[j] -= 1

                    key = (tuple(S2), tuple(T2))
                    # Absorption?
                    if   sum(S2)==0 and sum(T2)>0:      X += pij
                    elif sum(T2)==0 and sum(S2)>0:      pass
                    elif sum(S2)==0 and sum(T2)==0:     pass
                    else:                               X += pij * P_A2(key[0], key[1])

        return X / (1 - p_stay)
    return P_A2


from fractions import Fraction
def validate_p(p_list, tol=0):
    """
    Prüft, ob p_list eine Wahrscheinlichkeitsverteilung ist (Summe = 1).
    Bei Fractions: tol=0, bei floats kann tol>0 sein.
    """
    if any(pi < 0 for pi in p_list):
        raise ValueError(f"Wahrscheinlichkeiten müssen ≥0 sein, gefunden: {p_list}")
    total = sum(p_list)
    if isinstance(total, Fraction):
        if total != 1:
            raise ValueError(f"Summe der Wahrscheinlichkeiten muss 1 sein, ist aber {total}")
    else:
        if abs(total - 1) > tol:
            raise ValueError(f"Summe der Wahrscheinlichkeiten muss 1 sein, ist aber {total:.6f}")

            

def P_fixiert_2dice(
    S, p1_vals, p2_vals,
    N=100_000, alpha=0.05, seed=None,
    width=0.6, sort_by='PA',
    csv_path=None, save_pdf=None, plot=True
):
    """
    Visualisiert für eine feste Strategie S die Gewinn-Wahrscheinlichkeiten
    P_A und P_B im Zwei-Würfel-Fall (Simulation mit Wald-CI).

    Args:
        S (tuple[int]): Basis-Strategie von A.
        p1_vals, p2_vals (list[float] oder list[Fraction]): Würfel-Verteilungen.
        N (int): Anzahl der Simulationsläufe.
        alpha (float): Signifikanzniveau (z.B. 0.05 für 95 % CI).
        seed (int): Zufalls-Seed für Reproduzierbarkeit.
        width (float): Balkenbreite (0.2–1.0).
        sort_by (str): 'PA' oder 'PB' – Sortierreihenfolge.
        csv_path (str): Dateiname für CSV-Ausgabe.
        save_pdf (str): Dateiname für PDF-Export der Grafik.
        plot (bool): Grafik anzeigen oder nur speichern.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from chip_strategies import generate_strategies_with_limits, simuliere_duell_2_wuerfel

    # 1) Generiere alle Gegner-Strategien T ≠ S
    total = sum(S)
    m = len(S)
    Ts = list(generate_strategies_with_limits(total, m, (0,)*m, (total,)*m))
    Ts = [T for T in Ts if T != S]

    # 2) Simuliere P_A, P_B für jeden T
    results = []
    rng_seed = seed or 0
    for T in Ts:
        cis = simuliere_duell_2_wuerfel(S, T, p1_vals, p2_vals,
                                        N=N, alpha=alpha, seed=rng_seed)
        pa_hat = cis['P_A'][0]
        pb_hat = cis['P_B'][0]
        results.append((T, float(pa_hat), float(pb_hat)))

    # 3) Sortieren
    idx = 1 if sort_by=='PA' else 2
    results.sort(key=lambda x: x[idx], reverse=True)
    Ts_sorted = [r[0] for r in results]
    PA_vals = np.array([r[1] for r in results])
    PB_vals = np.array([r[2] for r in results])

    # 4) CSV-Export
    if csv_path:
        import csv
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['T_strategy','P_A','P_B'])
            for T, pa, pb in results:
                w.writerow([T, pa, pb])

    # 5) Plot
    x = np.arange(len(Ts_sorted))
    fig, ax = plt.subplots(figsize=(max(8, len(Ts_sorted)*0.3), 5))
    ax.bar(x, PA_vals, width, label='P_A', color='gray', edgecolor='black')
    ax.bar(x, PB_vals, width, bottom=PA_vals,
           label='P_B', color='white', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([str(T) for T in Ts_sorted],
                       rotation=45, ha='right')
    ax.set_xlabel('Strategie T')
    ax.set_ylabel('Gewinnwahrscheinlichkeit')
    ax.set_title(f'2-Würfel: P_A vs P_B für S={S}')
    ax.legend()
    plt.tight_layout()

    if save_pdf:
        fig.savefig(save_pdf, format='pdf')
    if plot:
        plt.show()
    else:
        plt.close(fig)

            

def main():
    parser = argparse.ArgumentParser(description="Chip-Abräum Strategie-Optimierer")
    parser.add_argument('--mode', choices=['exhaustive', 'hill'], default='exhaustive',
                        help='Suchmodus: Vollsuche oder Hill-Climbing')
    parser.add_argument('--chips', type=int, default=None,
                        help='Gesamtzahl Chips (Default: aus Basis-Strategie)')
    parser.add_argument('--min', dest='min_limits', type=parse_limits,
                        default=None, help='Min-Limits pro Feld (CSV)')
    parser.add_argument('--max', dest='max_limits', type=parse_limits,
                        default=None, help='Max-Limits pro Feld (CSV)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed für Hill-Climbing')
    parser.add_argument('--no-plot', dest='plot', action='store_false', default=True,
                        help='Deaktiviert die Grafik und zeigt nur CSV-Ausgabe mit head(10)')
    parser.add_argument('--csv', dest='csv_path', type=str, default=None,
                        help='Pfad zur CSV-Ausgabe (wird gespeichert, wenn --no-plot gesetzt ist)')
    args = parser.parse_args()

    base = DEFAULT_BASE_STRAT
    p_vals = DEFAULT_P_LIST
    total = args.chips or sum(base)
    min_limits = args.min_limits or tuple(0 for _ in base)
    max_limits = args.max_limits or tuple(total for _ in base)

    better = find_better_strategies(
        base_strategy=base,
        total_chips=total,
        min_limits=min_limits,
        max_limits=max_limits,
        p_vals=p_vals,
        exhaustive=(args.mode == 'exhaustive'),
        seed=args.seed
    )

    if not better:
        print("Keine bessere Strategie gefunden.")
        sys.exit(0)

    if args.plot:
        print("Gefundene bessere Strategien (Top 10):")
        for strat, diff in better[:10]:
            print(f"Strategie {strat} verbessert um {diff:.4f}")
    else:
        csv_out = args.csv_path or 'PA_PB_results.csv'
        P_fixiert(
            base, p_vals,
            xlabel="",
            ylabel="P_A + P_B",
            title=f"CSV-Ausgabe für S={base}",
            width=0.6,
            sort_by='PB',
            save_pdf=None,
            csv_path=csv_out,
            plot=False
        )
        try:
            import pandas as pd
            df = pd.read_csv(csv_out)
            print(df.head(10))
        except ImportError:
            print("Installiere pandas für eine Kopfzeilenausgabe, CSV gespeichert in", csv_out)
        sys.exit(0)

if __name__ == '__main__':
    main()
