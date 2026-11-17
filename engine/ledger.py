"""Forecast + decision ledgers: append rows, score forecasts (Brier), reliability table."""
from __future__ import annotations
import csv, os
import numpy as np
from .core import CATS

FORECAST_COLS = ["logged_nz", "week", "team", "opp", "p_win"] + [f"p_{c}" for c in CATS] + \
                ["games_us", "games_opp", "outcome", "cat_outcomes", "brier_match", "brier_cats"]


def append_forecast(path, row):
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, FORECAST_COLS)
        if new: w.writeheader()
        w.writerow({k: row.get(k, "") for k in FORECAST_COLS})


def brier_match(p, outcome):
    """outcome: 1 win, 0 loss, 0.5 tie."""
    return (p - outcome) ** 2


def brier_cats(ps, outcomes):
    """mean Brier over the 9 categories (outcomes 1/0/0.5)."""
    return float(np.mean([(p - o) ** 2 for p, o in zip(ps, outcomes)]))


def score_week(path, week, outcome, cat_outcomes):
    """fill outcome + Brier columns for the forecast row(s) of a week; returns the scored rows."""
    rows = list(csv.DictReader(open(path)))
    done = []
    for r in rows:
        if int(r["week"]) == week and not r["outcome"]:
            r["outcome"] = outcome
            r["cat_outcomes"] = " ".join(str(x) for x in cat_outcomes)
            r["brier_match"] = round(brier_match(float(r["p_win"]), outcome), 4)
            if all(r[f"p_{c}"] != "" for c in CATS):
                r["brier_cats"] = round(brier_cats([float(r[f"p_{c}"]) for c in CATS], cat_outcomes), 4)
            done.append(r)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, FORECAST_COLS); w.writeheader(); w.writerows(rows)
    return done


def reliability(path, bins=(0, .2, .4, .6, .8, 1.0001)):
    """calibration table over all scored match forecasts: bin -> (n, mean forecast, observed rate)."""
    rows = [r for r in csv.DictReader(open(path)) if r["outcome"] != ""]
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        b = [r for r in rows if lo <= float(r["p_win"]) < hi]
        if b:
            out.append((f"{lo:.1f}-{min(hi,1):.1f}", len(b), float(np.mean([float(r["p_win"]) for r in b])),
                        float(np.mean([float(r["outcome"]) for r in b]))))
    return out
