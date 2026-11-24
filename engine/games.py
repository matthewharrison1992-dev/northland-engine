"""Games planner: how many capped, position-legal games a roster gets in a matchup week, what each player
adds to that total, and which add/drop moves (single swaps and two-add relays) buy the most extra games.

Games are the volume lever: every counting category scales with starts. This module only counts games;
whether a move is worth an add is decided afterwards with core.sim_week (this week v the real opponent)
and core.team_week_dist + core.allplay (the long-run cost of the player who leaves).

Schedules: list of day sets of NBA team abbreviations (core.TEAM_FIX applied), index 0 = first NZ day of the
matchup (Tuesday). health[name] = list of P(active) per day (0 for OUT days, ~0.95 ACTIVE, ~0.85 day-to-day).
"""
from __future__ import annotations
import json, itertools
import numpy as np
from .core import best_lineup, TEAM_FIX


def schedule_from_connector(path_or_dict, first_nz_date=None):
    """nba_schedule() output -> list of 7 sets of teams by NZ date (the 'tip_nz' field), plus the NZ day labels."""
    d = json.load(open(path_or_dict)) if isinstance(path_or_dict, str) else path_or_dict
    by = {}
    for g in d["games"]:
        day = " ".join(g["tip_nz"].split()[:3])  # 'Tue 01 Dec'
        by.setdefault((g["us_date"], day), set()).update(TEAM_FIX.get(t, t) for t in (g["away"], g["home"]))
    keys = sorted(by)  # US date order = NZ day order
    return [by[k] for k in keys], [k[1] for k in keys]


def expected_starts(roster, days, health, val, n=300, seed=5, day_from=0, active=None):
    """Monte Carlo of daily availability -> expected starts per player and per day (capped at the 10 legal slots).
    active[name] = set of day indexes the player is on the roster (default: every day)."""
    rng = np.random.default_rng(seed)
    pos = {p.name: p.pos for p in roster}
    per_player = {p.name: 0.0 for p in roster}
    per_day = []
    cache = {}
    for d in range(day_from, len(days)):
        cand = [p for p in roster if p.nba in days[d] and health[p.name][d] > 0
                and (active is None or d in active.get(p.name, range(len(days))))]
        if not cand:
            per_day.append(0.0); continue
        av = rng.random((n, len(cand))) < np.array([health[p.name][d] for p in cand])
        tot = 0
        for s in range(n):
            key = tuple(p.name for j, p in enumerate(cand) if av[s, j])
            if key not in cache: cache[key] = best_lineup(list(key), pos, val)[0]
            ch = cache[key]
            tot += len(ch)
            for nm in ch: per_player[nm] += 1
        per_day.append(tot / n)
    return {k: v / n for k, v in per_player.items()}, per_day


def day_table(roster, days, health, val, labels, day_from=0):
    """per day: players with games, expected usable starts, open slots (10 - usable)."""
    _, per_day = expected_starts(roster, days, health, val, day_from=day_from)
    rows = []
    for i, d in enumerate(range(day_from, len(days))):
        names = [p.name for p in roster if p.nba in days[d] and health[p.name][d] > 0]
        rows.append({"day": labels[d], "playing": len(names), "usable": round(per_day[i], 1),
                     "open": round(10 - per_day[i], 1), "names": names})
    return rows


def marginal_games(roster, days, health, val, day_from=0):
    """games the team loses if each player is removed (0 = he adds nothing this week: natural drop)."""
    _, base = expected_starts(roster, days, health, val, day_from=day_from)
    tot = sum(base); out = {}
    for p in roster:
        _, pd = expected_starts([q for q in roster if q is not p], days, health, val, day_from=day_from)
        out[p.name] = tot - sum(pd)
    return out


def _total(roster, days, health, val, day_from, active, n=300):
    _, pd = expected_starts(roster, days, health, val, n=n, day_from=day_from, active=active)
    return sum(pd)


def best_swaps(roster, fas, drops, days, health, val, day_from=0, top=10, n=300):
    """single add/drop: drop D after his last game before switch day s, add A from day s.
    Returns rows sorted by games gained: (gain, add, drop, switch_day)."""
    n = len(days)
    base = _total(roster, days, health, val, day_from, None, n)
    rows = []
    for a in fas:
        for dn in drops:
            best = None
            for s in range(day_from, n):
                act = {dn: set(range(0, s)), a.name: set(range(s, n))}
                g = _total(roster + [a], days, health, val, day_from, act, n) - base
                if best is None or g > best[0] + 1e-9: best = (g, s)
            rows.append({"gain": round(best[0], 2), "add": a.name, "add_team": a.nba, "drop": dn, "switch": best[1]})
    rows.sort(key=lambda r: -r["gain"])
    return rows[:top], base


def best_relays(roster, fas, drops, days, health, val, day_from=0, top=5, shortlist=8, n=80):
    """two adds through one roster spot: drop D, add A (days s1..s2-1), then drop A and add B (s2..end).
    Only the top `shortlist` free agents by single-swap gain are paired (speed)."""
    n = len(days)
    base = _total(roster, days, health, val, day_from, None, n)
    singles, _ = best_swaps(roster, fas, drops, days, health, val, day_from, top=len(fas) * len(drops), n=n)
    keep = []
    for r in singles:
        if r["add"] not in keep: keep.append(r["add"])
    fa = {f.name: f for f in fas}
    cands = [fa[k] for k in keep[:shortlist]]
    rows = []
    for dn in drops:
        for a, b in itertools.permutations(cands, 2):
            best = None
            for s1 in range(day_from, n - 1):
                for s2 in range(s1 + 1, n):
                    act = {dn: set(range(0, s1)), a.name: set(range(s1, s2)), b.name: set(range(s2, n))}
                    g = _total(roster + [a, b], days, health, val, day_from, act, n) - base
                    if best is None or g > best[0] + 1e-9: best = (g, s1, s2)
            rows.append({"gain": round(best[0], 2), "drop": dn, "first": a.name, "first_team": a.nba,
                         "second": b.name, "second_team": b.nba, "switch1": best[1], "switch2": best[2]})
    rows.sort(key=lambda r: -r["gain"])
    return rows[:top], base
