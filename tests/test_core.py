import csv, os, numpy as np
from engine import core
from engine.core import Player, best_lineup, SLOT_OK, SLOTS

D = os.path.join(os.path.dirname(__file__), "..", "data")


def P(name, pos, nba="BOS"):
    s = dict(zip(core.FIELDS, [20, 30, 15, 6, 4, 1, .5, 2, 1.5, 6, 13, 2, 2.5]))
    return Player(name, nba, pos, "ACTIVE", "BE", "NTH", {"S": s, "L15": dict(s, GP=6), "L7": dict(s, GP=3)})


def test_lineup_is_position_legal():
    pos = {"a": ["PG"], "b": ["SG"], "c": ["SF"], "d": ["PF"], "e": ["C"], "f": ["C"], "g": ["C"],
           "h": ["PF"], "i": ["PG", "SG"], "j": ["SG", "SF"], "k": ["C"], "l": ["C"]}
    val = {n: 10 - i * .1 for i, n in enumerate(pos)}
    chosen, slots = best_lineup(list(pos), pos, val)
    assert len(chosen) == 10
    for s, n in slots.items():
        assert SLOT_OK[s.rstrip("0123456789")] & set(pos[n])


def test_pf_shortage_counts_9():
    # lesson 13 Nov: no PF-eligible player -> only 9 fillable slots? (PF slot can't be filled)
    pos = {f"c{i}": ["C"] for i in range(5)} | {f"g{i}": ["PG", "SG"] for i in range(4)} | {"w": ["SF"], "w2": ["SF"]}
    chosen, _ = best_lineup(list(pos), pos, {n: 1 for n in pos})
    assert len(chosen) == 9


def test_projection_shrinks_and_is_sane():
    pool = [P(f"g{i}", ["PG"]) for i in range(3)] + [P(f"w{i}", ["SF"]) for i in range(3)] + [P(f"b{i}", ["C"]) for i in range(3)]
    pri = core.pool_priors(pool)
    pr = core.project(pool[0], pri)
    assert 25 < pr["MIN"] < 35 and 0.3 < pr["FG%"] < 0.6 and pr["PTS"] > 5
    pr2 = core.project(pool[0], pri, prior=dict(zip(core.FIELDS, [70, 30, 15, 6, 4, 1, .5, 2, 1.5, 6, 13, 2, 2.5])))
    assert abs(pr2["PTS"] - pr["PTS"]) < 1e-6  # identical prior -> same projection


def test_stable_seed_deterministic():
    assert core._stable_seed("Cooper Flagg") == core._stable_seed("Cooper Flagg")


def test_data_files_consistent():
    fx = list(csv.DictReader(open(os.path.join(D, "fixtures.csv"))))
    weeks = {}
    for r in fx: weeks.setdefault(int(r["week"]), []).extend([r["home"], r["away"]])
    assert sorted(weeks) == list(range(1, 21))
    assert all(len(set(t)) == 10 for t in weeks.values())
    wt = list(csv.DictReader(open(os.path.join(D, "weekly_totals.csv"))))
    rows, target = core.league_week_matrix(wt)
    assert len(rows) >= 40 and len(rows) % 10 == 0  # 10 team rows per completed week
    # all-play: NTH week 3 was 0 all-play wins (lessons 10 Nov)
    assert core.actual_allplay(wt, 3, "NTH") == 0


def test_season_sim_runs():
    wt = list(csv.DictReader(open(os.path.join(D, "weekly_totals.csv"))))
    rows, _ = core.league_week_matrix(wt)
    mean, resid, _ = core.team_strength(rows)
    fx = list(csv.DictReader(open(os.path.join(D, "fixtures.csv"))))
    st = {"CAN": [3, 1, 0], "AKL": [3, 1, 0], "MAR": [3, 1, 0], "SOU": [3, 1, 0], "MAN": [2, 2, 0], "WLG": [2, 2, 0],
          "TAS": [2, 2, 0], "NTH": [1, 3, 0], "HWK": [1, 3, 0], "OTG": [0, 4, 0]}
    res = core.season_sim(fx, st, mean, resid, 5, n=2000)
    assert abs(sum(r["playoffs"] for r in res.values()) - 4) < 1e-6
    assert abs(sum(r["title"] for r in res.values()) - 1) < 1e-6


def test_fill_in_swaps_for_ruled_out_starter():
    # planned sits still start if a starter is ruled out before lock; a hard force_bench does not
    team = [P(f"t{i}", ["PG", "SG", "SF", "PF", "C"]) for i in range(11)]
    opp = [P("o0", ["C"])]
    pool = team + opp
    pri = core.pool_priors(pool + [P("g", ["PG"]), P("w", ["SF"])])
    projs = {p.name: core.project(p, pri) for p in pool}
    pp = {p.name: [1.0] for p in pool}
    pp["t0"] = [0.0]  # starter ruled out
    live = {"A": {}, "B": {}}
    kw = dict(n=400, seed=5, scratch=0.0)
    soft = core.sim_week(team, opp, projs, pri, [{"BOS"}], pp, live, fill_in={0: {"t10"}}, **kw)
    hard = core.sim_week(team, opp, projs, pri, [{"BOS"}], pp, live, force_bench={0: {"t10"}}, **kw)
    assert soft["proj"]["A"]["PTS"] > hard["proj"]["A"]["PTS"] * 1.05  # 10 starters v 9
    pp["t0"] = [1.0]  # everyone healthy: the sit stays on the bench either way
    soft = core.sim_week(team, opp, projs, pri, [{"BOS"}], pp, live, fill_in={0: {"t10"}}, **kw)
    hard = core.sim_week(team, opp, projs, pri, [{"BOS"}], pp, live, force_bench={0: {"t10"}}, **kw)
    assert abs(soft["proj"]["A"]["PTS"] / hard["proj"]["A"]["PTS"] - 1) < 0.02  # same 10 (draws differ slightly)
