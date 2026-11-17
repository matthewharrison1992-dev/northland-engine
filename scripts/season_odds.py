"""Playoff odds from data/: python -m scripts.season_odds [from_week] [--override NTH=0.04]"""
import sys, json, csv
from engine import core

args = [a for a in sys.argv[1:] if not a.startswith("--")]
ov = {}
for a in sys.argv[1:]:
    if a.startswith("--override"):
        t, p = a.split(" ", 1)[-1].split("=") if " " in a else sys.argv[sys.argv.index(a) + 1].split("=")
        ov[t] = float(p)
lg = json.load(open("data/league.json"))
wk = int(args[0]) if args else lg["current_matchup_period"]
rows, _ = core.league_week_matrix(core.load_csv("data/weekly_totals.csv"))
mean, resid, _ = core.team_strength(rows)
fx = core.load_csv("data/fixtures.csv")
res = core.season_sim(fx, lg["standings"], mean, resid, wk, n=20000,
                      week_override={(wk, t): p for t, p in ov.items()} or None)
for t, r in sorted(res.items(), key=lambda x: -x[1]["playoffs"]):
    print(f"{t}  playoffs {r['playoffs']:.0%}  title {r['title']:.0%}  exp wins {r['exp_wins']:.1f}")
