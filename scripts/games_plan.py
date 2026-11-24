"""Games plan for one matchup week: python -m scripts.games_plan SNAPSHOT SCHEDULE.json [options]

SNAPSHOT: core.load_players format; owner column = NTH, FA, or the opponent's abbreviation.
SCHEDULE: nba_schedule() output saved as JSON (the whole matchup week, Tue..Mon NZ).
Options:  --from K      first NZ day still to play (0 = Tuesday; Monday preview of next week = 0)
          --opp ABBR    opponent owner tag in the snapshot (adds the win-% check with core.sim_week)
          --drops a,b   droppable 1-year players (default: every NTH player not in league.json multi_year)
          --health F    JSON {name: [p_day0..p_day6]} overrides (OUT until a return date, questionable, etc.)
          --live F      JSON {"A": raw totals, "B": raw totals} so far this week (default zeros)
          --longrun     add the paired rest-of-season all-play cost of each top move (slower)
Prints: day-by-day usable games both sides, each NTH player's marginal games, best single swaps and two-add
relays by games gained, then win % before/after for the best few (paired seeds) and the long-run cost."""
import sys, json, argparse
import numpy as np
from engine import core, games

ap = argparse.ArgumentParser()
ap.add_argument("snap"); ap.add_argument("sched")
ap.add_argument("--from", dest="day_from", type=int, default=0)
ap.add_argument("--opp"); ap.add_argument("--drops"); ap.add_argument("--health"); ap.add_argument("--live")
ap.add_argument("--longrun", action="store_true"); ap.add_argument("--sims", type=int, default=8000)
a = ap.parse_args()

P = core.load_players(a.snap)
pri = core.pool_priors(P)
proj = {p.name: core.project(p, pri) for p in P}
val = {p.name: core.value_score(proj[p.name], pri) for p in P}
days, labels = games.schedule_from_connector(a.sched)
base_h = {"ACTIVE": .95, "DAY_TO_DAY": .85, "OUT": 0.0}
health = {p.name: [base_h.get(p.injury, .95)] * len(days) for p in P}
if a.health: health.update(json.load(open(a.health)))
lg = json.load(open("data/league.json"))
multi = set(lg["northland"]["multi_year"])
NTH = [p for p in P if p.owner == "NTH" and p.slot != "IR"]
FA = [p for p in P if p.owner == "FA" and health[p.name][-1] > 0]
OPP = [p for p in P if a.opp and p.owner == a.opp and p.slot != "IR"]
drops = a.drops.split(",") if a.drops else [p.name for p in NTH if p.name not in multi]

print(f"Days: {' | '.join(labels)}   (from {labels[a.day_from]})")
for side, ro in (("NTH", NTH), (a.opp, OPP)):
    if not ro: continue
    t = games.day_table(ro, days, health, val, labels, a.day_from)
    print(f"{side:4s} usable " + "  ".join(f"{r['day'][:3]} {r['usable']:.1f}/{r['playing']}" for r in t)
          + f"   total {sum(r['usable'] for r in t):.1f}")
mg = games.marginal_games(NTH, days, health, val, a.day_from)
print("NTH marginal games (what each player adds after the 10-cap): " +
      ", ".join(f"{k.split()[-1]} {v:.1f}" for k, v in sorted(mg.items(), key=lambda x: x[1])))
sw, base = games.best_swaps(NTH, FA, drops, days, health, val, a.day_from, top=12)
print(f"\nBest single swaps (base {base:.1f} games):")
for r in sw:
    print(f"  +{r['gain']:.1f}  add {r['add']} ({r['add_team']}) from {labels[r['switch']]}, drop {r['drop']}"
          f"  | per-game value add {val[r['add']]:.1f} v drop {val[r['drop']]:.1f}")
rl, _ = games.best_relays(NTH, FA, drops, days, health, val, a.day_from)
print("Best two-add relays (one roster spot):")
for r in rl:
    print(f"  +{r['gain']:.1f}  drop {r['drop']} -> {r['first']} ({r['first_team']}) {labels[r['switch1']]} -> "
          f"{r['second']} ({r['second_team']}) {labels[r['switch2']]}")

if OPP:
    live = json.load(open(a.live)) if a.live else {"A": {}, "B": {}}
    rem = [days[d] if d >= a.day_from else set() for d in range(len(days))]
    def pw(team, hp):
        r = [core.sim_week(team, OPP, proj, pri, rem, hp, live, n=a.sims, seed=s) for s in (1, 2, 3)]
        return np.mean([x["p_win"] for x in r]), {c: np.mean([x["cat_win"][c] for x in r]) for c in core.CATS}
    b, bc = pw(NTH, health)
    print(f"\nWin % v {a.opp}: no move {b:.1%}")
    tests = [("swap", r) for r in sw[:4]] + [("relay", r) for r in rl[:2]]
    for kind, r in tests:
        hp = dict(health); extra = []
        if kind == "swap":
            s = r["switch"]; hp[r["drop"]] = [x if d < s else 0 for d, x in enumerate(health[r["drop"]])]
            hp[r["add"]] = [x if d >= s else 0 for d, x in enumerate(health[r["add"]])]; extra = [r["add"]]
            lab = f"+{r['add']} -{r['drop']} ({labels[s][:3]})"
        else:
            s1, s2 = r["switch1"], r["switch2"]
            hp[r["drop"]] = [x if d < s1 else 0 for d, x in enumerate(health[r["drop"]])]
            hp[r["first"]] = [x if s1 <= d < s2 else 0 for d, x in enumerate(health[r["first"]])]
            hp[r["second"]] = [x if d >= s2 else 0 for d, x in enumerate(health[r["second"]])]
            extra = [r["first"], r["second"]]
            lab = f"relay -{r['drop']} > {r['first']} > {r['second']}"
        by = {p.name: p for p in P}
        w, cw = pw(NTH + [by[x] for x in extra], hp)
        moved = ", ".join(f"{c} {cw[c]-bc[c]:+.0%}" for c in core.CATS if abs(cw[c] - bc[c]) >= .03)
        print(f"  {lab:60s} {w:.1%} ({(w-b)*100:+.1f} pts)  {moved}")

if a.longrun:
    rows, _ = core.league_week_matrix(core.load_csv("data/weekly_totals.csv"))
    hlr = {p.name: (.92 if p.injury == "ACTIVE" else .85) for p in P}
    def ap_(ro):
        dist, g = core.team_week_dist(ro, proj, pri, hlr, n_weeks=300)
        return core.allplay(dist, rows)[0] * 9, g
    b9, bg = ap_(NTH)
    print(f"\nLong run (typical week, all-play wins of 9): current {b9:.2f} ({bg:.1f} games)")
    by = {p.name: p for p in P}; seen = set()
    for r in sw[:6]:
        key = (r["add"], r["drop"])
        if key in seen: continue
        seen.add(key)
        ro = [p for p in NTH if p.name != r["drop"]] + [by[r["add"]]]
        v, g = ap_(ro)
        print(f"  keep {r['add']} instead of {r['drop']}: {v - b9:+.2f} all-play wins/week")
