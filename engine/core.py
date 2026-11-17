"""Northland Kauri engine v2 — core models.

Layers (each pure-function, data in → numbers out; no hidden state):
  load_*        : parse snapshot / league / fixtures files (source data, never edited by models)
  project_*     : per-game player projections (competing models, selected by backtest)
  best_lineup   : position-legal daily lineup (10 slots) by max matching
  sim_week      : Monte Carlo of the live matchup from current totals, real NBA schedule, availability probs
  team_week_dist: rest-of-season "typical week" distribution of a roster (capped, injury-weighted)
  allplay       : roster value = P(beat a random league team-week) per category and overall
  season_sim    : remaining regular season + playoffs from team-strength model (bootstrap residual vectors)
"""
from __future__ import annotations
import json, csv, math, re
from dataclasses import dataclass, field
import numpy as np

CATS = ["FG", "FT", "3PM", "REB", "AST", "STL", "BLK", "TO", "PTS"]
COUNT = ["PTS", "REB", "AST", "STL", "BLK", "TO", "3PM", "FGA", "FTA"]
FIELDS = ["GP", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TO", "3PM", "FGM", "FGA", "FTM", "FTA"]
SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UT", "UT", "UT"]
SLOT_OK = {"PG": {"PG"}, "SG": {"SG"}, "SF": {"SF"}, "PF": {"PF"}, "C": {"C"},
           "G": {"PG", "SG"}, "F": {"SF", "PF"}, "UT": {"PG", "SG", "SF", "PF", "C"}}
TEAM_FIX = {"PHL": "PHI", "PHO": "PHX"}  # ESPN abbreviations -> schedule abbreviations


# ---------------------------------------------------------------- loading
@dataclass
class Player:
    name: str; nba: str; pos: list; injury: str; slot: str; owner: str
    splits: dict = field(default_factory=dict)  # 'S','L7','L15' -> dict of FIELDS

    @property
    def arch(self):
        if "C" in self.pos: return "big"
        if self.pos[0] == "PG": return "guard"
        return "wing"


def _split(txt):
    if not txt.strip(): return None
    v = [float(x) for x in txt.split(",")]
    return dict(zip(FIELDS, v))


def load_players(path):
    """snapshot format, one player per line:
    name|nba|pos1,pos2|injury|lineup_slot|owner|S fields|L7 fields|L15 fields   (fields = FIELDS, comma-separated)"""
    out = []
    for line in open(path):
        if line.startswith("#") or "|" not in line: continue
        p = line.rstrip("\n").split("|")
        pl = Player(p[0], TEAM_FIX.get(p[1], p[1]), p[2].split(","), p[3], p[4], p[5])
        for key, txt in zip(["S", "L7", "L15"], p[6:9] + [""] * (9 - len(p))):
            s = _split(txt)
            if s: pl.splits[key] = s
        out.append(pl)
    return out


def load_league(path): return json.load(open(path))


def load_csv(path): return list(csv.DictReader(open(path)))


# ---------------------------------------------------------------- projections
def _sub(a, b):
    """aggregate split a minus subset split b -> per-game split for the remaining games (None if empty)."""
    n = a["GP"] - b["GP"]
    if n < 0.5: return None
    r = {"GP": n}
    for f in FIELDS[1:]:
        r[f] = max(0.0, (a[f] * a["GP"] - b[f] * b["GP"]) / n)
    return r


def pool_priors(players):
    """minutes-weighted per-minute rates and shooting by archetype, from the pool itself."""
    pri = {}
    for arch in ("guard", "wing", "big"):
        g = [p.splits["S"] for p in players if p.arch == arch and "S" in p.splits and p.splits["S"]["MIN"] > 10]
        mins = sum(s["MIN"] * s["GP"] for s in g)
        pr = {f: sum(s[f] * s["GP"] for s in g) / mins for f in COUNT}
        fga = sum(s["FGA"] * s["GP"] for s in g); fta = sum(s["FTA"] * s["GP"] for s in g)
        pr["FG%"] = sum(s["FGM"] * s["GP"] for s in g) / fga
        pr["FT%"] = sum(s["FTM"] * s["GP"] for s in g) / fta
        pri[arch] = pr
    return pri


# shrinkage strengths (minutes for rates, attempts for %), tuned by backtest()
K_MIN = {"PTS": 120, "REB": 150, "AST": 150, "STL": 400, "BLK": 300, "TO": 250, "3PM": 250, "FGA": 100, "FTA": 200}
K_FG, K_FT = 60, 30
PRIOR_SEASON_MIN = 300  # weight (minutes) of last season's per-minute rates when prior_season() data is supplied


def project(p, pri, model="D", recency=0.5, min_recent=0.6, splits=None, prior=None):
    """Per-game projection dict (counting means + FG%/FT% + MIN).
    Models:  A = season raw   B = recent raw   C = recency blend raw   D = C + empirical-Bayes shrink to archetype.
    prior: optional last-season per-game dict (FIELDS incl. MIN) blended into model D as PRIOR_SEASON_MIN minutes."""
    sp = splits or p.splits
    S = sp.get("S"); R = sp.get("L15")
    if S is None: return None
    if model == "A" or R is None:
        base_parts = [(S, 1.0)]; mins = S["MIN"]
    elif model == "B":
        base_parts = [(R, 1.0)]; mins = R["MIN"]
    else:
        pre = _sub(S, R)
        base_parts = [(R, 1.0)] + ([(pre, recency)] if pre else [])
        mins = min_recent * R["MIN"] + (1 - min_recent) * S["MIN"]
    tot_min = sum(s["MIN"] * s["GP"] * w for s, w in base_parts)
    use_prior = model == "D" and prior and prior.get("MIN", 0) > 0
    out = {"MIN": mins}
    for f in COUNT:
        num = sum(s[f] * s["GP"] * w for s, w in base_parts)
        den = tot_min
        if use_prior:
            num += PRIOR_SEASON_MIN * prior.get(f, 0) / prior["MIN"]; den += PRIOR_SEASON_MIN
        rate = num / max(den, 1e-9)
        if model == "D":
            k = K_MIN[f]; rate = (num + k * pri[p.arch][f]) / (den + k)
        out[f] = rate * mins
    fgm = sum(s["FGM"] * s["GP"] * w for s, w in base_parts); fga = sum(s["FGA"] * s["GP"] * w for s, w in base_parts)
    ftm = sum(s["FTM"] * s["GP"] * w for s, w in base_parts); fta = sum(s["FTA"] * s["GP"] * w for s, w in base_parts)
    if use_prior:
        sc = PRIOR_SEASON_MIN / prior["MIN"]
        fgm += sc * prior.get("FGM", 0); fga += sc * prior.get("FGA", 0)
        ftm += sc * prior.get("FTM", 0); fta += sc * prior.get("FTA", 0)
    if model == "D":
        out["FG%"] = (fgm + K_FG * pri[p.arch]["FG%"]) / (fga + K_FG)
        out["FT%"] = (ftm + K_FT * pri[p.arch]["FT%"]) / (fta + K_FT)
    else:
        out["FG%"] = fgm / fga if fga else pri[p.arch]["FG%"]
        out["FT%"] = ftm / fta if fta else pri[p.arch]["FT%"]
    return out


def backtest(players, pri):
    """Time-ordered test: predict last-7-day per-game output using only games BEFORE that window.
    train = season minus L7 (as 'S') and (L15 minus L7) as the 'recent' window. Returns RMSE by model (z-units)."""
    rows = []
    for p in players:
        S, L7, L15 = p.splits.get("S"), p.splits.get("L7"), p.splits.get("L15")
        if not (S and L7 and L15) or L7["GP"] < 2: continue
        pre = _sub(S, L7); mid = _sub(L15, L7)
        if not pre or pre["GP"] < 4: continue
        tr = {"S": pre}
        if mid and mid["GP"] >= 1: tr["L15"] = mid
        rows.append((p, tr, L7))
    sd = {f: np.std([L7[f] for _, _, L7 in rows]) + 1e-9 for f in COUNT}
    res = {}
    configs = {"A season": ("A", .5, .6), "B recent": ("B", .5, .6), "C blend": ("C", .5, .6),
               "D blend+shrink": ("D", .5, .6), "D' heavier recency": ("D", .8, .8), "D'' lighter recency": ("D", .3, .4)}
    for name, (m, rc, mr) in configs.items():
        err = []; mae_min = []
        for p, tr, L7 in rows:
            pr = project(p, pri, m, rc, mr, splits=tr)
            w = L7["GP"]
            err.append(w * np.mean([((pr[f] - L7[f]) / sd[f]) ** 2 for f in COUNT]))
            mae_min.append(w * abs(pr["MIN"] - L7["MIN"]))
        W = sum(L7["GP"] for _, _, L7 in rows)
        res[name] = {"rmse_z": math.sqrt(sum(err) / W), "mae_minutes": sum(mae_min) / W}
    return len(rows), res


# ---------------------------------------------------------------- value & lineup
def value_score(pr, pri):
    """simple 9-cat z-sum per game vs pool-ish scales; used only to ORDER lineup choices."""
    sc = {"PTS": 6, "REB": 3, "AST": 2.2, "STL": .6, "BLK": .6, "TO": 1.0, "3PM": 1.0}
    v = sum(pr[c] / sc[c] for c in sc if c != "TO") - pr["TO"] / sc["TO"]
    v += (pr["FG%"] - .47) * pr["FGA"] / 2.0 + (pr["FT%"] - .78) * pr["FTA"] / 1.2
    return v


def best_lineup(cands, pos_of, val):
    """cands: names available today. Greedy by value with augmenting-path feasibility = optimal for matroid."""
    order = sorted(cands, key=lambda n: -val[n])
    match = {}  # slot index -> name

    def try_assign(n, seen):
        for i, s in enumerate(SLOTS):
            if i in seen or not (SLOT_OK[s] & set(pos_of[n])): continue
            seen.add(i)
            if i not in match or try_assign(match[i], seen):
                match[i] = n; return True
        return False

    chosen = []
    for n in order:
        snap = dict(match)
        if try_assign(n, set()): chosen.append(n)
        else: match.clear(); match.update(snap)
        if len(chosen) == 10: break
    return chosen, {SLOTS[i] + (str(i) if SLOTS[i] == "UT" else ""): n for i, n in sorted(match.items())}


# ---------------------------------------------------------------- game simulation
def sample_games(pr, n, rng):
    """n simulated games for one player -> dict of arrays FGM,FGA,FTM,FTA,3PM,REB,AST,STL,BLK,TO,PTS."""
    mfac = np.exp(rng.normal(0, .15, n))
    g = rng.gamma(8, 1 / 8, (n, 1))
    lam = lambda f: np.maximum(pr[f], 0) * mfac * g[:, 0]
    fga = rng.poisson(lam("FGA")); fta = rng.poisson(lam("FTA"))
    pfg = np.clip(rng.beta(pr["FG%"] * 40, (1 - pr["FG%"]) * 40, n), .05, .95)
    pft = np.clip(rng.beta(pr["FT%"] * 30, (1 - pr["FT%"]) * 30 + 1e-6, n), .05, .99)
    fgm = rng.binomial(fga, pfg); ftm = rng.binomial(fta, pft)
    tpm = np.minimum(rng.poisson(lam("3PM")), fgm)
    out = {"FGM": fgm, "FGA": fga, "FTM": ftm, "FTA": fta, "3PM": tpm,
           "PTS": 2 * fgm + tpm + ftm}
    for f in ("REB", "AST", "STL", "BLK", "TO"): out[f] = rng.poisson(lam(f))
    return out


RAW = ["FGM", "FGA", "FTM", "FTA", "3PM", "REB", "AST", "STL", "BLK", "TO", "PTS"]


def to_cats(t):
    return {"FG": t["FGM"] / np.maximum(t["FGA"], 1), "FT": t["FTM"] / np.maximum(t["FTA"], 1),
            **{c: t[c] for c in ["3PM", "REB", "AST", "STL", "BLK", "TO", "PTS"]}}


def compare(a, b):
    """category results for arrays (sims) -> wins(+1)/loss(-1)/tie(0) per cat, and match outcome."""
    A, B = to_cats(a), to_cats(b); res = {}
    for c in CATS:
        d = (B[c] - A[c]) if c == "TO" else (A[c] - B[c])
        res[c] = np.sign(np.round(d, 6))
    w = sum((res[c] > 0) for c in CATS); l = sum((res[c] < 0) for c in CATS)
    return res, w, l


def sim_week(team_players, opp_players, projs, pri, games_by_day, play_prob, live, n=20000, seed=1,
             force_bench=None, scratch=0.05):
    """games_by_day: list of sets of NBA teams playing each remaining day.
    play_prob[name] -> list per day (prob of being active that day; known before lock).
    live: {"A": raw totals so far, "B": raw totals so far} (keys from RAW).
    Lineup each day = best legal 10 among available; then a late-scratch prob for ACTIVE starters (no replacement)."""
    rng = np.random.default_rng(seed)
    tot = {}
    for side, roster in (("A", team_players), ("B", opp_players)):
        T = {f: np.full(n, float(live[side].get(f, 0))) for f in RAW}
        val = {p.name: value_score(projs[p.name], pri) for p in roster}
        pos = {p.name: p.pos for p in roster}
        for d, teams in enumerate(games_by_day):
            playing = [p for p in roster if p.nba in teams and play_prob[p.name][d] > 0
                       and not (force_bench and p.name in force_bench.get(d, set()))]
            if not playing: continue
            avail = np.stack([rng.random(n) < play_prob[p.name][d] for p in playing], 1)
            keys = np.packbits(avail, axis=1); kk = [bytes(k) for k in keys]
            cache = {}
            start = np.zeros_like(avail)
            for i, k in enumerate(kk):
                if k not in cache:
                    names = [p.name for j, p in enumerate(playing) if avail[i, j]]
                    ch, _ = best_lineup(names, pos, val)
                    cache[k] = np.array([p.name in ch for p in playing])
                start[i] = cache[k]
            start &= rng.random(start.shape) > scratch
            for j, p in enumerate(playing):
                idx = np.where(start[:, j])[0]
                if len(idx) == 0: continue
                g = sample_games(projs[p.name], len(idx), rng)
                for f in RAW: T[f][idx] += g[f]
        tot[side] = T
    res, w, l = compare(tot["A"], tot["B"])
    out = {"p_win": float(np.mean(w > l)), "p_tie": float(np.mean(w == l)), "p_loss": float(np.mean(w < l)),
           "cat_win": {c: float(np.mean(res[c] > 0) + .5 * np.mean(res[c] == 0)) for c in CATS},
           "median_score": f"{int(np.median(w))}-{int(np.median(l))}",
           "proj": {s: {c: float(np.median(v)) for c, v in to_cats(tot[s]).items()} for s in ("A", "B")}}
    return out


# ---------------------------------------------------------------- rest-of-season roster value
DAY_W = np.array([.55, .7, .85, .55, .8, .7, .65])  # relative NBA game load Mon..Sun (approx.)


def random_week(nba_teams, rng):
    days = {}
    for t in nba_teams:
        k = 4 if rng.random() < .45 else 3
        p = DAY_W / DAY_W.sum()
        days[t] = set(rng.choice(7, k, replace=False, p=p))
    return [{t for t in nba_teams if d in days[t]} for d in range(7)]


def team_week_dist(roster, projs, pri, health, n_weeks=400, sims_per_week=10, seed=7, bank_size=4000, schedules=None):
    """Distribution of a roster's weekly raw totals over NBA schedules (common random numbers:
    same seed -> same schedules, so roster comparisons are paired). health[name] = P(active on a game day).
    schedules: optional list of real weeks (each a list of 7 sets of NBA teams) to cycle through instead of random weeks."""
    rng = np.random.default_rng(seed)
    teams = sorted({p.nba for p in roster})
    val = {p.name: value_score(projs[p.name], pri) for p in roster}
    pos = {p.name: p.pos for p in roster}
    bank = {p.name: sample_games(projs[p.name], bank_size, np.random.default_rng(_stable_seed(p.name))) for p in roster}
    bankM = {n: np.stack([b[f] for f in RAW], 1) for n, b in bank.items()}
    N = n_weeks * sims_per_week
    T = np.zeros((N, len(RAW))); games = np.zeros(N)
    cache = {}
    for w in range(n_weeks):
        sched = schedules[w % len(schedules)] if schedules else random_week(teams, rng)
        for d, tms in enumerate(sched):
            playing = [p for p in roster if p.nba in tms]
            if not playing: continue
            av = rng.random((sims_per_week, len(playing))) < np.array([health[p.name] for p in playing])
            for s in range(sims_per_week):
                key = tuple(p.name for j, p in enumerate(playing) if av[s, j])
                if key not in cache: cache[key] = best_lineup(list(key), pos, val)[0]
                ch = cache[key]; row = w * sims_per_week + s
                for nm in ch: T[row] += bankM[nm][rng.integers(bank_size)]
                games[row] += len(ch)
    return {f: T[:, i] for i, f in enumerate(RAW)}, float(games.mean())


def _stable_seed(name):
    """deterministic per-player seed (Python's hash() is salted per process, which broke pairing across runs)."""
    h = 2166136261
    for ch in name.encode():
        h = ((h ^ ch) * 16777619) % 2**32
    return h


def league_week_matrix(weekly_rows, scale_to_week=None):
    """historical team-weeks -> category arrays, counting cats rescaled so each week's league mean
    equals the mean of weeks>=2 (week 1 was a 6-day week)."""
    import collections
    byw = collections.defaultdict(list)
    for r in weekly_rows: byw[int(r["week"])].append(r)
    full = [w for w in byw if w >= 2] or list(byw)
    target = {c: np.mean([float(r[c]) for w in full for r in byw[w]]) for c in CATS}
    out = []
    for w, rs in byw.items():
        m = {c: np.mean([float(r[c]) for r in rs]) for c in CATS}
        for r in rs:
            row = {"team": r["team"], "week": w}
            for c in CATS:
                row[c] = float(r[c]) if c in ("FG", "FT") else float(r[c]) * target[c] / m[c]
            out.append(row)
    return out, target


def allplay(dist, league_rows, rng_seed=3, n=20000):
    """P(roster week beats a random historical league team-week) overall and per category."""
    rng = np.random.default_rng(rng_seed)
    i = rng.integers(0, len(dist["PTS"]), n); j = rng.integers(0, len(league_rows), n)
    A = to_cats({f: dist[f][i] for f in RAW})
    B = {c: np.array([league_rows[k][c] for k in j]) for c in CATS}
    w = np.zeros(n); l = np.zeros(n); per = {}
    for c in CATS:
        d = (B[c] - A[c]) if c == "TO" else (A[c] - B[c])
        per[c] = float(np.mean(d > 0) + .5 * np.mean(d == 0)); w += d > 0; l += d < 0
    return float(np.mean(w > l) + .5 * np.mean(w == l)), per


def actual_allplay(weekly_rows, week, team):
    """real all-play wins for one team-week: matchups won against each of the other 9 teams' totals."""
    rows = {r["team"]: r for r in weekly_rows if int(r["week"]) == week}
    me = rows[team]; wins = 0.0
    for t, r in rows.items():
        if t == team: continue
        w = l = 0
        for c in CATS:
            d = float(r[c]) - float(me[c]) if c == "TO" else float(me[c]) - float(r[c])
            w += d > 0; l += d < 0
        wins += 1 if w > l else .5 if w == l else 0
    return wins


# ---------------------------------------------------------------- season simulation
def team_strength(league_rows, shrink=0.35):
    """per-team category means (shrunk to league mean) + pooled residual vectors (keeps category correlation)."""
    import collections
    by = collections.defaultdict(list)
    for r in league_rows: by[r["team"]].append(r)
    lm = {c: np.mean([r[c] for r in league_rows]) for c in CATS}
    mean = {t: {c: (1 - shrink) * np.mean([r[c] for r in rs]) + shrink * lm[c] for c in CATS} for t, rs in by.items()}
    resid = np.array([[r[c] - np.mean([x[c] for x in by[r["team"]]]) for c in CATS] for r in league_rows])
    k = np.mean([len(v) for v in by.values()]); resid *= math.sqrt(k / (k - 1))  # undo demeaning shrinkage
    return mean, resid, lm


def season_sim(fixtures, standings, mean, resid, from_week, n=20000, seed=11, week_override=None,
               n_playoff=4, divisions=None, mean_override=None):
    """fixtures: rows with week, home, away. standings: {team: [W, L, T]} before from_week.
    week_override: {(week, team): p_win} for matchups simulated elsewhere (e.g. the live week).
    mean_override: {team: {cat: value}} to test a roster change (e.g. a trade) as a paired delta.
    Returns playoff%, final%, title%, mean final wins per team."""
    rng = np.random.default_rng(seed)
    teams = list(standings)
    W = np.array([[standings[t][0] + .5 * standings[t][2] for t in teams]] * n, float)
    CW = np.zeros_like(W)  # category-win tiebreaker proxy
    idx = {t: i for i, t in enumerate(teams)}
    mm = dict(mean); mm.update(mean_override or {})
    M = np.array([[mm[t][c] for c in CATS] for t in teams])
    sign = np.array([-1 if c == "TO" else 1 for c in CATS])
    weeks = sorted({int(f["week"]) for f in fixtures if int(f["week"]) >= from_week})

    def play(a, b):
        ra = resid[rng.integers(0, len(resid), n)]; rb = resid[rng.integers(0, len(resid), n)]
        d = (M[a] + ra - (M[b] + rb)) * sign
        return (d > 0).sum(1), (d < 0).sum(1)

    for wk in weeks:
        for f in [f for f in fixtures if int(f["week"]) == wk]:
            a, b = idx[f["home"]], idx[f["away"]]
            if week_override and (wk, f["home"]) in week_override:
                p = week_override[(wk, f["home"])]; r = rng.random(n)
                W[:, a] += r < p; W[:, b] += r >= p; continue
            if week_override and (wk, f["away"]) in week_override:
                p = week_override[(wk, f["away"])]; r = rng.random(n)
                W[:, b] += r < p; W[:, a] += r >= p; continue
            w, l = play(a, b)
            W[:, a] += (w > l) + .5 * (w == l); W[:, b] += (l > w) + .5 * (w == l)
            CW[:, a] += w; CW[:, b] += l
    key = W + CW / 1000 + rng.random(W.shape) / 1e6
    rank = (-key).argsort(1)
    seeds = rank[:, :n_playoff]
    po = np.array([np.mean((seeds == i).any(1)) for i in range(len(teams))])

    # playoffs: 1v4, 2v3, final; 2-week rounds -> residual sd / sqrt(2)
    def h2hv(a, b):
        ra = resid[rng.integers(0, len(resid), n)] / math.sqrt(2); rb = resid[rng.integers(0, len(resid), n)] / math.sqrt(2)
        d = (M[a] + ra - (M[b] + rb)) * sign
        w = (d > 0).sum(1); l = (d < 0).sum(1)
        return (w > l) | ((w == l) & (rng.random(n) < .5))

    A1, B1, A2, B2 = seeds[:, 0], seeds[:, 3], seeds[:, 1], seeds[:, 2]
    s1 = np.where(h2hv(A1, B1), A1, B1); s2 = np.where(h2hv(A2, B2), A2, B2)
    ch = np.where(h2hv(s1, s2), s1, s2)
    return {t: {"playoffs": float(po[i]), "final": float(np.mean((s1 == i) | (s2 == i))),
                "title": float(np.mean(ch == i)), "exp_wins": float(W[:, i].mean())}
            for t, i in idx.items()}
