"""
espn_v2_tools.py — extra READ-ONLY tools for Matt's ESPN Fantasy MCP server (Northland v2).

Install: put this file next to your server's main file, then add two lines to the main file AFTER `mcp = FastMCP(...)`
is created and BEFORE `mcp.run(...)`:

    from espn_v2_tools import register
    register(mcp)

Cookies/IDs are read from environment variables (the ones your server already uses on Render).
Accepted names (first one found wins):
    ESPN_S2  | espn_s2 | ESPN_ESPN_S2
    SWID     | swid    | ESPN_SWID
    LEAGUE_ID| ESPN_LEAGUE_ID          (default 455752333)
    SEASON   | ESPN_SEASON | YEAR      (default 2027)

Tools added (all read-only; nothing here can change your team or league):
    nba_schedule(start_date, end_date)   every NBA game in a date range (US dates), with ESPN scoring period
    league_settings()                    playoff format, matchup periods, add limits, trade deadline, roster slots
    player_gamelog(name, last_n)         per-game box scores this season
    prior_season(name)                   2025-26 per-game averages
    transactions_history(from_period, to_period)   every add/drop/trade by scoring period
    espn_debug(name)                     raw stat-split metadata for one player (used once to verify the parsers)
"""
from __future__ import annotations
import os, json, time, datetime as dt
from typing import Optional
import requests
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York"); NZ = ZoneInfo("Pacific/Auckland")

STAT = {"0": "PTS", "1": "BLK", "2": "STL", "3": "AST", "6": "REB", "11": "TO", "13": "FGM", "14": "FGA",
        "15": "FTM", "16": "FTA", "17": "3PM", "18": "3PA", "40": "MIN", "42": "GP"}
PRO_TEAM = {0: "FA", 1: "ATL", 2: "BOS", 3: "NOP", 4: "CHI", 5: "CLE", 6: "DAL", 7: "DEN", 8: "DET", 9: "GSW",
            10: "HOU", 11: "IND", 12: "LAC", 13: "LAL", 14: "MIA", 15: "MIL", 16: "MIN", 17: "BKN", 18: "NYK",
            19: "ORL", 20: "PHL", 21: "PHO", 22: "POR", 23: "SAC", 24: "SAS", 25: "OKC", 26: "UTA", 27: "WAS",
            28: "TOR", 29: "MEM", 30: "CHA"}
ACQ = {"ADD": "ADDED", "DROP": "DROPPED", "LINEUP": "MOVED"}


def _env(*names, default=None):
    for n in names:
        v = os.environ.get(n)
        if v: return v
    return default


def _cfg():
    season = int(_env("SEASON", "ESPN_SEASON", "YEAR", default="2027"))
    league = int(_env("LEAGUE_ID", "ESPN_LEAGUE_ID", default="455752333"))
    cookies = {"espn_s2": _env("ESPN_S2", "espn_s2", "ESPN_ESPN_S2", default=""),
               "SWID": _env("SWID", "swid", "ESPN_SWID", default="")}
    base = f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{season}"
    return season, league, cookies, base


def _get(url, params=None, filt=None, tries=3):
    _, _, cookies, _ = _cfg()
    headers = {"X-Fantasy-Filter": json.dumps(filt)} if filt else {}
    last = None
    for i in range(tries):
        r = requests.get(url, params=params, headers=headers, cookies=cookies, timeout=30)
        if r.status_code == 200:
            return r.json()
        last = f"ESPN HTTP {r.status_code}: {r.text[:200]}"
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(last)


# ------------------------------------------------------------------ parsers (pure; unit-tested)
def parse_schedule(payload, start: dt.date, end: dt.date):
    games = {}
    for t in payload.get("settings", {}).get("proTeams", []):
        for period, gl in (t.get("proGamesByScoringPeriod") or {}).items():
            for g in gl:
                ts = dt.datetime.fromtimestamp(g["date"] / 1000, tz=dt.timezone.utc)
                us = ts.astimezone(ET).date()                      # US Eastern calendar date (DST-aware)
                if not (start <= us <= end): continue
                games[g["id"]] = {"us_date": us.isoformat(), "tip_utc": ts.strftime("%Y-%m-%dT%H:%MZ"),
                                  "tip_nz": ts.astimezone(NZ).strftime("%a %d %b %H:%M"),
                                  "away": PRO_TEAM.get(g.get("awayProTeamId"), str(g.get("awayProTeamId"))),
                                  "home": PRO_TEAM.get(g.get("homeProTeamId"), str(g.get("homeProTeamId"))),
                                  "scoring_period": int(period)}
    out = sorted(games.values(), key=lambda x: x["tip_utc"])
    per_team = {}
    for g in out:
        for side in ("away", "home"): per_team.setdefault(g[side], []).append(g["us_date"])
    return {"games": out, "games_per_team": {k: len(v) for k, v in sorted(per_team.items())}}


def parse_settings(payload):
    s = payload.get("settings", {})
    sch = s.get("scheduleSettings", {}); acq = s.get("acquisitionSettings", {}); tr = s.get("tradeSettings", {})
    ros = s.get("rosterSettings", {})
    dl = tr.get("deadlineDate")
    return {
        "name": s.get("name"),
        "regular_season_matchup_periods": sch.get("matchupPeriodCount"),
        "playoff_team_count": sch.get("playoffTeamCount"),
        "playoff_matchup_period_length": sch.get("playoffMatchupPeriodLength"),
        "playoff_seeding_rule": sch.get("playoffSeedingRule"),
        "playoff_seeding_tiebreaker": sch.get("playoffSeedingRuleBy"),
        "matchup_periods": sch.get("matchupPeriods"),
        "acquisition_limit_season": acq.get("acquisitionLimit"),
        "acquisition_limit_per_matchup": acq.get("matchupAcquisitionLimit"),
        "waiver_hours": acq.get("waiverHours"),
        "trade_deadline_utc": dt.datetime.fromtimestamp(dl / 1000, tz=dt.timezone.utc).isoformat() if dl else None,
        "lineup_slot_counts": ros.get("lineupSlotCounts"),
        "scoring_type": s.get("scoringSettings", {}).get("scoringType"),
    }


def _line(stats):
    d = {STAT[k]: v for k, v in (stats or {}).items() if k in STAT}
    for k in ("FGM", "FGA", "FTM", "FTA"): d.setdefault(k, 0)
    d["FG%"] = round(d["FGM"] / d["FGA"], 3) if d["FGA"] else None
    d["FT%"] = round(d["FTM"] / d["FTA"], 3) if d["FTA"] else None
    return d


def parse_gamelog(card, season, last_n):
    """per-game entries: actual stats (statSourceId 0) for one scoring period (scoringPeriodId > 0)."""
    games = []
    for st in card.get("player", {}).get("stats", []):
        if st.get("seasonId") != season or st.get("statSourceId") != 0: continue
        sp = st.get("scoringPeriodId", 0)
        if sp and sp > 0 and st.get("stats"):
            line = _line(st["stats"]); line["scoring_period"] = sp
            if line.get("MIN", 0) or line.get("FGA", 0) or line.get("PTS", 0): games.append(line)
    games.sort(key=lambda g: -g["scoring_period"])
    return games[:last_n]


def parse_prior(card, season):
    """season total/average for the previous season (statSplitTypeId 0, statSourceId 0)."""
    for st in card.get("player", {}).get("stats", []):
        if st.get("seasonId") == season - 1 and st.get("statSourceId") == 0 and st.get("statSplitTypeId") == 0:
            avg = st.get("averageStats") or {}
            if avg:
                return _line(avg) | {"GP": (st.get("stats") or {}).get("42")}
            tot = st.get("stats") or {}; gp = tot.get("42") or 0
            return _line({k: (v / gp if gp else 0) for k, v in tot.items()}) | {"GP": gp}
    return None


def parse_transactions(payload, names):
    out = []
    for t in payload.get("transactions", []):
        if t.get("status") != "EXECUTED": continue
        when = dt.datetime.fromtimestamp(t.get("processDate", t.get("proposedDate", 0)) / 1000, tz=dt.timezone.utc)
        for it in t.get("items", []):
            out.append({"utc": when.isoformat(), "type": t.get("type"), "action": ACQ.get(it.get("type"), it.get("type")),
                        "player": names.get(it.get("playerId"), it.get("playerId")),
                        "from_team": it.get("fromTeamId"), "to_team": it.get("toTeamId")})
    return out


# ------------------------------------------------------------------ fetch helpers
_PLAYERS = {"t": 0, "map": {}}


def _player_index():
    if time.time() - _PLAYERS["t"] > 6 * 3600 or not _PLAYERS["map"]:
        _, _, _, base = _cfg()
        data = _get(f"{base}/players", params={"scoringPeriodId": 0, "view": "players_wl"},
                    filt={"filterActive": {"value": True}})
        _PLAYERS["map"] = {p["id"]: p["fullName"] for p in data}
        _PLAYERS["t"] = time.time()
    return _PLAYERS["map"]


def _player_id(name):
    idx = _player_index(); low = name.lower().strip()
    exact = [i for i, n in idx.items() if n.lower() == low]
    if exact: return exact[0], idx[exact[0]]
    part = [i for i, n in idx.items() if low in n.lower()]
    if len(part) == 1: return part[0], idx[part[0]]
    raise ValueError(f"Player '{name}' not found or ambiguous: {[idx[i] for i in part[:8]]}")


def _card(pid, top_periods=82):
    season, league, _, base = _cfg()
    filt = {"players": {"filterIds": {"value": [pid]},
                        "filterStatsForTopScoringPeriodIds": {"value": top_periods,
                                                              "additionalValue": [f"00{season}", f"00{season-1}", f"10{season}"]}}}
    data = _get(f"{base}/segments/0/leagues/{league}", params={"view": "kona_playercard"}, filt=filt)
    pl = data.get("players", [])
    if not pl: raise RuntimeError("ESPN returned no player card")
    return pl[0]


# ------------------------------------------------------------------ MCP registration
def register(mcp):
    @mcp.tool()
    def nba_schedule(start_date: str, end_date: str) -> dict:
        """Every NBA game between two US dates (YYYY-MM-DD, inclusive): tip times in UTC and NZ, teams, ESPN scoring period, and games per team."""
        _, _, _, base = _cfg()
        data = _get(base, params={"view": "proTeamSchedules_wl"})
        return parse_schedule(data, dt.date.fromisoformat(start_date), dt.date.fromisoformat(end_date))

    @mcp.tool()
    def league_settings() -> dict:
        """League rules from ESPN: regular-season length, playoff teams and round length, seeding tiebreaker, add limits (season and per matchup), waiver hours, trade deadline, roster slots."""
        season, league, _, base = _cfg()
        return parse_settings(_get(f"{base}/segments/0/leagues/{league}", params={"view": "mSettings"}))

    @mcp.tool()
    def player_gamelog(name: str, last_n: int = 10) -> dict:
        """Per-game box scores this season for one player (newest first): MIN, PTS, REB, AST, STL, BLK, TO, 3PM, FGM/FGA, FTM/FTA, scoring period."""
        pid, full = _player_id(name); season = _cfg()[0]
        return {"name": full, "games": parse_gamelog(_card(pid), season, last_n)}

    @mcp.tool()
    def prior_season(name: str) -> dict:
        """One player's previous-season (2025-26) per-game averages, for projection blending."""
        pid, full = _player_id(name); season = _cfg()[0]
        return {"name": full, "season": season - 1, "per_game": parse_prior(_card(pid), season)}

    @mcp.tool()
    def transactions_history(from_period: int = 1, to_period: Optional[int] = None) -> dict:
        """Every executed add, drop and trade in the league between two scoring periods (days), oldest first."""
        season, league, _, base = _cfg()
        if to_period is None:
            to_period = _get(f"{base}/segments/0/leagues/{league}", params={"view": "mStatus"}).get("scoringPeriodId", from_period)
        names = _player_index(); out = []
        for sp in range(from_period, to_period + 1):
            data = _get(f"{base}/segments/0/leagues/{league}", params={"view": "mTransactions2", "scoringPeriodId": sp})
            out.extend(parse_transactions(data, names))
        uniq = {(o["utc"], o["player"], o["action"]): o for o in out}
        return {"transactions": sorted(uniq.values(), key=lambda o: o["utc"])}

    @mcp.tool()
    def espn_debug(name: str) -> dict:
        """Diagnostic: raw stat-split metadata ESPN returns for one player (no stat values). Used to verify the parsers."""
        pid, full = _player_id(name)
        card = _card(pid)
        meta = [{k: s.get(k) for k in ("seasonId", "statSourceId", "statSplitTypeId", "scoringPeriodId", "id")}
                | {"n_stats": len(s.get("stats") or {}), "has_avg": bool(s.get("averageStats"))}
                for s in card.get("player", {}).get("stats", [])]
        return {"name": full, "player_id": pid, "splits": meta[:120]}

    return mcp
