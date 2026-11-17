import datetime as dt, sys, os, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "connector"))
sys.modules.setdefault("requests", types.ModuleType("requests"))
import espn_v2_tools as t


def test_schedule_us_date_and_nz_time():
    ts = int(dt.datetime(2026, 11, 25, 0, 30, tzinfo=dt.timezone.utc).timestamp() * 1000)  # 7:30pm ET Tue 24 Nov
    payload = {"settings": {"proTeams": [{"proGamesByScoringPeriod": {"36": [{"id": 1, "date": ts, "awayProTeamId": 6, "homeProTeamId": 2}]}},
                                         {"proGamesByScoringPeriod": {"36": [{"id": 1, "date": ts, "awayProTeamId": 6, "homeProTeamId": 2}]}}]}}
    out = t.parse_schedule(payload, dt.date(2026, 11, 23), dt.date(2026, 11, 29))
    assert len(out["games"]) == 1
    g = out["games"][0]
    assert g["us_date"] == "2026-11-24" and g["tip_nz"] == "Wed 25 Nov 13:30" and g["away"] == "DAL"
    assert out["games_per_team"] == {"BOS": 1, "DAL": 1}


def test_settings():
    p = {"settings": {"scheduleSettings": {"matchupPeriodCount": 20, "playoffTeamCount": 4, "playoffMatchupPeriodLength": 2},
                      "acquisitionSettings": {"acquisitionLimit": 40, "matchupAcquisitionLimit": 6}}}
    s = t.parse_settings(p)
    assert s["regular_season_matchup_periods"] == 20 and s["playoff_team_count"] == 4 and s["acquisition_limit_season"] == 40


def test_gamelog_and_prior():
    card = {"player": {"stats": [
        {"seasonId": 2027, "statSourceId": 0, "scoringPeriodId": 28, "stats": {"0": 20, "13": 8, "14": 15, "40": 33}},
        {"seasonId": 2027, "statSourceId": 0, "scoringPeriodId": 26, "stats": {"0": 12, "13": 5, "14": 12, "40": 30}},
        {"seasonId": 2027, "statSourceId": 1, "scoringPeriodId": 29, "stats": {"0": 18}},
        {"seasonId": 2026, "statSourceId": 0, "statSplitTypeId": 0, "scoringPeriodId": 0,
         "stats": {"42": 70}, "averageStats": {"0": 16.2, "13": 6.3, "14": 12.7}}]}}
    g = t.parse_gamelog(card, 2027, 5)
    assert [x["scoring_period"] for x in g] == [28, 26] and g[0]["FG%"] == round(8 / 15, 3)
    pr = t.parse_prior(card, 2027)
    assert pr["PTS"] == 16.2 and pr["GP"] == 70
