# northland-engine

Decision engine for **Northland Kauri** (ESPN league *Aotearoa Dynasty League*, ID 455752333, season 2027).
Private. Read by the daily "Northland daily lineup" task (6:24am NZ) and by chats in the *ESPN NBA Fantasy* project.
Nothing here can change the ESPN team; every transaction is Matt's.

## Layout
| Path | What |
|---|---|
| `engine/core.py` | Projections (model D: recency-weighted, shrunk per-minute rates, optional last-season blend), position-legal lineup optimiser, weekly matchup Monte Carlo (`sim_week`), rest-of-season roster value (`team_week_dist` + `allplay`), real all-play (`actual_allplay`), season/playoff sim (`team_strength` + `season_sim`). |
| `engine/ledger.py` | Forecast ledger: append, Brier scoring, reliability table. |
| `connector/espn_v2_tools.py` | Reference copy of the read-only tools added to the ESPN MCP server (`espn-connector` repo). |
| `data/fixtures.csv` | All 20 regular-season weeks (home/away abbrevs), from ESPN `matchup_scores`. |
| `data/weekly_totals.csv` | Final 9-category totals for every team-week played. Append each Tuesday. |
| `data/league.json` | Standings, adds used, divisions, rules, Northland contracts, stream budget. |
| `data/forecast_log.csv` | Pre-week forecasts and their scores (Brier). |
| `data/decision_log.csv` | Recommendations, alternatives, evidence, confidence, what Matt did, outcome. |
| `scripts/season_odds.py` | `python -m scripts.season_odds <week> --override NTH=<p_win this week>` |
| `tests/` | `python -m pytest -q` |

## Team abbreviations
CAN Canterbury Earthshakers · AKL Auckland Marauders · MAR Marlborough Megabytes · SOU Southland Aphros ·
MAN Manawatu Windstorm · WLG Wellington Taniwha · TAS Te Tai-o-Aorere Skywalkers · NTH Northland Kauri ·
HWK Hawkes Bay Gannets · OTG Wanaka Wukongs

## Weekly upkeep (Tuesday review)
1. Append last week's 10 rows to `data/weekly_totals.csv` from `matchup_scores(<week>)`; update `standings` and `season_adds` in `data/league.json`.
2. Score last week's forecast: `ledger.score_week("data/forecast_log.csv", week, outcome, cat_outcomes)`.
3. Run the week sim and append the new forecast; run `scripts/season_odds.py`.
4. Commit with message `Week <n> review`.

## Rules the engine encodes
Decisions use **differences** between rosters on paired random numbers, never absolute levels (roster model reads ~8% low on volume).
Streaming policy (adopted 18 Nov 2026): stream only when the week is 25–75% and one add is worth ≥ +5 points; never above 85% or below 15%; budget 19 streams Weeks 6–20, 6 injury reserve, 6 playoff reserve; multi-year players are never dropped.

## Changes from the 18 Nov project copy
- Per-player random seeds now use a stable hash (Python's `hash()` is salted per process, so paired comparisons weren't reproducible across runs).
- `project(..., prior=)` blends last season's per-minute rates (300 minutes), for `prior_season()` data.
- `team_week_dist(..., schedules=)` can use the real NBA schedule from `nba_schedule()`.
- `season_sim(..., mean_override=)` tests trades as paired deltas; `actual_allplay()` added.
- Check: `season_odds 5 --override NTH=0.04` reproduces the 18 Nov odds (AKL 96%, CAN 82%, MAR 69%, NTH 15%).
