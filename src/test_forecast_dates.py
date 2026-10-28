"""Upcoming-game, DST, season-label and forecast timestamp checks."""
import json
import pandas as pd
from pipeline import ROOT,dump
from refresh_forecasts import upcoming_games,kickoff_utc

games=pd.read_csv(ROOT/'data/live/games.csv',low_memory=False)
cutoff=pd.Timestamp('2026-10-27T12:00:00+07:00')
# Reconstruct only the schedule's time/status boundary at this fixed test date.
historical=games.copy()
times=kickoff_utc(historical)
historical.loc[times>cutoff,['home_score','away_score']]=float('nan')
upcoming=upcoming_games(historical,cutoff)
assert upcoming.iloc[0].game_id=='2026_08_CAR_GB'
assert upcoming.iloc[0].kickoff_utc.tz_convert('Asia/Jakarta')==pd.Timestamp('2026-10-30T07:15:00+07:00')
assert not upcoming.week.eq(7).any()
before=upcoming_games(historical,pd.Timestamp('2026-10-30T07:14:00+07:00'))
after=upcoming_games(historical,pd.Timestamp('2026-10-30T07:16:00+07:00'))
assert before.iloc[0].game_id=='2026_08_CAR_GB'
assert '2026_08_CAR_GB' not in set(after.game_id)
# DST ends Nov 1, so a 13:00 Eastern start becomes 01:00 WIB next day.
row=upcoming[upcoming.game_id=='2026_08_ARI_DAL'].iloc[0]
assert row.kickoff_utc.tz_convert('Asia/Jakarta')==pd.Timestamp('2026-11-02T01:00:00+07:00')
january=upcoming_games(historical,pd.Timestamp('2027-01-08T12:00:00+07:00'))
assert len(january)>0 and january.season.eq(2026).all() and january.week.eq(18).all()
assert upcoming_games(historical,pd.Timestamp('2027-01-12T12:00:00+07:00')).empty
payload=json.loads((ROOT/'dist/data.json').read_text(encoding='utf-8'))
snapshot=payload['forecast_snapshot'];asof=pd.Timestamp(snapshot['as_of_utc'])
forecasts=pd.read_csv(ROOT/'reports/forecasts_next_game.csv')
assert not forecasts.player_id.duplicated().any()
assert (pd.to_datetime(forecasts.kickoff_utc,utc=True)>asof).all()
actual_upcoming=upcoming_games(games,asof)
next_by_team={}
for game in actual_upcoming.to_dict('records'):
    next_by_team.setdefault(game['home_team'],game['game_id'])
    next_by_team.setdefault(game['away_team'],game['game_id'])
assert all(r.game_id==next_by_team[r.team] for r in forecasts.itertuples())
assert len(payload['players'])==len(forecasts)
assert snapshot['latest_complete_week']>=7
assert isinstance(snapshot['missing_completed_games'],list)
result={'week7_not_forecast_after_completion':True,'first_week8_kickoff_wib_correct':True,
        'started_game_removed_even_if_scores_missing':True,'US_DST_conversion_correct':True,
        'January2027_belongs_to_season2026':True,'postseason_not_silently_forecast':True,
        'each_player_targets_next_game_of_team':True,'all_forecasts_after_asof':True,
        'week7_or_later_complete':True,'source_gaps_disclosed':True,'forecast_n':len(forecasts),
        'weeks':sorted(int(w) for w in forecasts.week.unique())}
dump(ROOT/'reports/forecast_date_verification.json',result)
print(json.dumps(result,indent=2))
