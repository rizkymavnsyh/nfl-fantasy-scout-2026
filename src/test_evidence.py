"""Behavioral checks for availability, frozen inputs, and post-game labels."""
import json
import tempfile
from pathlib import Path
import pandas as pd
from availability import annotate
from prospective import lineup,score_snapshot,sha

now='2026-10-27T14:00:00Z'
players=pd.DataFrame([{'player_id':'A','team':'GB','season':2026,'week':8},
                      {'player_id':'B','team':'GB','season':2026,'week':8},
                      {'player_id':'C','team':'GB','season':2026,'week':8}])
roster=pd.DataFrame([{'gsis_id':p,'team':'GB','status':s,'season':2026,'week':8} for p,s in [('A','ACT'),('B','RES'),('C','ACT')]])
injuries=pd.DataFrame([{'gsis_id':'A','team':'GB','week':7,'report_status':'Out'},
                      {'gsis_id':'C','team':'GB','week':8,'report_status':'Out'}])
manifest={k:{'retrieved_at_utc':now,'error':None} for k in ['roster','injuries']}
got=annotate(players,roster,injuries,manifest,now)
assert got.lineup_eligible.tolist()==[True,False,False]
assert got.injury_status.tolist()==['Not published','Not published','Out']
assert not annotate(players,roster,injuries,manifest,'2026-10-30T14:00:00Z').lineup_eligible.any()
changed=roster.copy();changed.loc[0,'team']='NYJ'
assert not annotate(players,changed,injuries,manifest,now).iloc[0].lineup_eligible
assert not annotate(players,None,None,{},now).lineup_eligible.any()

positions=['QB','QB','RB','RB','RB','WR','WR','WR','TE','TE']
pool=pd.DataFrame([{'player_id':str(i),'position':p,'prediction':10-i,'baseline':i} for i,p in enumerate(positions)])
ids=lineup(pool,'prediction');assert len(ids)==len(set(ids))==7
assert ids!=lineup(pool,'baseline')
assert lineup(pool[pool.position.eq('QB')],'prediction') is None

with tempfile.TemporaryDirectory() as temp:
    dest=Path(temp)
    pd.DataFrame([{'player_id':'A','name':'One','season':2026,'week':8,'game_id':'G',
                   'kickoff_utc':'2026-10-30T00:15:00Z','prediction':5.,'baseline':4.}]).to_csv(dest/'forecasts.csv',index=False)
    (dest/'lineups.json').write_text('[]')
    manifest={'frozen_at_utc':now,'forecast_sha256':sha(dest/'forecasts.csv'),'lineups_sha256':sha(dest/'lineups.json')}
    (dest/'manifest.json').write_text(json.dumps(manifest))
    games=pd.DataFrame([{'game_id':'G','gameday':'2026-10-29','gametime':'20:15','home_team':'GB','away_team':'CAR','home_score':20.,'away_score':10.}])
    stats=pd.DataFrame([{'game_id':'G','player_id':'other1','team':'GB','fantasy_points_ppr':4.},
                        {'game_id':'G','player_id':'other2','team':'CAR','fantasy_points_ppr':2.}])
    _,_,r=score_snapshot(dest,stats,games,now='2026-10-31T00:00:00Z');assert r['n_scored']==1 and r['mae']==5.
    # Do not manufacture zero when one team's stats are missing.
    _,_,r=score_snapshot(dest,stats.iloc[:1],games,now='2026-10-31T00:00:00Z');assert r['n_scored']==0
    # No scores from a future game, even if somebody supplies future final scores.
    _,_,r=score_snapshot(dest,stats,games,now=now);assert r['n_scored']==0
    (dest/'forecasts.csv').write_text('modified')
    try:score_snapshot(dest,stats,games,now=now)
    except ValueError as exc:assert 'Hash' in str(exc)
    else:raise AssertionError('Tampered snapshot accepted')
print('PASS: stale/missing roster, team mismatch, reserve, target-week injury, unique lineup, future/missing results, and snapshot tampering.')
