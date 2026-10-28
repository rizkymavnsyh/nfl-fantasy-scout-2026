"""Guard role/injury timing and prevent hindsight-based fantasy candidate filters."""
import numpy as np
import pandas as pd
from pregame_features import enrich,fantasy_pool

frame=pd.DataFrame([{'player_id':'A','team':'X','position':'QB','opponent':'Y','season':2023,'week':w,
 'date':'2023-09-20','game_id':f'G{w}','attempts_mean3':20.,'targets_mean3':0.,'carries_mean3':2.,
 'fantasy_points_ppr_mean3':15.} for w in [2,3]])
snaps=pd.DataFrame([{'player_id':pid,'team':'X','season':2023,'week':w,'game_id':f'G{w}',
                     'offense_snaps':50.,'offense_pct':pct} for pid,w,pct in [('A',1,.5),('B',2,.8),('A',3,.9)]])
injuries=pd.DataFrame([{'player_id':'A','team':'X','season':2023,'week':w,'report_status':'Out',
 'practice_status':'Did Not Participate In Practice','modified':pd.Timestamp('2023-09-15',tz='UTC')} for w in [2,3]])
got,cols=enrich(frame,snaps,injuries)
assert got.iloc[1].v5_snap_pct_lag1==0. # Candidate absent from a supplied prior game.
assert got.iloc[1].v5_injury_prev_out==1.
assert got.iloc[0].v5_injury_prev_out==0. # Week-2 injury cannot enter week-2 forecast.
changed=snaps.copy();changed.loc[changed.week>=3,['offense_snaps','offense_pct']]=999.
altered=injuries.copy();altered.loc[altered.week>=3,'report_status']='Questionable'
pd.testing.assert_frame_equal(got[cols],enrich(frame,changed,altered)[0][cols])
late=injuries.copy();late['modified']=pd.Timestamp('2023-09-21',tz='UTC')
assert enrich(frame,snaps,late)[0].iloc[1].v5_injury_prev_out==0.
without=snaps[snaps.week.ne(2)]
assert enrich(frame,without,injuries)[0].iloc[1].v5_snap_pct_lag1==.5 # Missing source is not zero.
a=got.copy();a['target']=[0.,50.];b=a.copy();b['target']=[100.,0.]
assert np.array_equal(fantasy_pool(a),fantasy_pool(b))
assert all(fantasy_pool(got))
inference,_=enrich(frame.iloc[-1:],snaps,injuries,candidate_history=frame.iloc[:-1])
np.testing.assert_allclose(got.iloc[-1][cols].to_numpy(dtype=float),inference.iloc[0][cols].to_numpy(dtype=float),equal_nan=True)
unknown=frame.copy();unknown['player_id']='C'
future_only=snaps.copy();future_only.loc[future_only.player_id.eq('A'),'player_id']='C'
future_only=future_only[~((future_only.player_id=='C')&(future_only.week<3))]
without_future=future_only[future_only.player_id.ne('C')]
pd.testing.assert_frame_equal(enrich(unknown,future_only,injuries)[0][cols],enrich(unknown,without_future,injuries)[0][cols])
print('PASS: target-week snaps/injuries excluded, late updates rejected, missing-source distinction, candidate filter ignores outcomes.')
