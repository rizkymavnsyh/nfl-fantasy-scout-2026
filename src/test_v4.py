"""Check new features do not read target-week or future source outcomes."""
import pandas as pd
import numpy as np
from context_features import enrich
from forecast_models import make_model

frame=pd.DataFrame([{'player_id':'p','team':'A','opponent':'B','position':'QB','season':2023,'week':3,
                    'attempts_mean3':20.,'carries_mean3':2.,'targets_mean3':0.,'fantasy_points_ppr_mean3':15.}])
stats=pd.DataFrame([{'team':'A','opponent_team':'B','position':'QB','season':2023,'week':w,
                      'attempts':10.*w,'carries':float(w),'targets':0.,'fantasy_points_ppr':10.*w} for w in [1,2,3,4]])
original,cols=enrich(frame,stats)
changed=stats.copy();changed.loc[changed.week>=3,['attempts','carries','fantasy_points_ppr']]=9999.
mutated,_=enrich(frame,changed)
pd.testing.assert_frame_equal(original[cols],mutated[cols])
assert original.iloc[0].v4_team_attempts_mean3==15.
later=frame.copy();later['week']=4
new,_=enrich(later,changed)
assert new.iloc[0].v4_team_attempts_mean3>original.iloc[0].v4_team_attempts_mean3
other_season=stats.copy();other_season['season']=2024
pd.testing.assert_frame_equal(original[cols],enrich(frame,pd.concat([stats,other_season]))[0][cols])

# Small synthetic end-to-end fit, preserving nonconsecutive DataFrame indices.
X=pd.DataFrame({'position':['QB','RB','WR','TE']*30,'form':np.tile([0.,5.,10.,15.,20.,25.],20)},index=np.arange(120)*2)
y=np.maximum(0,X.form.to_numpy()-3.)
for family in ['position','hurdle']:
    model=make_model(['position','form'],family=family,loss='regression',iterations=5,minimum=5,leaves=3).fit(X,y)
    p=model.predict(X.iloc[:8]);assert p.shape==(8,) and np.isfinite(p).all()
print('PASS: same/future-week source perturbations excluded, prior-week changes visible, season isolation, specialist and hurdle fits.')
X['season']=np.repeat([2021,2022,2023],40)
model=make_model(['position','form','season'],family='recency',loss='regression',iterations=5,minimum=5,leaves=3).fit(X,y)
prediction=model.predict(X.iloc[:8]);alternate=X.iloc[:8].copy();alternate['season']=2099
assert np.allclose(prediction,model.predict(alternate))
print('PASS: recency season metadata changes fit weights but is excluded from predictor inputs.')
