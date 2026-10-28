"""Target-week play-by-play must not change V7 features for that week."""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from pipeline import ROOT
from redzone_features import enrich,load_plays,player_games,VALUES

plays=load_plays()
games,rates=player_games(plays)
assert set(rates['rush'])==set(range(len(rates['buckets'])-1)),'Every field-position bucket needs a 2019-2022 rate.'
assert rates['rush'][0]>rates['rush'][4],'Goal-line carries must score more often than carries at the 20-40.'
frame=pd.read_csv(ROOT/'reports/feature_dataset.csv',nrows=None)
frame=frame[frame.season.eq(2024)&frame.week.between(8,10)].reset_index(drop=True)
frame['v6_implied_team_total']=22.5
a,cols=enrich(frame,games)
changed=games.copy()
mask=changed.season.eq(2024)&changed.week.eq(9)
changed.loc[mask,VALUES]+=50
b,_=enrich(frame,changed)
week9=a.week.eq(9)
assert_frame_equal(a.loc[week9,cols],b.loc[week9,cols],check_exact=True)
week10=a.week.eq(10)
assert not np.allclose(a.loc[week10,'v7_x_td_mean3'].fillna(0),b.loc[week10,'v7_x_td_mean3'].fillna(0)),'Later weeks should respond to new history.'
print('PASS: target-week play-by-play does not leak into V7 features;',len(cols),'features,',f"coverage {a.v7_x_td_mean3.notna().mean():.1%}")
