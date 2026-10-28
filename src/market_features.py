"""Pre-game betting market context from the nflverse schedule (spread and total lines)."""
import numpy as np
import pandas as pd
from pipeline import ROOT,TEAM_ALIASES

LEAGUE_IMPLIED=22.5

def load_games(path=None):
    games=pd.read_csv(path or ROOT/'data/raw/games.csv',low_memory=False)
    games['home_team']=games.home_team.replace(TEAM_ALIASES);games['away_team']=games.away_team.replace(TEAM_ALIASES)
    return games

def enrich(frame,games):
    frame=frame.drop(columns=[c for c in frame if c.startswith('v6_')]).reset_index(drop=True).copy()
    lines=games[['game_id','home_team','spread_line','total_line','roof','div_game']].drop_duplicates('game_id')
    joined=frame[['game_id','team']].merge(lines,on='game_id',how='left',validate='many_to_one')
    # nflverse spread_line is positive when the home team is favored.
    side=np.where(joined.team.eq(joined.home_team),1.,-1.)
    spread=pd.to_numeric(joined.spread_line,errors='coerce')*side
    total=pd.to_numeric(joined.total_line,errors='coerce')
    frame['v6_team_spread']=spread.to_numpy()
    frame['v6_total_line']=total.to_numpy()
    frame['v6_implied_team_total']=((total+spread)/2).to_numpy()
    frame['v6_implied_opp_total']=((total-spread)/2).to_numpy()
    frame['v6_dome']=joined.roof.isin(['dome','closed']).astype(float).to_numpy()
    frame['v6_div_game']=pd.to_numeric(joined.div_game,errors='coerce').to_numpy()
    frame['v6_form_x_implied']=frame.v2_ppr_ewm6*frame.v6_implied_team_total/LEAGUE_IMPLIED
    cols=[c for c in frame if c.startswith('v6_')]
    return frame,cols
