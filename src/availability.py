"""Timestamped nflverse roster/injury evidence. ACT does not mean game-day active."""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import pandas as pd
import requests
from pipeline import ROOT, dump, TEAM_ALIASES

SOURCES = {
    'roster': ('roster_2026.csv', 'https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_2026.csv'),
    'injuries': ('injuries_2026.csv', 'https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_2026.csv')}

def load_sources(download=True):
    session = requests.Session(); session.trust_env = False
    now = datetime.now(timezone.utc).isoformat()
    manifest_path = ROOT/'reports/availability_sources.json'
    previous = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    frames, manifest = {}, {}
    for key, (name, url) in SOURCES.items():
        path = ROOT/'data/live'/name
        try:
            if download:
                response = session.get(url, timeout=60); response.raise_for_status()
                # Parse before replacing a working cached source.
                import io
                frame = pd.read_csv(io.BytesIO(response.content), low_memory=False)
                required = {'gsis_id', 'team', 'week', 'status' if key=='roster' else 'report_status'}
                if not required.issubset(frame.columns): raise ValueError('Schema sumber berubah')
                path.write_bytes(response.content)
                item = {'url':url, 'retrieved_at_utc':now, 'last_modified':response.headers.get('Last-Modified'),
                        'sha256':hashlib.sha256(response.content).hexdigest(), 'error':None}
            else:
                item = previous.get(key, {'url':url, 'retrieved_at_utc':None,
                                         'error':'Cache belum memiliki waktu pengambilan yang terverifikasi'})
                frame = pd.read_csv(path, low_memory=False)
            if download is False and item.get('sha256') != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError('Hash cache berbeda dari manifest')
            frames[key] = frame
            manifest[key] = item
        except (requests.RequestException, OSError, ValueError, KeyError) as exc:
            # Never silently turn a failed/stale source into a healthy-player label.
            manifest[key] = {'url':url, 'retrieved_at_utc':now, 'error':str(exc)}
    dump(manifest_path, manifest)
    return frames, manifest

def annotate(players, roster, injuries, manifest, now=None):
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    rows = []
    roster = roster.copy() if roster is not None else pd.DataFrame()
    if len(roster):
        roster = roster[roster.season.eq(2026)].dropna(subset=['gsis_id'])
        roster['team'] = roster.team.replace(TEAM_ALIASES)
    for p in players.to_dict('records'):
        rr = roster[roster.gsis_id.eq(p['player_id'])] if len(roster) else roster
        if len(rr): rr = rr.sort_values('week').tail(1)
        status = str(rr.status.iloc[0]) if len(rr) else 'UNKNOWN'
        team = str(rr.team.iloc[0]) if len(rr) else None
        evidence = manifest.get('roster', {})
        stamp = evidence.get('retrieved_at_utc')
        fresh = bool(stamp and not evidence.get('error') and 0 <= (now-pd.Timestamp(stamp)).total_seconds() <= 48*3600)
        if evidence.get('last_modified'):
            age = (now-pd.Timestamp(parsedate_to_datetime(evidence['last_modified']))).total_seconds()
            fresh = fresh and 0 <= age <= 48*3600
        current = pd.DataFrame()
        if injuries is not None and len(injuries):
            current = injuries[(injuries.gsis_id.eq(p['player_id'])) & (injuries.week.eq(p['week'])) &
                               (injuries.team.replace(TEAM_ALIASES).eq(p['team']))]
        # Conflicting duplicate reports are not silently resolved by row ordering.
        labels = current.report_status.fillna('Not listed').unique() if len(current) else []
        injury = str(labels[0]) if len(labels)==1 else 'Conflicting reports' if len(labels)>1 else 'Not published'
        bad = injury.lower() in {'out', 'doubtful', 'conflicting reports'}
        ie = manifest.get('injuries', {})
        istamp = ie.get('retrieved_at_utc')
        injury_fresh = bool(istamp and not ie.get('error') and 0 <= (now-pd.Timestamp(istamp)).total_seconds() <= 48*3600)
        # Stale injury evidence can only add caution, never grant eligibility.
        eligible = fresh and status=='ACT' and team==p['team'] and not bad
        reason = ('Roster not verified or stale' if not fresh else
                  'Roster team differs from model team' if team and team!=p['team'] else
                  'Roster status '+status if status!='ACT' else
                  'Game injury report: '+injury if bad else
                  'Active roster (ACT). Game-day status not confirmed')
        p.update(roster_status=status, roster_team=team, roster_week=int(rr.week.iloc[0]) if len(rr) else None,
                 injury_status=injury, injury_week=int(p['week']) if len(current) else None,
                 injury_report_fresh=injury_fresh, availability_as_of_utc=stamp,
                 lineup_eligible=bool(eligible), availability_note=reason,
                 risk=reason+' | Game injury report: '+injury)
        rows.append(p)
    return pd.DataFrame(rows)

def update(download=True):
    frames, manifest = load_sources(download)
    forecasts = pd.read_csv(ROOT/'reports/forecasts_next_game.csv',float_precision='round_trip')
    forecasts = annotate(forecasts, frames.get('roster'), frames.get('injuries'), manifest)
    for name in ['forecasts_next_game.csv','forecasts_week8.csv']:
        forecasts.to_csv(ROOT/'reports'/name,index=False)
    fields = ['roster_status','roster_team','roster_week','injury_status','injury_week',
              'injury_report_fresh','availability_as_of_utc','lineup_eligible','availability_note','risk']
    payload = json.loads((ROOT/'dist/data.json').read_text(encoding='utf-8'))
    lookup = forecasts.set_index('player_id')
    for player in payload['players']:
        row = lookup.loc[player['id']]
        for field in fields:
            value = row[field]
            player[field] = None if pd.isna(value) else value.item() if hasattr(value,'item') else value
    summary = {'sources':manifest,'eligible':int(forecasts.lineup_eligible.sum()),'total':len(forecasts),
               'checked_at_utc':datetime.now(timezone.utc).isoformat(),
               'injury_latest_week':int(frames['injuries'].week.max()) if 'injuries' in frames else None,
               'game_day_active_confirmed':False,'freshness_hours':48}
    payload['availability'] = summary
    dump(ROOT/'reports/availability.json',summary); dump(ROOT/'dist/data.json',payload)
    extra=[c for c in ['fantasy_pool','prior_snap_pct','prior_snap_weeks_ago','prior_injury_note'] if c in forecasts]
    forecasts[['player_id','name','position','team','opponent','season','week','game_id','kickoff_utc',
               'prediction','low','high','baseline']+fields+extra].to_csv(ROOT/'dist/forecasts.csv',index=False)
    print(json.dumps({k:v for k,v in summary.items() if k!='sources'}))
    return summary

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--cached',action='store_true')
    update(not parser.parse_args().cached)
