"""Optional, compressed point-in-time research snapshots; no profile-table writes."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from nba_api.stats.endpoints import leaguedashptstats, synergyplaytypes, leaguedashplayershotlocations, leaguedashteamshotlocations
from nba.audit_tracking import load_measures
from nba.aggregate_playtypes import PLAY_TYPES
from nba.ingestion.sources import NBAClient, result_rows


def specifications(season, through):
    cutoff=through.strftime('%m/%d/%Y')
    for mode in ['Totals','PerGame']:
        for measure in load_measures():
            yield f'tracking_{measure}_{mode}',leaguedashptstats.LeagueDashPtStats,dict(
                season=season,season_type_all_star='Regular Season',per_mode_simple=mode,
                player_or_team='Player',pt_measure_type=measure,team_id_nullable=0,
                league_id_nullable='00',date_to_nullable=cutoff)
        for side,grouping in [('P','offensive'),('T','defensive')]:
            for play in PLAY_TYPES:
                # Synergy has no DateTo: captured_at, not a claimed historical
                # cutoff, determines whether this snapshot was available pregame.
                yield f'playtype_{side}_{play}_{mode}',synergyplaytypes.SynergyPlayTypes,dict(
                    season=season,season_type_all_star='Regular Season',per_mode_simple=mode,
                    player_or_team_abbreviation=side,type_grouping_nullable=grouping,play_type_nullable=play)
        for side,endpoint,measure in [('players',leaguedashplayershotlocations.LeagueDashPlayerShotLocations,'Base'),
                                      ('defense',leaguedashteamshotlocations.LeagueDashTeamShotLocations,'Opponent')]:
            yield f'shot_zones_{side}_{mode}',endpoint,dict(
                season=season,season_type_all_star='Regular Season',per_mode_detailed=mode,
                measure_type_simple=measure,distance_range='By Zone',team_id_nullable=0,
                league_id_nullable='00',date_to_nullable=cutoff,pace_adjust='N')


def collect_profiles(root, season, through, cadence='weekly', timeout=25):
    if cadence=='none':
        return {'cadence':'none','captured':0,'skipped':0,'failed':[]}
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    state_path=root/'latest.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {}
    now=datetime.now(timezone.utc)
    run_directory=root/f'{now:%Y%m%dT%H%M%S%fZ}'
    client=NBAClient(run_directory,timeout=timeout,retries=2)
    interval=timedelta(days=7 if cadence=='weekly' else 1)
    result={'cadence':cadence,'captured':0,'skipped':0,'failed':[],'directory':str(run_directory)}
    for label,endpoint,parameters in specifications(season,through):
        prior=state.get(label)
        if prior and now-datetime.fromisoformat(prior['captured_at'])<interval:
            result['skipped']+=1
            continue
        try:
            payload=client.fetch(endpoint,label,**parameters)
            sets=payload.get('resultSets',payload.get('resultSet'))
            sets=sets if isinstance(sets,list) else [sets]
            if not any(isinstance(s,dict) and s.get('rowSet') for s in sets):
                raise ValueError('No published rows yet')
            state[label]={'captured_at':now.isoformat(),'path':str(run_directory/f'{label}.json.gz'),
                          'date_cutoff':None if label.startswith('playtype_') else through.isoformat()}
            result['captured']+=1
        except Exception as exc:
            result['failed'].append({'dataset':label,'error':str(exc)})
            print(f'Optional research snapshot unavailable: {label}: {type(exc).__name__}',flush=True)
    temporary=state_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state,indent=2))
    temporary.replace(state_path)
    (run_directory/'manifest.json').write_text(json.dumps(result,indent=2))
    return result
