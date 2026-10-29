"""Collect/validate current-season NBA data; --apply commits a prepared transaction."""

import argparse
from datetime import date, datetime, timedelta, timezone
import fcntl
import gzip
import json
from pathlib import Path
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from nba.audit_gamelogs import season_argument
from nba.ingestion.sources import NBAClient, EASTERN, current_season, collect_core, collect_missing_profiles, collect_ambiguous_box_scores
from nba.ingestion.database import snapshot_database, build_plan, apply_plan
from nba.ingestion.profiles import collect_profiles

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_DATA=ROOT/'python/data/raw/ingestion'


def load_json(path):
    if str(path).endswith('.gz'):
        with gzip.open(path,'rt') as handle:
            return json.load(handle)
    return json.loads(Path(path).read_text())


def write_json(path,data,compressed=False):
    if compressed:
        with gzip.open(path,'wt') as handle:
            json.dump(data,handle,default=str,separators=(',',':'))
    else:
        Path(path).write_text(json.dumps(data,indent=2,default=str))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--season',type=season_argument)
    parser.add_argument('--through',type=date.fromisoformat,help='Last date to include; defaults to yesterday in New York')
    parser.add_argument('--apply',action='store_true',help='Commit after validation; default is a read-only preview')
    parser.add_argument('--apply-plan',type=Path,help='Apply an existing reviewed plan.json without refetching NBA data')
    parser.add_argument('--source',type=Path,help='Replay a saved source.json.gz for preparation')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--data-root',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--timeout',type=int,default=25)
    parser.add_argument('--retries',type=int,default=3)
    parser.add_argument('--scoreboard-days',type=int,default=7)
    parser.add_argument('--profiles',choices=['none','weekly','daily'],default='none')
    parser.add_argument('--allow-historical',action='store_true',help='Explicitly permit writing a previous season')
    args=parser.parse_args()
    if args.timeout<=0 or args.retries<1 or args.scoreboard_days<1:
        parser.error('Timeout, retries and scoreboard days must be positive')
    if args.apply_plan and (args.source or args.output_dir or args.apply):
        parser.error('--apply-plan cannot be combined with --source, --output-dir or --apply')
    source=load_json(args.source) if args.source else None
    plan=load_json(args.apply_plan) if args.apply_plan else None
    season=season_argument(args.season or (plan or source or {}).get('season') or current_season())
    through=args.through or (date.fromisoformat((plan or source)['through']) if plan or source else datetime.now(EASTERN).date()-timedelta(days=1))
    if through>datetime.now(EASTERN).date():
        parser.error('Cannot ingest future dates')
    if season!=current_season() and (args.apply or args.apply_plan) and not args.allow_historical:
        parser.error('Historical writes require --allow-historical; default daily ingestion only changes the current season')
    season_root=args.data_root/season
    season_root.mkdir(parents=True,exist_ok=True)
    with (season_root/'ingestion.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another ingestion run is active; skipping duplicate invocation.',flush=True)
            return
        if plan:
            if season!=plan['season'] or through.isoformat()!=plan['through']:
                parser.error('--apply-plan season/cutoff must match the prepared plan')
            folder=args.apply_plan.parent
            source=load_json(folder/'source.json.gz')
            before=load_json(folder/'database_before.json.gz')
        else:
            run_id=f'{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}'
            folder=args.output_dir or season_root/'runs'/run_id
            folder.mkdir(parents=True,exist_ok=False)
            before=snapshot_database(season)
            if source is None:
                client=NBAClient(folder/'nba_sources',timeout=args.timeout,retries=args.retries)
                source=collect_core(client,season,through,scoreboard_days=args.scoreboard_days)
                source=collect_ambiguous_box_scores(client,source)
                source=collect_missing_profiles(client,source,[int(r['nba_api_id']) for r in before['players'] if r['nba_api_id'] is not None])
                write_json(folder/'requests.json',client.requests)
            if source['season']!=season or source['through']!=through.isoformat():
                parser.error('Replay source season/cutoff differs from requested filters')
            write_json(folder/'source.json.gz',source,compressed=True)
            write_json(folder/'database_before.json.gz',before,compressed=True)
            plan=build_plan(source,before)
            write_json(folder/'plan.json',plan)
        print(json.dumps({'season':season,'through':plan['through'],'changes':plan['changes'],'summary':plan['summary'],'plan':str(folder/'plan.json')},indent=2),flush=True)
        if args.apply or args.apply_plan:
            result=apply_plan(plan,source,before)
            write_json(folder/'result.json',result)
            if result['summary']['games'] and args.profiles!='none':
                result['research_snapshots']=collect_profiles(season_root/'profiles',season,through,args.profiles,args.timeout)
                write_json(folder/'result.json',result)
            temporary=season_root/'last_success.tmp'
            write_json(temporary,{**result,'run_directory':str(folder)})
            temporary.replace(season_root/'last_success.json')
            print('Committed and independently verified. '+str(folder/'result.json'),flush=True)
        else:
            print('Preview only: database unchanged. Apply with --apply-plan '+str(folder/'plan.json'),flush=True)


if __name__=='__main__':
    main()
