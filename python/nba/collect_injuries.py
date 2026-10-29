"""Capture the latest official injury report with nbainjuries; --apply stores it."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from psycopg2.extras import RealDictCursor
from nba.repair_shot_locations import connection
from nba.ingestion.injuries import collect_report, store_report
from nba.ingestion.sources import NBAClient, current_season, result_rows
from nba_api.stats.endpoints import commonallplayers

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--report-at',type=datetime.fromisoformat,help='Explicit timezone-aware report timestamp')
    parser.add_argument('--data-root',type=Path,default=ROOT/'python/data/raw/ingestion/injuries')
    args=parser.parse_args()
    if args.report_at and args.report_at.tzinfo is None:
        parser.error('--report-at must include a timezone, e.g. 2026-10-28T19:30:00-04:00')
    args.data_root.mkdir(parents=True,exist_ok=True)
    with (args.data_root/'collection.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('Injury collector already active; skipping duplicate invocation.')
            return
        folder=args.data_root/f'{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}'
        with connection() as conn:
            conn.set_session(readonly=True)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute('SELECT nba_api_id,display_name,first_name,last_name FROM public.players')
                players=[dict(r) for r in cur.fetchall()]
                cur.execute('SELECT abbreviation,full_name FROM public.teams')
                teams=[dict(r) for r in cur.fetchall()]
            conn.rollback()
        # Include rookies who have not appeared yet, without inventing master
        # rows or relying on fuzzy player-name matches.
        client=NBAClient(folder/'nba_sources')
        roster=result_rows(client.fetch(commonallplayers.CommonAllPlayers,'current_players',
                           season=current_season(),is_only_current_season=1,league_id='00'))
        players.extend(dict(nba_api_id=r['PERSON_ID'],display_name=r['DISPLAY_FIRST_LAST']) for r in roster)
        snapshot=collect_report(folder,players,teams,args.report_at)
        summary=dict(report_at=snapshot['report_at'], fetched_at=snapshot['fetched_at'],
                     report_age_minutes=round((datetime.fromisoformat(snapshot['fetched_at'])-datetime.fromisoformat(snapshot['report_at'])).total_seconds()/60,1),
                     statuses=dict(Counter(r['status'] or 'Not submitted' for r in snapshot['observations'])),
                     unmatched_out=[r['player_name'] for r in snapshot['observations'] if r['confirmed_out'] and r['player_id'] is None],
                     directory=str(folder), database_writes=False)
        if args.apply:
            summary.update(store_report(snapshot),database_writes=True)
        (folder/'result.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
