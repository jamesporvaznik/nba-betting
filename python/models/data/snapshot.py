"""Consistent read-only snapshots for the selected training pipeline."""

from datetime import datetime, timezone
from decimal import Decimal
import json

from psycopg2.extras import RealDictCursor
from nba.repair_shot_locations import connection

QUERIES = {
    'players': 'SELECT playerid,gameid,season,date,team,opponent,homeaway,minutes,points,rebounds,assists,steals,blocks,turnovers,fga,fta,"3PA" FROM public.player_gamelog WHERE season=ANY(%s) ORDER BY playerid,date',
    'teams': 'SELECT season,date,team,opponent_abbrev,team_pace,def_rating,opponent_fgm,opponent_fga,opponent_3pm,opponent_3pa,opponent_fta,opponent_pts,opponent_trb,opponent_ast,team_orb,team_trb,opponent_orb FROM public.team_defensive_stats WHERE season=ANY(%s) ORDER BY season,team,date',
    'absences': 'SELECT player_id,season,date,team,opponent,stint_number,stint_first_game_date,stint_last_game_date FROM public.player_game_absences WHERE season=ANY(%s) ORDER BY season,player_id,date',
    'positions': 'SELECT nba_api_id,position FROM public.players ORDER BY nba_api_id',
}


def capture_training_snapshot(seasons):
    snapshot = {'captured_at': datetime.now(timezone.utc).isoformat(), 'database_writes': False}
    with connection() as conn:
        conn.set_session(readonly=True, isolation_level='REPEATABLE READ')
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            for name, query in QUERIES.items():
                cursor.execute(query, (list(seasons),) if name != 'positions' else None)
                snapshot[name] = [dict(row) for row in cursor.fetchall()]
        conn.rollback()
    return json.loads(json.dumps(snapshot, default=lambda v: float(v) if isinstance(v, Decimal) else str(v)))


def restrict_snapshot(snapshot, seasons):
    result = dict(snapshot)
    for key in ['players', 'teams', 'absences']:
        result[key] = [row for row in snapshot[key] if row['season'] in seasons]
    if {row['season'] for row in result['players']} != set(seasons):
        raise ValueError(f'Missing requested game-log seasons: {seasons}')
    return result
