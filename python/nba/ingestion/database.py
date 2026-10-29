"""Prepare source-checked season-scoped upserts and verify them transactionally."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json

from psycopg2 import sql
from psycopg2.extras import RealDictCursor, execute_values

from nba.repair_shot_locations import connection
from nba.repair_gamelogs import GAME_COLUMNS, STINT_COLUMNS, player_record
from nba.build_player_absences import COLUMNS as ABSENCE_COLUMNS
from nba.ingestion.transform import TEAM_COLUMNS, transform, effective_player_rows

PLAYER_COLUMNS = ('nba_api_id','first_name','last_name','display_name','birth_date',
                  'height_inches','weight_lbs','draft_year','position','team_id','is_active')
TABLES = {
    'players': (PLAYER_COLUMNS, ('nba_api_id',)),
    'player_gamelog': (GAME_COLUMNS, ('playerid','gameid')),
    'team_defensive_stats': (TEAM_COLUMNS, ('team','date')),
    'player_team_stints': (STINT_COLUMNS, ('player_id','season','stint_number')),
    'player_game_absences': (tuple(ABSENCE_COLUMNS), ('player_id','season','date')),
}


def serial(value):
    return json.loads(json.dumps(value,default=str))


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str,separators=(',',':')).encode()).hexdigest()


def read_database(cursor, season):
    cursor.execute('SELECT * FROM public.players ORDER BY nba_api_id')
    players=serial([dict(r) for r in cursor.fetchall()])
    cursor.execute('SELECT * FROM public.teams ORDER BY nba_api_id')
    teams=serial([dict(r) for r in cursor.fetchall()])
    existing={}
    for table in list(TABLES)[1:]:
        cursor.execute(sql.SQL('SELECT * FROM public.{} WHERE season=%s ORDER BY {}').format(
            sql.Identifier(table),sql.SQL(',').join(map(sql.Identifier,TABLES[table][1]))),(season,))
        existing[table]=serial([dict(r) for r in cursor.fetchall()])
    cursor.execute("SELECT table_name,column_name,numeric_scale FROM information_schema.columns WHERE table_schema='public' AND data_type='numeric' AND table_name=ANY(%s) ORDER BY table_name,column_name",(list(TABLES),))
    scales={table:{} for table in TABLES}
    for row in cursor.fetchall():
        scales[row['table_name']][row['column_name']]=row['numeric_scale']
    historical={}
    for table in list(TABLES)[1:]:
        cursor.execute(sql.SQL("SELECT count(*) AS rows,md5(COALESCE(string_agg(md5(to_jsonb(r)::text),'' ORDER BY {}),'')) AS sha FROM public.{} r WHERE season<>%s").format(
            sql.SQL(',').join(map(sql.Identifier,TABLES[table][1])),sql.Identifier(table)),(season,))
        historical[table]=dict(cursor.fetchone())
    return {'players':players,'teams':teams,'existing':existing,'scales':scales,'historical':historical}


def snapshot_database(season):
    with connection() as conn:
        conn.set_session(readonly=True,isolation_level='REPEATABLE READ')
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            snapshot=read_database(cursor,season)
        conn.rollback()
    return snapshot


def key(row,table):
    return tuple(row[c].strip() if isinstance(row[c],str) else row[c] for c in TABLES[table][1])


def projected(row,table):
    return {c:row.get(c) for c in TABLES[table][0]}


def desired_players(source,database):
    known={int(p['nba_api_id']):p for p in database['players'] if p['nba_api_id'] is not None}
    teams={int(t['nba_api_id']):int(t['id']) for t in database['teams']}
    roster={}
    for row in source['roster']:
        pid=int(row['PERSON_ID'])
        if pid in roster:
            raise ValueError('Duplicate player in current roster source')
        roster[pid]=row
    records=[]
    pids=sorted({int(r['PLAYER_ID']) for r in effective_player_rows(source)})
    for pid in pids:
        if pid in known:
            row=projected(known[pid],'players')
        else:
            info=source['profiles'].get(str(pid))
            if info is None:
                raise ValueError(f'Missing verified master profile for new player {pid}')
            row=player_record(info,pid)
            row['team_id']=None
        current=roster.get(pid,source['profiles'].get(str(pid)))
        if current is not None:
            tid=int(current.get('TEAM_ID') or 0)
            if tid and tid not in teams:
                raise ValueError('Current player team is absent from teams master')
            row['team_id']=teams.get(tid)
            status=str(current.get('ROSTERSTATUS','')).lower()
            if status in ('1','active','0','inactive'):
                row['is_active']=status in ('1','active')
        records.append(projected(row,'players'))
    return records


def build_plan(source,database):
    desired=transform(source,database)
    desired['players']=desired_players(source,database)
    if database['existing']['player_gamelog'] and not desired['player_gamelog']:
        raise ValueError('Empty NBA source cannot replace a populated season')
    before={'players':database['players'],**database['existing']}
    changes,removals={},{}
    for table,(columns,keys) in TABLES.items():
        old={key(r,table):projected(r,table) for r in before[table]}
        if len(old)!=len(before[table]):
            raise ValueError(f'Duplicate stored keys in {table}')
        new={key(r,table):projected(r,table) for r in desired[table]}
        if len(new)!=len(desired[table]):
            raise ValueError(f'Duplicate desired keys in {table}')
        missing=set(old)-set(new)
        if table in ['player_gamelog','team_defensive_stats'] and missing:
            raise ValueError(f'Existing {table} games are outside the complete source; refusing to remove or shrink coverage')
        if table in ['player_team_stints','player_game_absences']:
            if table=='player_game_absences':
                for r in before[table]:
                    if r.get('source')!='gamelog_no_appearance_within_observed_stint' or r.get('pregame_confirmed'):
                        raise ValueError('Absence table contains non-derived observations; refusing reconciliation')
            removals[table]=[old[k] for k in sorted(missing)]
        changes[table]={'inserted':sum(k not in old for k in new),
                        'updated':sum(k in old and serial(new[k])!=serial(old[k]) for k in new),
                        'removed':len(removals.get(table,[]))}
    return {'season':source['season'],'through':source['through'],
            'prepared_at':datetime.now(timezone.utc).isoformat(),
            'before_sha256':digest(database),'source_sha256':digest(source),
            'desired':serial(desired),'changes':changes,'removals':removals,
            'desired_sha256':digest(desired),'historical':database.get('historical',{}),
            'summary':{'games':len(desired['team_defensive_stats'])//2,'player_appearances':len(desired['player_gamelog']),
                       'box_score_reconciled_games':sorted(source.get('box_scores',{})),
                       'team_games':len(desired['team_defensive_stats']),
                       'games_per_team':dict(sorted(Counter(r['team'] for r in desired['team_defensive_stats']).items()))}}


def upsert(cursor,table,rows):
    if not rows:
        return
    columns,keys=TABLES[table]
    values=[c for c in columns if c not in keys]
    assignments=[sql.SQL('{}=EXCLUDED.{}').format(sql.Identifier(c),sql.Identifier(c)) for c in values]
    if table=='team_defensive_stats':
        assignments.append(sql.SQL('updated_at=now()'))
    query=sql.SQL('INSERT INTO public.{} AS existing ({}) VALUES %s ON CONFLICT ({}) DO UPDATE SET {} WHERE ROW({}) IS DISTINCT FROM ROW({})').format(
        sql.Identifier(table),sql.SQL(',').join(map(sql.Identifier,columns)),
        sql.SQL(',').join(map(sql.Identifier,keys)),sql.SQL(',').join(assignments),
        sql.SQL(',').join(sql.SQL('existing.{}').format(sql.Identifier(c)) for c in values),
        sql.SQL(',').join(sql.SQL('EXCLUDED.{}').format(sql.Identifier(c)) for c in values))
    execute_values(cursor,query.as_string(cursor),[tuple(r[c] for c in columns) for r in rows],page_size=500)


def verify(cursor,plan):
    actual=read_database(cursor,plan['season'])
    actual_tables={'players':actual['players'],**actual['existing']}
    if actual['historical']!=plan.get('historical',actual['historical']):
        raise ValueError('A completed historical season changed unexpectedly')
    for table,rows in plan['desired'].items():
        indexed={key(r,table):serial(projected(r,table)) for r in actual_tables[table]}
        for row in rows:
            if indexed.get(key(row,table))!=serial(projected(row,table)):
                raise ValueError(f'Stored {table} row differs from validated source: {key(row,table)}')
        if table!='players' and set(indexed)!={key(r,table) for r in rows}:
            raise ValueError(f'Stored {table} coverage differs after ingestion')
    return actual


def apply_plan(plan,source,before):
    if digest(source)!=plan['source_sha256'] or digest(before)!=plan['before_sha256'] or digest(plan['desired'])!=plan['desired_sha256']:
        raise ValueError('Prepared input checksum changed')
    rebuilt=build_plan(source,before)
    if any(rebuilt[field]!=plan[field] for field in ['desired','changes','removals','historical','summary']):
        raise ValueError('Prepared plan does not reproduce')
    with connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SET LOCAL lock_timeout='10s'")
            cursor.execute("SET LOCAL statement_timeout='120s'")
            cursor.execute("SELECT pg_advisory_xact_lock(hashtext(%s))",(f'nba_daily_ingestion_{plan["season"]}',))
            cursor.execute('LOCK TABLE public.players, public.teams, public.player_gamelog, public.team_defensive_stats, public.player_team_stints, public.player_game_absences IN SHARE ROW EXCLUSIVE MODE')
            if digest(read_database(cursor,plan['season']))!=plan['before_sha256']:
                raise ValueError('Database changed after preparation; rerun ingestion')
            previous={'players':before['players'],**before['existing']}
            for table in TABLES:
                old={key(r,table):serial(projected(r,table)) for r in previous[table]}
                changed=[r for r in plan['desired'][table] if old.get(key(r,table))!=serial(projected(r,table))]
                upsert(cursor,table,changed)
            for table,rows in plan['removals'].items():
                if not rows:
                    continue
                keys=TABLES[table][1]
                query=sql.SQL('DELETE FROM public.{} WHERE {}').format(sql.Identifier(table),
                    sql.SQL(' AND ').join(sql.SQL('{}=%s').format(sql.Identifier(c)) for c in keys))
                for row in rows:
                    cursor.execute(query,tuple(row[c] for c in keys))
            verify(cursor,plan)
        conn.commit()
    with connection() as conn:
        conn.set_session(readonly=True)
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            verify(cursor,plan)
        conn.rollback()
    return {'committed_at':datetime.now(timezone.utc).isoformat(),'season':plan['season'],
            'changes':plan['changes'],'summary':plan['summary'],'independent_verification_passed':True}
