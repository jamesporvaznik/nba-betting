"""Prepare or create historical no-appearance rows within observed team stints."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from psycopg2.extras import RealDictCursor, execute_values
from nba.repair_shot_locations import connection

SEASONS = ['2022-23', '2023-24', '2024-25', '2025-26']
COLUMNS = ['player_id', 'season', 'date', 'team_id', 'team', 'opponent',
           'game_id', 'stint_number', 'stint_first_game_date', 'stint_last_game_date']
QUERIES = {
    'games': 'SELECT playerid,season,date,team FROM public.player_gamelog WHERE season=ANY(%s) ORDER BY season,playerid,date,team',
    'stints': 'SELECT player_id,team_id,season,stint_number,first_game_date,last_game_date FROM public.player_team_stints WHERE season=ANY(%s) ORDER BY season,player_id,stint_number',
    'calendar': 'SELECT season,date,team,opponent_abbrev,game_id FROM public.team_defensive_stats WHERE season=ANY(%s) ORDER BY season,team,date',
    'teams': 'SELECT nba_api_id,abbreviation FROM public.teams ORDER BY nba_api_id',
}


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def capture(cursor):
    snapshot = {}
    for key, sql in QUERIES.items():
        cursor.execute(sql, (SEASONS,) if key != 'teams' else None)
        snapshot[key] = [dict(row) for row in cursor.fetchall()]
    return json.loads(json.dumps(snapshot, default=str))


def build(snapshot):
    teams = {int(r['nba_api_id']): r['abbreviation'].strip() for r in snapshot['teams']}
    if len(set(teams.values())) != len(teams):
        raise ValueError('Ambiguous team abbreviations')
    calendars, calendar_keys = defaultdict(list), set()
    for row in snapshot['calendar']:
        key = row['season'], row['team'].strip(), row['date']
        if key in calendar_keys:
            raise ValueError('Duplicate team game date')
        calendar_keys.add(key)
        calendars[key[:2]].append(row)
    appearances, player_dates = set(), set()
    for row in snapshot['games']:
        key = row['season'], row['team'].strip(), row['date']
        if key not in calendar_keys:
            raise ValueError('Player appearance has no team calendar game')
        pkey = row['playerid'], row['season'], row['date']
        if pkey in player_dates:
            raise ValueError('Duplicate player appearance date')
        player_dates.add(pkey)
        appearances.add((row['playerid'], *key))
    rows, used, covered = [], set(), set()
    observed_team_dates = {key[1:] for key in appearances}
    for stint in snapshot['stints']:
        pid, season = stint['player_id'], stint['season']
        team = teams[int(stint['team_id'])]
        first, last = stint['first_game_date'], stint['last_game_date']
        if first > last:
            raise ValueError('Invalid stint date range')
        for date in [first, last]:
            if (pid, season, team, date) not in appearances:
                raise ValueError('Stint boundary is not an observed appearance')
        for game in calendars[(season, team)]:
            date = game['date']
            if not first <= date <= last:
                continue
            key = pid, season, team, date
            if key in covered:
                raise ValueError('Overlapping player stints')
            covered.add(key)
            if key in appearances:
                continue
            # A team-calendar row without any player logs is unknown coverage,
            # not evidence that every member of its roster sat out.
            if (season, team, date) not in observed_team_dates:
                continue
            # Never infer an absence if the player appeared for another team.
            if (pid, season, date) in player_dates:
                raise ValueError('Stint overlaps an appearance for another team')
            if (pid, season, date) in used:
                raise ValueError('Ambiguous absent player team')
            used.add((pid, season, date))
            rows.append(dict(zip(COLUMNS, [pid, season, date, int(stint['team_id']), team,
                                          game['opponent_abbrev'].strip(), game['game_id'],
                                          stint['stint_number'], first, last])))
    if not appearances.issubset(covered):
        raise ValueError('Some appearances are outside the supplied team stints')
    return sorted(rows, key=lambda r: (r['season'], r['player_id'], r['date'], r['team']))


CREATE_SQL = '''CREATE TABLE public.player_game_absences (
    player_id bigint NOT NULL,
    season text NOT NULL,
    date date NOT NULL,
    team_id bigint NOT NULL,
    team text NOT NULL,
    opponent text NOT NULL,
    game_id text,
    stint_number bigint NOT NULL,
    stint_first_game_date date NOT NULL,
    stint_last_game_date date NOT NULL,
    source text NOT NULL DEFAULT 'gamelog_no_appearance_within_observed_stint',
    pregame_confirmed boolean NOT NULL DEFAULT false CHECK (NOT pregame_confirmed),
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (player_id, season, date),
    CHECK (date > stint_first_game_date AND date < stint_last_game_date)
);
CREATE INDEX player_game_absences_team_date_idx ON public.player_game_absences (season, team, date);
ALTER TABLE public.player_game_absences ENABLE ROW LEVEL SECURITY;
COMMENT ON TABLE public.player_game_absences IS
'Historical no-appearance observations between first and last recorded appearances within a team stint. Reasons unknown. Excludes stint boundaries/transition gaps. Missing row does not establish availability. Not timestamped pregame reports.';
'''


def prepare(folder):
    with connection() as conn:
        conn.set_session(readonly=True, isolation_level='REPEATABLE READ')
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT to_regclass('public.player_game_absences') AS existing")
            if cursor.fetchone()['existing']:
                raise ValueError('Target already exists; refusing to replace it')
            snapshot = capture(cursor)
        conn.rollback()
    rows = build(snapshot)
    plan = {'prepared_at': datetime.now(timezone.utc).isoformat(), 'source_sha256': digest(snapshot),
            'rows_sha256': digest(rows), 'rows': rows,
            'counts_by_season': dict(sorted(Counter(r['season'] for r in rows).items())),
            'coverage': 'Interior gaps only within observed first/last appearance stints. No transition/boundary extrapolation.',
            'source': 'live_database', 'database_writes': False}
    folder.mkdir(parents=True, exist_ok=False)
    (folder/'source_snapshot.json').write_text(json.dumps(snapshot))
    (folder/'plan.json').write_text(json.dumps(plan, indent=2))
    (folder/'schema.sql').write_text(CREATE_SQL)
    print(json.dumps({k: v for k, v in plan.items() if k != 'rows'}, indent=2))


def apply(folder):
    plan = json.loads((folder/'plan.json').read_text())
    if digest(plan['rows']) != plan['rows_sha256']:
        raise ValueError('Prepared rows changed')
    with connection() as conn:
        conn.set_session(isolation_level='REPEATABLE READ')
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SET LOCAL lock_timeout='10s'")
            cursor.execute('LOCK TABLE public.player_gamelog, public.player_team_stints, public.team_defensive_stats, public.teams IN SHARE MODE')
            snapshot = capture(cursor)
            if digest(snapshot) != plan['source_sha256'] or build(snapshot) != plan['rows']:
                raise ValueError('Source data changed since preparation')
            cursor.execute(CREATE_SQL)
            if plan['rows']:
                execute_values(cursor, f"INSERT INTO public.player_game_absences ({','.join(COLUMNS)}) VALUES %s",
                               [tuple(row[c] for c in COLUMNS) for row in plan['rows']], page_size=1000)
            verify(cursor, plan)
        conn.commit()
    with connection() as conn:
        conn.set_session(readonly=True)
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            verify(cursor, plan)
        conn.rollback()
    result = {'verified_at': datetime.now(timezone.utc).isoformat(), 'database_writes': True,
              'table': 'public.player_game_absences', 'rows': len(plan['rows']),
              'counts_by_season': plan['counts_by_season']}
    (folder/'apply_result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def verify(cursor, plan):
    cursor.execute(f"SELECT {','.join(COLUMNS)} FROM public.player_game_absences ORDER BY season,player_id,date,team")
    actual = json.loads(json.dumps([dict(row) for row in cursor.fetchall()], default=str))
    if actual != plan['rows']:
        raise ValueError('Stored absences differ from prepared rows')


def repair_unobserved_games(folder):
    """Remove only inferred absences on dates with zero team appearances."""
    folder.mkdir(parents=True, exist_ok=False)
    with connection() as conn:
        conn.set_session(isolation_level='REPEATABLE READ')
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SET LOCAL lock_timeout='10s'")
            cursor.execute('LOCK TABLE public.player_gamelog, public.player_team_stints, public.team_defensive_stats, public.teams IN SHARE MODE')
            cursor.execute('LOCK TABLE public.player_game_absences IN SHARE ROW EXCLUSIVE MODE')
            snapshot = capture(cursor)
            desired = build(snapshot)
            cursor.execute('SELECT * FROM public.player_game_absences ORDER BY season,player_id,date,team')
            before = json.loads(json.dumps([dict(r) for r in cursor.fetchall()], default=str))
            observed = {(r['season'], r['team'].strip(), r['date']) for r in snapshot['games']}
            removed = [r for r in before if (r['season'], r['team'], r['date']) not in observed]
            remaining = [{c: r[c] for c in COLUMNS} for r in before if (r['season'], r['team'], r['date']) in observed]
            if remaining != desired or any(r['source'] != 'gamelog_no_appearance_within_observed_stint' for r in removed):
                raise ValueError('Unexpected table/source differences; no changes made')
            (folder/'before.json').write_text(json.dumps(before, indent=2))
            (folder/'removed.json').write_text(json.dumps(removed, indent=2))
            (folder/'source_snapshot.json').write_text(json.dumps(snapshot))
            if removed:
                execute_values(cursor, '''DELETE FROM public.player_game_absences a
                    USING (VALUES %s) AS d(player_id,season,date)
                    WHERE a.player_id=d.player_id AND a.season=d.season AND a.date=d.date::date''',
                               [(r['player_id'], r['season'], r['date']) for r in removed], page_size=len(removed))
                if cursor.rowcount != len(removed):
                    raise ValueError('Deletion coverage mismatch')
            verify(cursor, {'rows': desired})
        conn.commit()
    with connection() as conn:
        conn.set_session(readonly=True)
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            verify(cursor, {'rows': desired})
        conn.rollback()
    report = {'removed': len(removed), 'remaining': len(desired),
              'removed_by_season': dict(Counter(r['season'] for r in removed)),
              'counts_by_season': dict(Counter(r['season'] for r in desired)),
              'verified_at': datetime.now(timezone.utc).isoformat()}
    (folder/'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', type=Path)
    mode.add_argument('--apply', type=Path)
    mode.add_argument('--repair-unobserved-games', type=Path)
    args = parser.parse_args()
    if args.repair_unobserved_games:
        repair_unobserved_games(args.repair_unobserved_games)
    else:
        prepare(args.prepare) if args.prepare else apply(args.apply)


if __name__ == '__main__':
    main()
