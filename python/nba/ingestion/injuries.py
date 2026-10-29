"""Timestamped injury observations; only an exact Out status confirms absence."""

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import requests

from nba.ingestion.sources import EASTERN

SCHEMA = '''CREATE TABLE IF NOT EXISTS public.nba_injury_reports (
    report_at timestamptz PRIMARY KEY,
    fetched_at timestamptz NOT NULL,
    source_url text NOT NULL,
    pdf_sha256 text NOT NULL,
    observations jsonb NOT NULL CHECK (jsonb_typeof(observations) = 'array'),
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (report_at <= fetched_at)
);
ALTER TABLE public.nba_injury_reports ENABLE ROW LEVEL SECURITY;
COMMENT ON TABLE public.nba_injury_reports IS
'Immutable timestamped official injury-report snapshots. Only Out is a confirmed absence. Other statuses and unsubmitted reports remain observations. Historical gamelog absences are separate.';
'''
STATUSES = {'out':'Out', 'questionable':'Questionable', 'doubtful':'Doubtful',
            'probable':'Probable', 'available':'Available'}


def normalized_name(value):
    value = str(value or '')
    # NBA master names can be first-last with a comma before a suffix.
    value = re.sub(r',\s*(Jr\.?|Sr\.?|II|III|IV)\s*$', r' \1', value, flags=re.I)
    if ',' in value:
        last, first = value.rsplit(',', 1)
        value = first+' '+last
    value = unicodedata.normalize('NFKD', value)
    return re.sub('[^a-z0-9]', '', value.encode('ascii','ignore').decode().lower())


def normalize_observations(rows, players, teams):
    names = defaultdict(set)
    for p in players:
        if p.get('nba_api_id') is None:
            continue
        for name in [p.get('display_name'), str(p.get('first_name') or '')+' '+str(p.get('last_name') or '')]:
            if normalized_name(name):
                names[normalized_name(name)].add(int(p['nba_api_id']))
    team_names = {normalized_name(t['full_name']):t['abbreviation'].strip() for t in teams}
    if 'LAC' in team_names.values():
        team_names[normalized_name('LA Clippers')] = 'LAC'
    result, seen = [], set()
    for raw in rows:
        day = datetime.strptime(raw['Game Date'], '%m/%d/%Y').date().isoformat()
        team = team_names.get(normalized_name(raw['Team']))
        matchup = str(raw['Matchup']).split('@')
        if team is None or len(matchup)!=2 or team not in matchup:
            raise ValueError('Unrecognized injury-report team/matchup')
        opponent = next(t for t in matchup if t!=team)
        name = raw.get('Player Name')
        reason = raw.get('Reason')
        unsubmitted = str(reason or '').strip().casefold()=='not yet submitted'
        status = STATUSES.get(str(raw.get('Current Status') or '').strip().casefold())
        if not unsubmitted and (not name or status is None):
            raise ValueError('Unrecognized injury-report player/status; refusing partial snapshot')
        candidates = names.get(normalized_name(name), set()) if name else set()
        pid = next(iter(candidates)) if len(candidates)==1 else None
        key = day, team, normalized_name(name)
        if key in seen:
            raise ValueError('Duplicate injury observation in one report')
        seen.add(key)
        result.append(dict(date=day, game_time_raw=raw['Game Time'], team=team,
                           opponent=opponent, player_name=name, player_id=pid,
                           status=status, report_submitted=not unsubmitted,
                           confirmed_out=status=='Out' and not unsubmitted, reason=reason))
    if not result:
        raise ValueError('Empty injury report')
    return result


def collect_report(directory, players, teams, report_at=None, now=None, lookback_hours=6, timeout=20):
    from nbainjuries import injury, _constants
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    now = (now or datetime.now(timezone.utc)).astimezone(EASTERN)
    if report_at is not None:
        stamp = report_at.astimezone(EASTERN)
        if stamp > now:
            raise ValueError('Cannot fetch a future injury report')
        candidates = [stamp]
    else:
        stamp = now.replace(minute=now.minute//15*15, second=0, microsecond=0)
        candidates = [stamp-timedelta(minutes=15*i) for i in range(lookback_hours*4+1)]
    attempts = []
    for stamp in candidates:
        naive = stamp.replace(tzinfo=None)
        url = injury.gen_url(naive)
        response = requests.get(url, timeout=timeout, headers=_constants.requestheaders)
        attempts.append(dict(url=url,status=response.status_code))
        (directory/'requests.json').write_text(json.dumps(attempts,indent=2))
        # The public NBA CDN returns either 403 or 404 for unpublished slots.
        # A prior successful PDF is required; errors never establish health.
        if response.status_code in (403,404) and report_at is None:
            continue
        response.raise_for_status()
        if not response.content.startswith(b'%PDF'):
            raise ValueError('Injury report did not return a PDF')
        path = directory/url.rsplit('/',1)[1]
        path.write_bytes(response.content)
        # Parse the downloaded file: one HTTP fetch, with an explicit timeout.
        raw = json.loads(injury.get_reportdata(naive, local=True, localdir=str(directory)))
        fetched_at = datetime.now(timezone.utc)
        observations = normalize_observations(raw, players, teams)
        snapshot = dict(report_at=stamp.astimezone(timezone.utc).isoformat(), fetched_at=fetched_at.isoformat(),
                        source_url=url, pdf_sha256=hashlib.sha256(response.content).hexdigest(),
                        observations=observations)
        (directory/'parsed_source.json').write_text(json.dumps(raw,indent=2))
        (directory/'snapshot.json').write_text(json.dumps(snapshot,indent=2))
        return snapshot
    raise RuntimeError(f'No published injury report in the last {lookback_hours} hours; availability is unknown')


def confirmed_absences(snapshot, game_date, team, as_of, game_start, max_age_minutes=60):
    """Pregame adapter. Never infer availability from absent/stale reports.

    Caller chooses the latest stored report fetched/published by as_of, and
    supplies the scheduled tip-off rather than guessing ambiguous PDF times.
    """
    if as_of.tzinfo is None or game_start.tzinfo is None:
        raise ValueError('Prediction and game-start times require timezones')
    published, fetched = (datetime.fromisoformat(snapshot[k]) for k in ('report_at','fetched_at'))
    if as_of>=game_start:
        raise ValueError('Pregame absences cannot be requested after tip-off')
    if published>as_of or fetched>as_of or as_of-published>timedelta(minutes=max_age_minutes):
        return dict(coverage_known=False, reason='report unavailable or stale at prediction time', rows=[])
    rows = [r for r in snapshot['observations'] if r['date']==game_date and r['team']==team]
    if not rows or any(not r['report_submitted'] for r in rows):
        return dict(coverage_known=False, reason='team report unavailable', rows=[])
    out = [r for r in rows if r['status']=='Out']
    unmatched = [r['player_name'] for r in out if r['player_id'] is None]
    if unmatched:
        return dict(coverage_known=False, reason='unmatched Out players', unmatched=unmatched, rows=[])
    return dict(coverage_known=True, reason=None, rows=out)


def store_report(snapshot):
    from psycopg2.extras import Json, RealDictCursor
    from nba.repair_shot_locations import connection
    with connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext('nba_injury_reports'))")
            cur.execute(SCHEMA)
            cur.execute('SELECT * FROM public.nba_injury_reports WHERE report_at=%s', (snapshot['report_at'],))
            old = cur.fetchone()
            inserted = old is None
            if old and old['pdf_sha256']!=snapshot['pdf_sha256']:
                raise ValueError('Published report changed under the same timestamp; preserve and review the new local capture')
            resolved = 0
            if inserted:
                cur.execute('INSERT INTO public.nba_injury_reports (report_at,fetched_at,source_url,pdf_sha256,observations) VALUES (%s,%s,%s,%s,%s)',
                            (snapshot['report_at'],snapshot['fetched_at'],snapshot['source_url'],snapshot['pdf_sha256'],Json(snapshot['observations'])))
                expected = snapshot['observations']
            else:
                expected,resolved = enrich_player_ids(old['observations'],snapshot['observations'])
                if resolved:
                    cur.execute('UPDATE public.nba_injury_reports SET observations=%s WHERE report_at=%s',
                                (Json(expected),snapshot['report_at']))
            # Publication, collection time and source statuses stay immutable.
        conn.commit()
    with connection() as conn:
        conn.set_session(readonly=True)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT * FROM public.nba_injury_reports WHERE report_at=%s', (snapshot['report_at'],))
            stored = cur.fetchone()
            if not stored or stored['pdf_sha256']!=snapshot['pdf_sha256'] or stored['observations']!=expected:
                raise ValueError('Independent injury-report verification failed')
        conn.rollback()
    return dict(inserted=inserted, player_ids_resolved=resolved, report_at=snapshot['report_at'], independently_verified=True)


def enrich_player_ids(old, new):
    """Fill unresolved identity metadata without revising a published status."""
    def key(row):
        return row['date'],row['team'],row['player_name']
    incoming={key(r):r for r in new}
    if len(incoming)!=len(new) or set(incoming)!={key(r) for r in old}:
        raise ValueError('Parsed report coverage changed under the same timestamp')
    result,filled=[],0
    for row in old:
        other=incoming[key(row)]
        if {k:v for k,v in row.items() if k!='player_id'}!={k:v for k,v in other.items() if k!='player_id'}:
            raise ValueError('Published observation changed under the same timestamp')
        if row['player_id'] is not None and other['player_id'] not in (None,row['player_id']):
            raise ValueError('Existing player identity changed')
        identity=row['player_id'] if row['player_id'] is not None else other['player_id']
        filled+=row['player_id'] is None and identity is not None
        result.append({**row,'player_id':identity})
    return result,filled


def load_report(as_of):
    """Latest snapshot that had actually been collected by prediction time."""
    from psycopg2.extras import RealDictCursor
    from nba.repair_shot_locations import connection
    if as_of.tzinfo is None:
        raise ValueError('Prediction time requires a timezone')
    with connection() as conn:
        conn.set_session(readonly=True)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT report_at,fetched_at,source_url,pdf_sha256,observations FROM public.nba_injury_reports WHERE report_at<=%s AND fetched_at<=%s ORDER BY report_at DESC LIMIT 1', (as_of,as_of))
            row=cur.fetchone()
        conn.rollback()
    if row is None:
        return None
    return {**dict(row), 'report_at':row['report_at'].isoformat(), 'fetched_at':row['fetched_at'].isoformat()}
