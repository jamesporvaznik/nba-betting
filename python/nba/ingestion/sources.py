"""Free official NBA stats requests with retries and timestamped raw captures."""

from collections import Counter
from datetime import date, datetime, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

from nba_api.stats.endpoints import leaguegamelog, playergamelogs, scoreboardv3, commonallplayers, commonplayerinfo, boxscoretraditionalv3

EASTERN = ZoneInfo('America/New_York')


def current_season(today=None):
    today = today or datetime.now(EASTERN).date()
    year = today.year if today.month >= 7 else today.year-1
    return f'{year}-{(year+1)%100:02d}'


def result_rows(payload, name=None):
    sets = payload.get('resultSets', payload.get('resultSet'))
    if isinstance(sets, dict):
        sets = [sets]
    if not isinstance(sets, list):
        raise ValueError('NBA response has no supported result sets')
    selected = next((r for r in sets if name is None or r.get('name') == name), None)
    if selected is None or not isinstance(selected.get('headers'), list):
        raise ValueError(f'NBA response is missing {name or "result headers"}')
    if any(len(row) != len(selected['headers']) for row in selected['rowSet']):
        raise ValueError('NBA row/header lengths differ')
    return [dict(zip(selected['headers'], row)) for row in selected['rowSet']]


class NBAClient:
    def __init__(self, directory, timeout=25, retries=3, delay=.8):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.timeout, self.retries, self.delay = timeout, retries, delay
        self.last_request = 0.
        self.requests = []

    def fetch(self, endpoint_class, label, **parameters):
        for attempt in range(self.retries):
            time.sleep(max(0., self.delay-(time.monotonic()-self.last_request)))
            try:
                print(f'NBA {label}: request {attempt+1}/{self.retries}', flush=True)
                self.last_request = time.monotonic()
                endpoint = endpoint_class(timeout=self.timeout, **parameters)
                payload = endpoint.get_dict()
                raw = json.dumps(payload, separators=(',', ':'), allow_nan=False).encode()
                capture = {'captured_at': datetime.now(timezone.utc).isoformat(),
                           'endpoint': endpoint.endpoint, 'parameters': endpoint.parameters,
                           'payload_sha256': hashlib.sha256(raw).hexdigest(), 'payload': payload}
                path = self.directory/f'{label}.json.gz'
                with gzip.open(path, 'wt') as handle:
                    json.dump(capture, handle, separators=(',', ':'), allow_nan=False)
                self.requests.append({k:v for k,v in capture.items() if k != 'payload'})
                return payload
            except Exception as exc:
                if attempt+1 == self.retries:
                    raise RuntimeError(f'NBA {label} failed after {self.retries} attempts ({type(exc).__name__})') from exc
                print(f'NBA {label}: retrying after {type(exc).__name__}', flush=True)
                time.sleep(2**attempt)


def collect_core(client, season, through, scoreboard_days=7):
    end = through.strftime('%m/%d/%Y')
    teams = result_rows(client.fetch(leaguegamelog.LeagueGameLog, 'team_games',
        season=season, player_or_team_abbreviation='T', season_type_all_star='Regular Season', date_to_nullable=end))
    players = result_rows(client.fetch(playergamelogs.PlayerGameLogs, 'player_games',
        season_nullable=season, season_type_nullable='Regular Season', date_to_nullable=end))
    roster = result_rows(client.fetch(commonallplayers.CommonAllPlayers, 'current_players',
        season=season, is_only_current_season=1, league_id='00'))
    dates = sorted({str(r['GAME_DATE'])[:10] for r in teams if r.get('WL') in ('W','L') and str(r['GAME_ID']).startswith('002')})
    first = date.fromisoformat(dates[0]) if dates else through
    start = max(first, through-timedelta(days=scoreboard_days-1))
    scoreboards = {}
    day = start
    while day <= through:
        payload = client.fetch(scoreboardv3.ScoreboardV3, f'scoreboard_{day.isoformat()}', game_date=day.isoformat(), league_id='00')
        board = payload['scoreboard']
        if board['gameDate'] != day.isoformat():
            raise ValueError('Scoreboard returned a different date')
        scoreboards[day.isoformat()] = [dict(GAME_ID=r['gameId'], GAME_STATUS_ID=r['gameStatus']) for r in board['games']]
        day += timedelta(days=1)
    return {'season':season, 'through':through.isoformat(), 'captured_at':datetime.now(timezone.utc).isoformat(),
            'team_games':teams, 'player_games':players, 'roster':roster, 'scoreboards':scoreboards, 'profiles':{}}


def collect_missing_profiles(client, source, existing_ids):
    from nba.ingestion.transform import effective_player_rows
    missing = sorted({int(r['PLAYER_ID']) for r in effective_player_rows(source)} - set(existing_ids))
    for pid in missing:
        payload = client.fetch(commonplayerinfo.CommonPlayerInfo, f'player_{pid}', player_id=pid, league_id_nullable='00')
        records = result_rows(payload, 'CommonPlayerInfo')
        if len(records) != 1 or int(records[0]['PERSON_ID']) != pid:
            raise ValueError(f'Unexpected profile coverage for player {pid}')
        source['profiles'][str(pid)] = records[0]
    return source


def collect_ambiguous_box_scores(client, source):
    counts = Counter((int(r['PLAYER_ID']), str(r['GAME_DATE'])[:10]) for r in source['player_games'])
    games = sorted({str(r['GAME_ID']) for r in source['player_games']
                    if counts[int(r['PLAYER_ID']), str(r['GAME_DATE'])[:10]] > 1})
    source['box_scores'] = {}
    for game in games:
        source['box_scores'][game] = client.fetch(boxscoretraditionalv3.BoxScoreTraditionalV3,
                                                  f'box_score_{game}', game_id=game)
    return source
