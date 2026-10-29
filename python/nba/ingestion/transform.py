"""Validate complete regular-season games and preserve audited table conventions."""

from collections import defaultdict, Counter
from datetime import date
from decimal import Decimal, ROUND_HALF_UP

from nba.repair_gamelogs import game_record, build_stints, player_record, GAME_COLUMNS, STINT_COLUMNS
from nba.repair_pace import compute_pregame, FIELDS as PACE_BASELINES
from nba.audit_team_defense import OPPONENT_FIELDS, TEAM_FIELDS
from nba.build_player_absences import build as build_absences

TEAM_COLUMNS = ('season','team','opponent_abbrev','date','home',*TEAM_FIELDS,*OPPONENT_FIELDS,
                'pace','team_pace','opponent_pace','off_rating','def_rating',*PACE_BASELINES,'game_id')
TEAM_COLUMNS = tuple(dict.fromkeys(TEAM_COLUMNS))


def dec(value):
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError('Non-finite source number')
    return result


def rounded(value, scale):
    return str(dec(value).quantize(Decimal(1).scaleb(-scale), rounding=ROUND_HALF_UP))


def regular_rows(rows, season, through):
    year = season[:4]
    output = []
    for row in rows:
        game = str(row['GAME_ID'])
        if not game.startswith('002') or row.get('WL') not in ('W','L'):
            continue
        if len(game) != 10 or game[3:5] != year[-2:]:
            raise ValueError('NBA game ID belongs to a different season')
        day = str(row['GAME_DATE'])[:10]
        date.fromisoformat(day)
        if not f'{year}-07-01' <= day < f'{int(year)+1}-07-01':
            raise ValueError('NBA game date belongs to a different season')
        if day > through:
            raise ValueError('NBA returned a game after the requested cutoff')
        if row.get('SEASON_ID') is not None and str(row['SEASON_ID']) != '2'+year:
            raise ValueError('NBA returned a mismatched season ID')
        if row.get('SEASON_YEAR') is not None and str(row['SEASON_YEAR']) != season:
            raise ValueError('NBA returned a mismatched player season')
        output.append(row)
    return output


BOX_FIELDS = dict(FGM='fieldGoalsMade', FGA='fieldGoalsAttempted', FG_PCT='fieldGoalsPercentage',
    FG3M='threePointersMade', FG3A='threePointersAttempted', FG3_PCT='threePointersPercentage',
    FTM='freeThrowsMade', FTA='freeThrowsAttempted', FT_PCT='freeThrowsPercentage',
    OREB='reboundsOffensive', DREB='reboundsDefensive', REB='reboundsTotal', AST='assists',
    STL='steals', BLK='blocks', TOV='turnovers', PF='foulsPersonal', PTS='points', PLUS_MINUS='plusMinusPoints')


def effective_player_rows(source):
    season, through = source['season'], source['through']
    raw_players = regular_rows(source['player_games'], season, through)
    duplicates = Counter((int(r['PLAYER_ID']), str(r['GAME_DATE'])[:10]) for r in raw_players)
    ambiguous = {str(r['GAME_ID']) for r in raw_players if duplicates[int(r['PLAYER_ID']), str(r['GAME_DATE'])[:10]]>1}
    replacements = source.get('box_scores', {})
    if not ambiguous.issubset(replacements):
        raise ValueError('Ambiguous player date requires an independent game box score')
    result = [r for r in raw_players if str(r['GAME_ID']) not in replacements]
    raw_teams = regular_rows(source['team_games'], season, through)
    for game, payload in replacements.items():
        box = payload['boxScoreTraditional']
        if str(box['gameId']) != game:
            raise ValueError('Box score game ID differs from requested game')
        metadata = {int(r['TEAM_ID']):r for r in raw_teams if str(r['GAME_ID'])==game}
        if set(metadata)!={int(box[s]['teamId']) for s in ('homeTeam','awayTeam')}:
            raise ValueError('Box score teams differ from final team logs')
        for side in ('homeTeam','awayTeam'):
            team = box[side]
            for player in team['players']:
                stats = player['statistics']
                # Blank means DNP/inactive; explicit 0:00 is an appearance.
                if not stats['minutes']:
                    continue
                minutes, seconds = stats['minutes'].split(':')
                row = {**metadata[int(team['teamId'])], **{k:stats[v] for k,v in BOX_FIELDS.items()},
                       'PLAYER_ID':int(player['personId']), 'PLAYER_NAME':player['firstName']+' '+player['familyName'],
                       'MIN':float(dec(minutes)+dec(seconds)/60)}
                result.append(row)
    return result


def validate_sources(source, teams):
    season, through = source['season'], source['through']
    raw_teams = regular_rows(source['team_games'], season, through)
    raw_players = effective_player_rows(source)
    team_map = {int(r['nba_api_id']):r['abbreviation'].strip() for r in teams}
    indexed, by_game = {}, defaultdict(list)
    for row in raw_teams:
        tid, game = int(row['TEAM_ID']), str(row['GAME_ID'])
        if team_map.get(tid) != row['TEAM_ABBREVIATION'].strip():
            raise ValueError('NBA team ID/abbreviation disagrees with master table')
        key = game,tid
        if key in indexed:
            raise ValueError('Duplicate source team/game')
        indexed[key] = row
        by_game[game].append(row)
    for game, rows in by_game.items():
        if len(rows) != 2:
            raise ValueError(f'Incomplete team pairing: {game}')
        one,two = rows
        if (str(one['GAME_DATE'])[:10] != str(two['GAME_DATE'])[:10] or
                one['MATCHUP'].split()[-1] != two['TEAM_ABBREVIATION'] or
                two['MATCHUP'].split()[-1] != one['TEAM_ABBREVIATION'] or
                {one['WL'],two['WL']} != {'W','L'}):
            raise ValueError('Paired source teams do not describe the same final game')
        if int(one['PTS'])==int(two['PTS']) or (int(one['PTS']) > int(two['PTS'])) != (one['WL']=='W'):
            raise ValueError('Final score and result disagree')
    player_groups, appearance_keys = defaultdict(list), set()
    exact = ['FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','REB','AST','STL','BLK','PF','PTS']
    for row in raw_players:
        key = str(row['GAME_ID']), int(row['TEAM_ID'])
        own = indexed.get(key)
        if own is None or str(row['GAME_DATE'])[:10] != str(own['GAME_DATE'])[:10] or row['MATCHUP'] != own['MATCHUP'] or row['WL'] != own['WL']:
            raise ValueError('Player appearance has no matching final team game')
        pkey = int(row['PLAYER_ID']),str(row['GAME_DATE'])[:10]
        if pkey in appearance_keys:
            raise ValueError('Duplicate or ambiguous player appearance date')
        appearance_keys.add(pkey)
        for field in exact+['TOV']:
            if dec(row[field]) < 0 or dec(row[field]) != dec(row[field]).to_integral_value():
                raise ValueError(f'Invalid source count: {field}')
        if dec(row['FGM']) > dec(row['FGA']) or dec(row['FTM']) > dec(row['FTA']) or dec(row['FG3M']) > dec(row['FG3A']):
            raise ValueError('Source shooting makes exceed attempts')
        if dec(row['PTS']) != 2*dec(row['FGM'])+dec(row['FG3M'])+dec(row['FTM']):
            raise ValueError('Player scoring counts do not reconcile')
        player_groups[key].append(row)
    if set(player_groups) != set(indexed):
        raise ValueError('Team games and player-game coverage differ; refusing partial collection')
    for key, rows in player_groups.items():
        team = indexed[key]
        for field in exact:
            if sum(dec(r[field]) for r in rows) != dec(team[field]):
                raise ValueError(f'Player/team totals disagree: {key} {field}')
        if sum(dec(r['TOV']) for r in rows) > dec(team['TOV']):
            raise ValueError('Player turnovers exceed team turnovers')
        duration = dec(team['MIN'])/5 if dec(team['MIN']) >= 200 else dec(team['MIN'])
        if duration < 48 or (duration-48) % 5:
            raise ValueError('Invalid regulation/overtime duration')
        if abs(sum(dec(r['MIN']) for r in rows)-duration*5) > Decimal('.15'):
            raise ValueError('Player minutes do not reconcile with completed team game')
    for day, rows in source['scoreboards'].items():
        final_ids = {str(r['GAME_ID']) for r in rows if int(r['GAME_STATUS_ID'])==3 and str(r['GAME_ID']).startswith('002')}
        source_ids = {g for g, records in by_game.items() if str(records[0]['GAME_DATE'])[:10]==day}
        if final_ids != source_ids:
            raise ValueError(f'Scoreboard/game-log final-game coverage differs on {day}')
    return raw_teams,raw_players


def normalize(row, scales):
    return {key:rounded(value,scales[key]) if value is not None and key in scales else value for key,value in row.items()}


def transform(source, database):
    season = source['season']
    raw_teams, raw_players = validate_sources(source, database['teams'])
    team_ids = {r['abbreviation'].strip():int(r['nba_api_id']) for r in database['teams']}
    games=[]
    for row in raw_players:
        exact=dict(row)
        for pct,made,attempts in [('FG_PCT','FGM','FGA'),('FG3_PCT','FG3M','FG3A'),('FT_PCT','FTM','FTA')]:
            exact[pct] = float(dec(row[made])/dec(row[attempts])) if dec(row[attempts]) else 0.
        record=game_record(exact,season)
        games.append(normalize(record,database['scales']['player_gamelog']))
    team_index={(str(r['GAME_ID']),int(r['TEAM_ID'])):r for r in raw_teams}
    team_records=[]
    for i,row in enumerate(raw_teams,1):
        others=[r for (game,tid),r in team_index.items() if game==str(row['GAME_ID']) and tid!=int(row['TEAM_ID'])]
        opp=others[0]
        day=str(row['GAME_DATE'])[:10];abbr=row['TEAM_ABBREVIATION'].strip();oppabbr=opp['TEAM_ABBREVIATION'].strip()
        duration=dec(row['MIN'])/5 if dec(row['MIN'])>=200 else dec(row['MIN'])
        own=dec(row['FGA'])+Decimal('.44')*dec(row['FTA'])-dec(row['OREB'])+dec(row['TOV'])
        other=dec(opp['FGA'])+Decimal('.44')*dec(opp['FTA'])-dec(opp['OREB'])+dec(opp['TOV'])
        if own<=0 or other<=0:
            raise ValueError('Nonpositive estimated possessions')
        record={'id':i,'season':season,'team':abbr,'opponent_abbrev':oppabbr,'date':day,
                'home':int('vs.' in row['MATCHUP']), 'game_id':f'{day}_{abbr}_{oppabbr}',
                **{field:row[raw] for field,raw in TEAM_FIELDS.items()},
                **{field:opp[raw] for field,raw in OPPONENT_FIELDS.items()},
                'pace':48*(own+other)/2/duration,'team_pace':48*own/duration,'opponent_pace':48*other/duration,
                'off_rating':100*dec(row['PTS'])/own,'def_rating':100*dec(opp['PTS'])/other}
        for pct,made,attempts in [('opponent_fg_pct','opponent_fgm','opponent_fga'),('opponent_3p_pct','opponent_3pm','opponent_3pa'),('opponent_ft_pct','opponent_ftm','opponent_fta')]:
            record[pct]=dec(record[made])/dec(record[attempts]) if dec(record[attempts]) else Decimal(0)
        team_records.append(normalize(record,database['scales']['team_defensive_stats']))
    if team_records:
        team_records=compute_pregame(team_records,{f:database['scales']['team_defensive_stats'][f] for f in PACE_BASELINES})
    for row in team_records:
        del row['id']
    stints=build_stints(games,team_ids)
    absence_input={'games':games,'stints':stints,'calendar':team_records,'teams':database['teams']}
    absences=build_absences(absence_input) if games else []
    return {'player_gamelog':games,'team_defensive_stats':team_records,'player_team_stints':stints,'player_game_absences':absences}
