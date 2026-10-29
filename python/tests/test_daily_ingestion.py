import copy
import unittest

from nba.ingestion.database import build_plan, desired_players
from nba.ingestion.transform import transform, validate_sources, BOX_FIELDS
from nba.ingestion.sources import current_season, result_rows
from datetime import date


def fixture():
    source = dict(season='2026-27', through='2026-10-27', team_games=[],
                  player_games=[], roster=[], profiles={}, scoreboards={})
    database = dict(teams=[dict(id=10, nba_api_id=100, abbreviation='AAA'),
                           dict(id=20, nba_api_id=200, abbreviation='BBB')],
                    players=[], existing={t: [] for t in ('player_gamelog', 'team_defensive_stats',
                    'player_team_stints', 'player_game_absences')}, historical={},
                    scales={'player_gamelog': dict(minutes=1, fg_pct=4, **{'3P_PCT': 4}, ft_pct=4, gamescore=2),
                            'team_defensive_stats': {f: 1 for f in ('pace', 'team_pace', 'opponent_pace',
                            'off_rating', 'def_rating', 'opponent_avg_pace', 'opponent_avg_team_pace',
                            'pace_deviation', 'team_pace_deviation')}})
    from nba.repair_pace import FIELDS
    database['scales']['team_defensive_stats'].update({f: 1 for f in FIELDS})
    for f in ('opponent_fg_pct', 'opponent_3p_pct', 'opponent_ft_pct'):
        database['scales']['team_defensive_stats'][f] = 4
    for n, day in enumerate(('2026-10-20', '2026-10-22', '2026-10-24'), 1):
        game = f'002260000{n}'
        source['scoreboards'][day] = [dict(GAME_ID=game, GAME_STATUS_ID=3)]
        for team, opp, tid, offset, win in [('AAA','BBB',100,0,True), ('BBB','AAA',200,10,False)]:
            rows = []
            for p in range(1, 6):
                row = dict(PLAYER_ID=p+offset, PLAYER_NAME=f'Player {p+offset}', TEAM_ID=tid,
                           TEAM_ABBREVIATION=team, MATCHUP=f'{team} '+('vs.' if win else '@')+f' {opp}',
                           GAME_DATE=day, GAME_ID=game, WL='W' if win else 'L', MIN=48,
                           FGM=2 if win else 1, FGA=3, FG3M=0, FG3A=1, FTM=0, FTA=0,
                           PTS=4 if win else 2, OREB=1, DREB=1, REB=2, AST=1, STL=0,
                           BLK=0, TOV=1, PF=1, PLUS_MINUS=2 if win else -2,
                           FG_PCT=.667 if win else .333, FG3_PCT=0, FT_PCT=0)
                rows.append(row)
            source['player_games'].extend(rows)
            totals = {f: sum(r[f] for r in rows) for f in ('MIN','FGM','FGA','FG3M','FG3A','FTM','FTA',
                        'PTS','OREB','DREB','REB','AST','STL','BLK','TOV','PF','PLUS_MINUS')}
            source['team_games'].append({**rows[0], **totals, 'SEASON_ID': '22026'})
    for pid in [*range(1,6), *range(11,16)]:
        database['players'].append(dict(nba_api_id=pid, display_name=f'Player {pid}', team_id=999,
                                       is_active=True, status='existing', bballref_id='preserved'))
        source['roster'].append(dict(PERSON_ID=pid, TEAM_ID=100 if pid<10 else 200, ROSTERSTATUS=1))
    return source, database


class DailyIngestionTests(unittest.TestCase):
    def test_exact_percentages_and_team_ids(self):
        source, db = fixture()
        out = transform(source, db)
        self.assertEqual(len(out['player_gamelog']), 30)
        self.assertEqual(out['player_gamelog'][0]['fg_pct'], '0.6667')
        self.assertEqual(out['player_team_stints'][0]['team_id'], 100)
        self.assertEqual(desired_players(source, db)[0]['team_id'], 10)

    def test_pace_uses_only_prior_games_and_overtime_duration(self):
        source, db = fixture()
        out = transform(source, db)['team_defensive_stats']
        self.assertIsNone(out[0]['opponent_avg_team_pace'])
        self.assertEqual(out[2]['opponent_avg_team_pace'], out[1]['team_pace'])
        changed = copy.deepcopy(source)
        for row in changed['team_games']:
            if row['GAME_DATE']=='2026-10-24':
                row['MIN']=265
        for row in changed['player_games']:
            if row['GAME_DATE']=='2026-10-24':
                row['MIN']=53
        new = transform(changed, db)['team_defensive_stats']
        self.assertEqual(out[:4], new[:4])
        self.assertLess(float(new[4]['pace']), float(out[4]['pace']))

    def test_partial_collection_and_missing_final_game_are_rejected(self):
        source, db = fixture()
        source['player_games'].pop()
        with self.assertRaisesRegex(ValueError, 'totals disagree'):
            validate_sources(source, db['teams'])

    def test_ambiguous_player_uses_independent_box_score_not_bulk_attribution(self):
        source, db = fixture()
        game='0022600001'
        box=dict(gameId=game)
        for side,tid in [('homeTeam',100),('awayTeam',200)]:
            players=[]
            for row in source['player_games']:
                if row['GAME_ID']==game and row['TEAM_ID']==tid:
                    players.append(dict(personId=row['PLAYER_ID'],firstName='Player',familyName=str(row['PLAYER_ID']),
                                   statistics={**{v:row[k] for k,v in BOX_FIELDS.items()},'minutes':'48:00'}))
            box[side]=dict(teamId=tid,players=players)
        wrong={**source['player_games'][5],'PLAYER_ID':1,'MIN':0,'PTS':1,'FTM':1,'FTA':1}
        source['player_games'].append(wrong)
        with self.assertRaisesRegex(ValueError,'independent game box score'):
            validate_sources(source,db['teams'])
        source['box_scores']={game:dict(boxScoreTraditional=box)}
        out=transform(source,db)['player_gamelog']
        self.assertEqual(len(out),30)
        self.assertEqual([(r['team'],r['points']) for r in out if r['playerid']==1 and r['date']=='2026-10-20'],[('AAA',4)])
        source, db = fixture()
        source['scoreboards']['2026-10-24'].append(dict(GAME_ID='0022600099', GAME_STATUS_ID=3))
        with self.assertRaisesRegex(ValueError, 'coverage differs'):
            validate_sources(source, db['teams'])

    def test_only_interior_absence_is_inferred(self):
        source, db = fixture()
        for row in source['player_games']:
            if row['PLAYER_ID']==1 and row['GAME_DATE']=='2026-10-22':
                row['PLAYER_ID']=6
        db['players'].append(dict(nba_api_id=6))
        result = transform(source, db)['player_game_absences']
        self.assertEqual([(r['player_id'], r['date']) for r in result], [(1,'2026-10-22')])

    def test_second_plan_is_idempotent_and_cannot_shrink_raw_coverage(self):
        source, db = fixture()
        plan = build_plan(source, db)
        db['players']=plan['desired']['players']
        db['existing']={t: rows for t, rows in plan['desired'].items() if t!='players'}
        repeat=build_plan(source, db)
        self.assertTrue(all(not any(counts.values()) for counts in repeat['changes'].values()))
        source['player_games']=[r for r in source['player_games'] if r['GAME_DATE']!='2026-10-24']
        source['team_games']=[r for r in source['team_games'] if r['GAME_DATE']!='2026-10-24']
        source['scoreboards'].pop('2026-10-24')
        with self.assertRaisesRegex(ValueError, 'refusing to remove'):
            build_plan(source, db)

    def test_wrong_season_and_future_games_are_rejected(self):
        source, db = fixture()
        source['through']='2026-10-21'
        with self.assertRaisesRegex(ValueError, 'cutoff'):
            validate_sources(source, db['teams'])
        source, db = fixture()
        source['team_games'][0]['GAME_ID']='0022500001'
        with self.assertRaisesRegex(ValueError, 'different season'):
            validate_sources(source, db['teams'])

    def test_season_rollover_and_named_results(self):
        self.assertEqual(current_season(date(2027,1,1)), '2026-27')
        self.assertEqual(result_rows({'resultSets':[dict(name='A', headers=['x'], rowSet=[[1]])]}, 'A'), [{'x':1}])


if __name__=='__main__':
    unittest.main()
