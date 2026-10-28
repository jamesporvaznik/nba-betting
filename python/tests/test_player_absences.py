import unittest

from nba.build_player_absences import build


class PlayerAbsenceTests(unittest.TestCase):
    def snapshot(self):
        snapshot = {
            'teams': [{'nba_api_id': 100, 'abbreviation': 'AAA'}, {'nba_api_id': 200, 'abbreviation': 'BBB'}],
            'games': [{'playerid': 1, 'season': '2024-25', 'date': date, 'team': team}
                      for date, team in [('2025-01-02', 'AAA'), ('2025-01-04', 'AAA'),
                                         ('2025-01-07', 'BBB'), ('2025-01-09', 'BBB')]],
            'stints': [{'player_id': 1, 'team_id': team, 'season': '2024-25', 'stint_number': n,
                        'first_game_date': first, 'last_game_date': last}
                       for team, n, first, last in [(100, 1, '2025-01-02', '2025-01-04'),
                                                   (200, 2, '2025-01-07', '2025-01-09')]],
            'calendar': [{'season': '2024-25', 'team': team, 'date': f'2025-01-{day:02d}',
                          'opponent_abbrev': 'CCC', 'game_id': str(day)}
                         for team in ['AAA', 'BBB'] for day in range(1, 11)]}
        for player, team_id, team in [(2, 100, 'AAA'), (3, 200, 'BBB')]:
            snapshot['games'].extend({'playerid': player, 'season': '2024-25', 'date': f'2025-01-{day:02d}', 'team': team}
                                     for day in range(1, 11))
            snapshot['stints'].append({'player_id': player, 'team_id': team_id, 'season': '2024-25',
                                       'stint_number': 1, 'first_game_date': '2025-01-01', 'last_game_date': '2025-01-10'})
        return snapshot

    def test_only_interior_gaps_not_boundaries_or_trade_transitions(self):
        rows = build(self.snapshot())
        self.assertEqual([(r['team'], r['date']) for r in rows],
                         [('AAA', '2025-01-03'), ('BBB', '2025-01-08')])

    def test_any_existing_log_including_zero_minutes_prevents_absence(self):
        snapshot = self.snapshot()
        snapshot['games'].append({'playerid': 1, 'season': '2024-25', 'date': '2025-01-03', 'team': 'AAA', 'minutes': 0})
        self.assertEqual([r['date'] for r in build(snapshot)], ['2025-01-08'])

    def test_calendar_gaps_are_not_invented(self):
        snapshot = self.snapshot()
        snapshot['calendar'] = [r for r in snapshot['calendar'] if r['date'] != '2025-01-03']
        snapshot['games'] = [r for r in snapshot['games'] if r['date'] != '2025-01-03']
        self.assertEqual([r['date'] for r in build(snapshot)], ['2025-01-08'])

    def test_no_team_player_logs_means_unknown_not_everyone_absent(self):
        snapshot = self.snapshot()
        snapshot['games'] = [r for r in snapshot['games'] if r['date'] != '2025-01-03']
        self.assertEqual([r['date'] for r in build(snapshot)], ['2025-01-08'])

    def test_invalid_stints_or_missing_calendar_or_duplicate_appearances_fail(self):
        for case in ['boundary', 'overlap', 'calendar', 'duplicate']:
            snapshot = self.snapshot()
            if case == 'boundary':
                snapshot['stints'][0]['first_game_date'] = '2025-01-01'
            elif case == 'overlap':
                snapshot['stints'].append(snapshot['stints'][0])
            elif case == 'calendar':
                snapshot['calendar'] = [r for r in snapshot['calendar'] if r['date'] != '2025-01-02']
            else:
                snapshot['games'].append(snapshot['games'][0])
            with self.subTest(case=case), self.assertRaises(ValueError):
                build(snapshot)


if __name__ == '__main__':
    unittest.main()
