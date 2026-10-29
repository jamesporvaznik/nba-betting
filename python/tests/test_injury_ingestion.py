from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch, Mock
from tempfile import TemporaryDirectory

from nba.ingestion.injuries import normalize_observations, confirmed_absences, normalized_name, collect_report, enrich_player_ids
from nba.install_ingestion_schedule import configurations


class InjuryIngestionTests(unittest.TestCase):
    def fixture(self):
        rows=[{'Game Date':'10/28/2026','Game Time':'07:00 (ET)','Matchup':'LAL@LAC',
               'Team':'LA Clippers','Player Name':f'Player, {i}','Current Status':s,'Reason':'Rest'}
              for i,s in enumerate(['Out','Questionable','Doubtful','Probable','Available'])]
        players=[dict(nba_api_id=i+1,display_name=f'{i} Player') for i in range(5)]
        teams=[dict(full_name='Los Angeles Clippers',abbreviation='LAC')]
        normalized=normalize_observations(rows,players,teams)
        as_of=datetime(2026,10,28,22,tzinfo=timezone.utc)
        snapshot=dict(report_at=(as_of-timedelta(minutes=15)).isoformat(),
                      fetched_at=(as_of-timedelta(minutes=10)).isoformat(),observations=normalized)
        return rows,players,teams,snapshot,as_of,as_of+timedelta(hours=1)

    def test_only_out_counts_and_names_use_exact_reordered_match(self):
        _,_,_,snapshot,now,start=self.fixture()
        self.assertEqual([r['confirmed_out'] for r in snapshot['observations']],[True,False,False,False,False])
        result=confirmed_absences(snapshot,'2026-10-28','LAC',now,start)
        self.assertTrue(result['coverage_known'])
        self.assertEqual([r['player_id'] for r in result['rows']],[1])

    def test_stale_and_future_fetched_reports_cannot_supply_pregame_features(self):
        _,_,_,snapshot,now,start=self.fixture()
        for change in [dict(report_at=(now-timedelta(hours=3)).isoformat()),
                       dict(fetched_at=(now+timedelta(minutes=1)).isoformat())]:
            with self.subTest(change=change):
                result=confirmed_absences({**snapshot,**change},'2026-10-28','LAC',now,start)
                self.assertFalse(result['coverage_known'])
                self.assertEqual(result['rows'],[])
        with self.assertRaisesRegex(ValueError,'tip-off'):
            confirmed_absences(snapshot,'2026-10-28','LAC',start,start)

    def test_unknown_players_and_unsubmitted_reports_do_not_imply_healthy_roster(self):
        rows,players,teams,snapshot,now,start=self.fixture()
        rows[0]['Player Name']='Unknown, Person'
        snapshot['observations']=normalize_observations(rows,players,teams)
        self.assertFalse(confirmed_absences(snapshot,'2026-10-28','LAC',now,start)['coverage_known'])
        rows=[{**rows[0],'Player Name':None,'Current Status':None,'Reason':'NOT YET SUBMITTED'}]
        snapshot['observations']=normalize_observations(rows,players,teams)
        self.assertFalse(snapshot['observations'][0]['confirmed_out'])
        self.assertFalse(confirmed_absences(snapshot,'2026-10-28','LAC',now,start)['coverage_known'])

    def test_status_change_does_not_carry_forward_previous_out(self):
        _,_,_,snapshot,now,start=self.fixture()
        snapshot['observations'][0].update(status='Questionable',confirmed_out=False)
        self.assertEqual(confirmed_absences(snapshot,'2026-10-28','LAC',now,start)['rows'],[])

    def test_unknown_status_is_rejected(self):
        rows,players,teams,_,_,_=self.fixture()
        rows[0]['Current Status']='Maybe'
        with self.assertRaisesRegex(ValueError,'status'):
            normalize_observations(rows,players,teams)

    def test_report_suffix_with_multiple_commas_matches_player(self):
        self.assertEqual(normalized_name('Udeh, Jr., Ernest'), normalized_name('Ernest Udeh Jr.'))
        self.assertEqual(normalized_name('Udeh, Jr., Ernest'), normalized_name('Ernest Udeh, Jr.'))

    def test_identity_enrichment_preserves_published_status(self):
        _,_,_,snapshot,_,_=self.fixture()
        old=snapshot['observations']
        new=[dict(r) for r in old]
        old[0]['player_id']=None
        enriched,count=enrich_player_ids(old,new)
        self.assertEqual(count,1)
        self.assertEqual(enriched[0]['player_id'],1)
        new[0]['status']='Questionable'
        with self.assertRaisesRegex(ValueError,'Published observation'):
            enrich_player_ids(old,new)

    def test_unpublished_403_slot_falls_back_to_a_verified_prior_report(self):
        rows,players,teams,_,now,_=self.fixture()
        blocked=Mock(status_code=403)
        published=Mock(status_code=200,content=b'%PDF fixture')
        with TemporaryDirectory() as folder, \
             patch('nba.ingestion.injuries.requests.get',side_effect=[blocked,published]) as get, \
             patch('nbainjuries.injury.get_reportdata',return_value=__import__('json').dumps(rows)):
            result=collect_report(folder,players,teams,now=now)
        self.assertEqual(get.call_count,2)
        self.assertEqual(datetime.fromisoformat(result['report_at']),now-timedelta(minutes=15))

    def test_schedule_passes_arguments_without_shell_or_credentials(self):
        jobs=configurations('/tmp/project with spaces',injuries=True)
        self.assertEqual(len(jobs),2)
        self.assertEqual(jobs[0]['ProgramArguments'][0],'/tmp/project with spaces/.venv/bin/python')
        self.assertIn('--apply',jobs[0]['ProgramArguments'])
        self.assertEqual(jobs[0]['StartCalendarInterval'],dict(Hour=7,Minute=15))
        self.assertEqual(len(jobs[1]['StartCalendarInterval']),56)
        self.assertFalse(jobs[1]['RunAtLoad'])


if __name__=='__main__':
    unittest.main()
