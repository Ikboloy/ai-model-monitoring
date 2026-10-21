import csv
from io import StringIO
import tempfile
from pathlib import Path
import unittest
from fastapi.testclient import TestClient
from main import create_app


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.client = TestClient(create_app(Path(self.temp.name) / 'events.sqlite', '/nonexistent'))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def test_empty_exports(self):
        response = self.client.get('/api/export/events.csv')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(list(csv.DictReader(StringIO(response.text))), [])
        self.assertEqual(response.headers['cache-control'], 'no-store')
        self.assertIn('attachment', response.headers['content-disposition'])
        report = self.client.get('/api/export/metrics.json').json()
        self.assertEqual(report['metrics']['requests'], 0)
        self.assertIn('version', report['model'])

    def test_export_preserves_provenance_without_recording_download(self):
        self.client.post('/api/demo', json={'count': 2})
        self.client.post('/api/predict', json={})
        rows = list(csv.DictReader(StringIO(self.client.get('/api/export/events.csv').text)))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]['status'], '422')
        self.assertEqual(rows[0]['latency_kind'], 'failure_processing')
        self.assertEqual(rows[0]['label'], '')
        self.assertEqual(rows[1]['source'], 'demo_baseline')
        self.assertEqual(rows[1]['latency_kind'], 'inference')
        report = self.client.get('/api/export/metrics.json').json()
        self.assertEqual(report['metrics']['requests'], 3)
        self.assertEqual(report['metrics']['errors'], 1)
        self.assertEqual(report['metrics']['demo_predictions'], 2)

    def test_export_excludes_expired_events(self):
        self.client.post('/api/demo', json={'count': 1})
        with self.client.app.state.store.connect() as db:
            db.execute('UPDATE events SET created=0')
        self.assertEqual(list(csv.DictReader(StringIO(self.client.get('/api/export/events.csv').text))), [])
