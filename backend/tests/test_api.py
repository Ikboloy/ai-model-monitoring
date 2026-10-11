import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient
from main import create_app
from model import population_stability_index, train_model
from store import EventStore

class ApiTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.path=Path(self.directory.name)/'events.sqlite'
        self.client=TestClient(create_app(self.path, static_dir='/nonexistent'))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None,None,None)
        self.directory.cleanup()

    def predict(self):
        return self.client.post('/api/predict',json={'sepal_length':5.1,'sepal_width':3.5,'petal_length':1.4,'petal_width':0.2})

    def test_prediction_and_metric_provenance(self):
        response=self.predict()
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json()['label'],'setosa')
        metrics=self.client.get('/api/metrics').json()
        self.assertEqual(metrics['requests'],1)
        self.assertEqual(metrics['class_distribution'],{'setosa':1})
        self.assertEqual(metrics['demo_predictions'],0)
        self.assertEqual(metrics['drift'][0]['status'],'insufficient_data')
        self.assertGreaterEqual(metrics['mean_latency_ms'],0)

    def test_validation_failure_is_counted(self):
        self.predict()
        bad=self.client.post('/api/predict',json={'sepal_length':-1})
        self.assertEqual(bad.status_code,422)
        data=self.client.get('/api/metrics').json()
        self.assertEqual(data['requests'],2)
        self.assertEqual(data['error_rate'],0.5)
        self.assertEqual(data['successful_predictions'],1)
        self.assertEqual(sum(x['requests'] for x in data['series']),2)

    def test_model_failure_is_counted(self):
        with patch.object(self.client.app.state.model.classifier,'predict_proba',side_effect=RuntimeError('fail')):
            self.assertEqual(self.predict().status_code,500)
        data=self.client.get('/api/metrics').json()
        self.assertEqual(data['error_rate'],1)
        self.assertIsNone(data['mean_latency_ms'])

    def test_empty_state(self):
        data=self.client.get('/api/metrics').json()
        self.assertEqual(data['requests'],0)
        self.assertIsNone(data['p95_latency_ms'])
        self.assertEqual(data['series'],[])

    def test_demo_is_explicit_and_bounded(self):
        self.assertEqual(self.client.post('/api/demo',json={'count':101}).status_code,422)
        self.assertEqual(self.client.post('/api/demo',json={'count':35,'shifted':True}).status_code,200)
        data=self.client.get('/api/metrics').json()
        self.assertEqual(data['requests'],35)
        self.assertEqual(data['demo_predictions'],35)
        self.assertEqual(data['recent'][0]['source'],'demo_shifted')
        self.assertEqual(data['drift'][0]['status'],'shift')
        with patch.dict(os.environ,{'ENABLE_DEMO':'0'}):
            self.assertEqual(self.client.post('/api/demo',json={'count':1}).status_code,403)

    def test_events_survive_store_reopen(self):
        self.predict()
        self.assertEqual(len(EventStore(self.path).recent()),1)

    def test_monitoring_window_excludes_old_events(self):
        self.predict()
        with self.client.app.state.store.connect() as db:
            db.execute('UPDATE events SET created=0')
        self.assertEqual(self.client.get('/api/metrics').json()['requests'],0)

class DriftTests(unittest.TestCase):
    def test_reference_identity_and_shift(self):
        reference=train_model().reference[:,0]
        self.assertAlmostEqual(population_stability_index(reference,reference),0)
        self.assertGreater(population_stability_index(reference,reference+3),0.25)

if __name__=='__main__':
    unittest.main()
