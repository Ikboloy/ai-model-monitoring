from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import time
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
import numpy as np

from model import FEATURES, VERSION, population_stability_index, train_model
from store import EventStore

class PredictionInput(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    sepal_length: float = Field(gt=0, le=20)
    sepal_width: float = Field(gt=0, le=20)
    petal_length: float = Field(gt=0, le=20)
    petal_width: float = Field(gt=0, le=20)

class DemoInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    count: int = Field(default=50, ge=1, le=100)
    shifted: bool = False


def iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def create_app(db_path=None, static_dir=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.model = train_model()
        app.state.store = EventStore(db_path or os.getenv('MONITOR_DB', 'data/events.sqlite'))
        yield

    app = FastAPI(title='ModelScope API', version='1.0.0', lifespan=lifespan)

    @app.middleware('http')
    async def timer(request, call_next):
        request.state.started = time.perf_counter()
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc):
        if request.url.path == '/api/predict':
            app.state.store.record(422, (time.perf_counter()-request.state.started)*1000)
        # Avoid echoing invalid values or leaking internal exception details.
        return JSONResponse(status_code=422, content={'detail': 'Provide four finite positive measurements between 0 and 20 cm, with no extra fields.'})

    def predict(features, source='user'):
        started = time.perf_counter()
        bundle = app.state.model
        try:
            probabilities = bundle.classifier.predict_proba([features])[0]
            index = int(np.argmax(probabilities))
            label = bundle.classes[index]
            probability = float(probabilities[index])
        except Exception:
            app.state.store.record(500, (time.perf_counter()-started)*1000, source=source)
            raise
        latency = (time.perf_counter()-started)*1000
        event_id = app.state.store.record(200, latency, features, label, probability, source)
        return {'id': event_id, 'label': label, 'probability': probability,
                'latency_ms': latency, 'model_version': VERSION}

    @app.get('/api/health')
    def health():
        return {'status': 'ok', 'model_version': VERSION}

    @app.get('/api/model')
    def model():
        bundle = app.state.model
        return {'version': VERSION, 'algorithm': 'Random forest', 'dataset': 'Iris',
                'features': FEATURES, 'classes': bundle.classes,
                'heldout_accuracy': bundle.accuracy, 'heldout_samples':45,
                'reference_samples':len(bundle.reference), 'random_seed':42}

    @app.post('/api/predict')
    def prediction(body: PredictionInput):
        try:
            return predict([getattr(body, feature) for feature in FEATURES])
        except Exception:
            return JSONResponse(status_code=500, content={'detail':'Prediction failed. Check server logs.'})

    @app.post('/api/demo')
    def demo(body: DemoInput):
        if os.getenv('ENABLE_DEMO', '1') != '1':
            return JSONResponse(status_code=403, content={'detail':'Demo traffic is disabled.'})
        rng = np.random.default_rng(int(uuid4().int % 2**32))
        bundle = app.state.model
        rows = bundle.reference[rng.integers(0,len(bundle.reference),body.count)].copy()
        if body.shifted:
            rows[:, 0] += 3.0  # Synthetic feature shift, not real-world evidence.
        for row in rows:
            predict(row.tolist(), 'demo_shifted' if body.shifted else 'demo_baseline')
        return {'generated':body.count,'source':'demo_shifted' if body.shifted else 'demo_baseline'}

    @app.get('/api/metrics')
    def metrics():
        events = app.state.store.recent()
        successes = [e for e in events if e['status']==200]
        failures = len(events)-len(successes)
        latencies=[e['latency'] for e in successes]
        series={}
        for e in events:
            bucket=int(e['created']//60)*60
            point=series.setdefault(bucket,{'time':iso(bucket),'requests':0,'errors':0,'latencies':[]})
            point['requests']+=1
            point['errors']+=int(e['status']!=200)
            if e['status']==200: point['latencies'].append(e['latency'])
        points=[]
        for key in sorted(series):
            point=series[key];values=point.pop('latencies')
            point['mean_latency_ms']=float(np.mean(values)) if values else None
            points.append(point)
        enough=len(successes)>=30
        drift=[]
        for i,feature in enumerate(FEATURES):
            value = population_stability_index(app.state.model.reference[:,i],
                [e['features'][i] for e in successes]) if enough else None
            drift.append({'feature':feature,'psi':value,
                'status':'insufficient_data' if value is None else ('shift' if value>=0.25 else 'watch' if value>=0.1 else 'stable')})
        return {'window':'Latest 1,000 prediction attempts within 24 hours',
                'requests':len(events),'errors':failures,
                'error_rate':failures/len(events) if events else 0,
                'mean_latency_ms':float(np.mean(latencies)) if latencies else None,
                'p95_latency_ms':float(np.percentile(latencies,95)) if latencies else None,
                'successful_predictions':len(successes),
                'demo_predictions':sum(e['source'].startswith('demo') for e in successes),
                'class_distribution':dict(Counter(e['label'] for e in successes)),
                'series':points, 'drift':drift,
                'recent':[{'id':e['id'],'time':iso(e['created']),'status':e['status'],
                    'label':e['label'],'probability':e['probability'],'latency_ms':e['latency'],
                    'source':e['source']} for e in events[:15]]}

    directory = Path(static_dir or os.getenv('STATIC_DIR', '../frontend/dist'))
    if directory.is_dir():
        app.mount('/', StaticFiles(directory=str(directory), html=True), name='dashboard')
    return app

app = create_app()
