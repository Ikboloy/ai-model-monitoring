import json
import sqlite3
from pathlib import Path
import time

class EventStore:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY, created REAL NOT NULL,
                status INTEGER NOT NULL, latency REAL NOT NULL,
                features TEXT, label TEXT, probability REAL, source TEXT NOT NULL
            )''')
            db.execute('CREATE INDEX IF NOT EXISTS events_created ON events(created)')

    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def record(self, status, latency, features=None, label=None, probability=None, source='user'):
        with self.connect() as db:
            cur = db.execute('INSERT INTO events(created,status,latency,features,label,probability,source) VALUES (?,?,?,?,?,?,?)',
                (time.time(), status, latency, json.dumps(features) if features else None, label, probability, source))
            # Bounded retention for a single-user demo; monitoring reads a smaller window.
            db.execute('DELETE FROM events WHERE id <= (SELECT MAX(id) FROM events)-10000')
            return cur.lastrowid

    def recent(self, limit=1000):
        with self.connect() as db:
            rows = db.execute('SELECT * FROM events WHERE created >= ? ORDER BY id DESC LIMIT ?',
                (time.time()-86400, limit)).fetchall()
        out=[]
        for row in rows:
            event=dict(row)
            event['features']=json.loads(event['features']) if event['features'] else None
            out.append(event)
        return out
