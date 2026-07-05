"""SQLite connection pool and initialization for Geometric Lens state."""

import os
import sqlite3
import threading
from contextlib import contextmanager


def _resolve_db_path() -> str:
    """Resolve the database path.

    Precedence: SQLITE_DB_PATH env var, then /data/state/geometric_state.db
    when /data/state exists (container deployments mount a volume there),
    else geometric_state.db in the working directory (host/dev runs).
    The parent directory is created when missing.
    """
    path = os.environ.get("SQLITE_DB_PATH")
    if not path:
        if os.path.isdir("/data/state"):
            path = "/data/state/geometric_state.db"
        else:
            path = "geometric_state.db"
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


DB_PATH = _resolve_db_path()

class SQLitePool:
    """A thread-safe singleton SQLite connection pool emulator using thread-local connections."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SQLitePool, cls).__new__(cls)
                cls._instance._local = threading.local()
                cls._instance._init_db()
        return cls._instance

    def _init_db(self):
        """Initialize schema and pragmas."""
        # Use a temporary connection just for initialization
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=5000;")

            # Metrics daily
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics_daily (
                    date TEXT,
                    key TEXT,
                    value INTEGER,
                    PRIMARY KEY (date, key)
                )
            """)
            
            # Metrics recent tasks
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics_recent_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_record TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tasks queue
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    priority TEXT,
                    status TEXT,
                    data TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tasks_status_priority_created
                ON tasks (status, priority, created_at)
            """)

            # Thompson sampling state
            conn.execute("""
                CREATE TABLE IF NOT EXISTS thompson_state (
                    difficulty_bin TEXT,
                    route TEXT,
                    alpha REAL DEFAULT 0.0,
                    beta REAL DEFAULT 0.0,
                    PRIMARY KEY (difficulty_bin, route)
                )
            """)
            
            # Routing stats
            conn.execute("""
                CREATE TABLE IF NOT EXISTS routing_stats (
                    key TEXT PRIMARY KEY,
                    value INTEGER DEFAULT 0
                )
            """)
            
            # Pattern cache
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    id TEXT PRIMARY KEY,
                    data TEXT,
                    tier TEXT,
                    score REAL
                )
            """)
            
            # Store metadata (e.g. version)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS store_metadata (
                    key TEXT PRIMARY KEY,
                    value INTEGER
                )
            """)
            
            # Co-occurrence graph
            conn.execute("""
                CREATE TABLE IF NOT EXISTS co_occurrence (
                    source_id TEXT,
                    target_id TEXT,
                    count INTEGER DEFAULT 0,
                    PRIMARY KEY (source_id, target_id)
                )
            """)
            
            conn.commit()
        finally:
            conn.close()

    @contextmanager
    def get_connection(self):
        """Get a thread-local SQLite connection with WAL enabled."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(DB_PATH)
            self._local.conn.execute("PRAGMA journal_mode=WAL;")
            self._local.conn.execute("PRAGMA synchronous=NORMAL;")
            self._local.conn.execute("PRAGMA busy_timeout=5000;")
            self._local.conn.row_factory = sqlite3.Row
        
        try:
            yield self._local.conn
            self._local.conn.commit()
        except Exception:
            self._local.conn.rollback()
            raise

def get_db_pool() -> SQLitePool:
    """Get the global SQLite connection pool."""
    return SQLitePool()
