"""
MarketPulse AI — SQLite Database Layer
Replaces GitHub CSV as the primary data store.
"""

import sqlite3
import json
import os
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager

IST = timezone(timedelta(hours=5, minutes=30))
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "marketpulse.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    headline_count INTEGER DEFAULT 0,
    avg_nss REAL DEFAULT 0,
    avg_risk REAL DEFAULT 0,
    regime TEXT DEFAULT '',
    msi REAL DEFAULT 0,
    msi_level TEXT DEFAULT '',
    predicted_direction TEXT DEFAULT '',
    actual_nifty_open REAL,
    actual_nifty_close REAL,
    actual_direction TEXT,
    regime_correct INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS headlines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pipeline_run_id INTEGER,
    title TEXT NOT NULL,
    description TEXT,
    source TEXT,
    source_url TEXT,
    published TEXT,
    hours_old REAL,
    url TEXT,
    is_govt_source INTEGER DEFAULT 0,
    sector TEXT,
    sentiment TEXT,
    sentiment_confidence REAL,
    impact_score REAL,
    valence REAL,
    arousal REAL,
    geopolitical_risk INTEGER DEFAULT 0,
    affected_companies TEXT,
    second_order_beneficiaries TEXT,
    catalyst_type TEXT,
    price_direction TEXT,
    time_horizon TEXT,
    conviction TEXT,
    macro_sensitivity TEXT,
    one_line_insight TEXT,
    signal_reason TEXT,
    contrarian_flag INTEGER DEFAULT 0,
    contrarian_reason TEXT,
    sentiment_num REAL,
    weighted_risk_score REAL,
    signal_decay REAL,
    recency_weighted_impact REAL,
    catalyst_weight REAL,
    z_score REAL,
    shock_status TEXT DEFAULT 'Normal',
    source_reliability REAL DEFAULT 1.0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pipeline_run_id) REFERENCES pipeline_runs(id)
);

CREATE TABLE IF NOT EXISTS sector_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pipeline_run_id INTEGER,
    date TEXT NOT NULL,
    sector TEXT NOT NULL,
    avg_weighted_risk REAL,
    sentiment_nss REAL,
    impact_weighted_sentiment REAL,
    confidence_weighted_sentiment REAL,
    composite_sentiment_index REAL,
    sentiment_velocity REAL,
    risk_level TEXT,
    avg_impact REAL,
    total_mentions INTEGER,
    positive_count INTEGER,
    negative_count INTEGER,
    neutral_count INTEGER,
    momentum_score REAL,
    divergence REAL,
    divergence_flag TEXT,
    govt_signals INTEGER,
    contrarian_count INTEGER,
    geopolitical_flags INTEGER,
    valence REAL,
    arousal REAL,
    sector_classification TEXT,
    investment_signal TEXT,
    dynamic_sector_weight REAL DEFAULT 1.0,
    signal_correct INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pipeline_run_id) REFERENCES pipeline_runs(id)
);

CREATE TABLE IF NOT EXISTS source_reliability (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name TEXT UNIQUE NOT NULL,
    total_headlines INTEGER DEFAULT 0,
    correct_direction INTEGER DEFAULT 0,
    avg_impact_accuracy REAL DEFAULT 0,
    reliability_score REAL DEFAULT 1.0,
    is_official INTEGER DEFAULT 0,
    last_calibrated TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT UNIQUE NOT NULL,
    predicted_regime TEXT,
    predicted_nss REAL,
    predicted_avg_risk REAL,
    sector_signals TEXT,
    actual_nifty_open REAL,
    actual_nifty_close REAL,
    actual_nifty_change_pct REAL,
    was_regime_correct INTEGER,
    sector_accuracy TEXT,
    overall_accuracy REAL,
    learning_reflection TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dynamic_weights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    weight_type TEXT NOT NULL,
    key TEXT NOT NULL,
    value REAL NOT NULL,
    calibration_source TEXT,
    last_calibrated TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(weight_type, key)
);

CREATE INDEX IF NOT EXISTS idx_headlines_sector ON headlines(sector);
CREATE INDEX IF NOT EXISTS idx_headlines_pipeline ON headlines(pipeline_run_id);
CREATE INDEX IF NOT EXISTS idx_sector_snapshots_date ON sector_snapshots(date);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(date);
"""


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Initialize database schema."""
    with get_db() as conn:
        conn.executescript(SCHEMA)
        _seed_source_reliability(conn)
        _migrate(conn)
    print("  ✓ Database initialized.")


def _migrate(conn: sqlite3.Connection):
    """Lightweight schema migrations for existing DBs."""
    try:
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(predictions)").fetchall()]
        if "learning_reflection" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN learning_reflection TEXT")
    except Exception:
        # Non-fatal; table might not exist yet on first boot.
        pass


def _seed_source_reliability(conn):
    """Seed initial source reliability scores."""
    sources = [
        ("ET Markets", 0, 1.0), ("ET Economy", 0, 1.0), ("ET Tech", 0, 1.0),
        ("ET Startups", 0, 0.9), ("ET Industry", 0, 1.0),
        ("Livemint Markets", 0, 1.05), ("Livemint Companies", 0, 1.0),
        ("Livemint Economy", 0, 1.0),
        ("BS Markets", 0, 1.05), ("BS Economy", 0, 1.0),
        ("MC Latest News", 0, 0.9),
        ("Financial Express Markets", 0, 0.95),
        ("PIB Economy", 1, 1.4), ("RBI", 1, 1.5), ("SEBI", 1, 1.5),
        ("Reuters India", 0, 1.2),
    ]
    for name, is_official, base_score in sources:
        conn.execute(
            """INSERT OR IGNORE INTO source_reliability 
               (source_name, is_official, reliability_score, last_calibrated) 
               VALUES (?, ?, ?, ?)""",
            (name, is_official, base_score, datetime.now(IST).isoformat())
        )


def _seed_dynamic_weights(conn):
    """Seed default dynamic weights (sector + catalyst)."""
    sector_defaults = {
        "Banking": 1.5, "Energy": 1.4, "IT": 1.2, "Fintech": 1.2,
        "Manufacturing": 1.1, "Healthcare": 1.1, "FMCG": 1.0,
        "Startup": 0.9, "Retail": 0.8, "Other": 0.7
    }
    catalyst_defaults = {
        "government_contract": 1.7, "policy_change": 1.6, "pib_announcement": 1.6,
        "rbi_action": 1.5, "sebi_action": 1.5, "fii_flow": 1.4,
        "capex_announcement": 1.4, "earnings": 1.3, "sector_tailwind": 1.2,
        "regulatory": 1.1, "management_change": 1.0, "global_event": 0.9,
        "merger_acquisition": 0.7, "funding": 0.7, "funding_round": 0.7,
        "supply_chain_disruption": 0.7, "other": 0.7
    }
    now = datetime.now(IST).isoformat()
    for key, val in sector_defaults.items():
        conn.execute(
            """INSERT OR IGNORE INTO dynamic_weights (weight_type, key, value, calibration_source, last_calibrated) 
               VALUES (?, ?, ?, ?, ?)""",
            ("sector", key, val, "default", now)
        )
    for key, val in catalyst_defaults.items():
        conn.execute(
            """INSERT OR IGNORE INTO dynamic_weights (weight_type, key, value, calibration_source, last_calibrated) 
               VALUES (?, ?, ?, ?, ?)""",
            ("catalyst", key, val, "default", now)
        )


# ─── Pipeline Run Management ────────────────────────────────────────

def create_pipeline_run():
    """Create a new pipeline run record and return its ID."""
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO pipeline_runs (started_at) VALUES (?)",
            (datetime.now(IST).isoformat(),)
        )
        return cursor.lastrowid


def complete_pipeline_run(run_id, headline_count, avg_nss, avg_risk, regime, msi, msi_level):
    """Mark a pipeline run as complete with results."""
    predicted_direction = "bullish" if avg_nss > 0 else "bearish" if avg_nss < 0 else "neutral"
    with get_db() as conn:
        conn.execute(
            """UPDATE pipeline_runs SET 
               completed_at=?, headline_count=?, avg_nss=?, avg_risk=?, 
               regime=?, msi=?, msi_level=?, predicted_direction=?
               WHERE id=?""",
            (datetime.now(IST).isoformat(), headline_count, avg_nss, avg_risk,
             regime, msi, msi_level, predicted_direction, run_id)
        )


# ─── Headlines ──────────────────────────────────────────────────────

def save_headlines(headlines_df, run_id):
    """Save all headlines from a pipeline run."""
    with get_db() as conn:
        for _, row in headlines_df.iterrows():
            conn.execute(
                """INSERT INTO headlines 
                   (pipeline_run_id, title, description, source, source_url, published, 
                    hours_old, url, is_govt_source, sector, sentiment, sentiment_confidence,
                    impact_score, valence, arousal, geopolitical_risk, affected_companies,
                    second_order_beneficiaries, catalyst_type, price_direction, time_horizon,
                    conviction, macro_sensitivity, one_line_insight, signal_reason,
                    contrarian_flag, contrarian_reason, sentiment_num, weighted_risk_score,
                    signal_decay, recency_weighted_impact, catalyst_weight, z_score, 
                    shock_status, source_reliability)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, str(row.get("title", "")), str(row.get("description", "")),
                 str(row.get("source", "")), str(row.get("source_url", "")),
                 str(row.get("published", "")), float(row.get("hours_old", 0)),
                 str(row.get("url", "")), int(bool(row.get("is_govt_source", False))),
                 str(row.get("sector", "Other")), str(row.get("sentiment", "neutral")),
                 float(row.get("sentiment_confidence", 0.5)),
                 float(row.get("impact_score", 5)), float(row.get("valence", 0.5)),
                 float(row.get("arousal", 0.5)), int(bool(row.get("geopolitical_risk", False))),
                 json.dumps(row.get("affected_companies", [])) if isinstance(row.get("affected_companies"), list) else str(row.get("affected_companies", "[]")),
                 json.dumps(row.get("second_order_beneficiaries", [])) if isinstance(row.get("second_order_beneficiaries"), list) else str(row.get("second_order_beneficiaries", "[]")),
                 str(row.get("catalyst_type", "other")),
                 str(row.get("price_direction", "neutral")),
                 str(row.get("time_horizon", "intraday")),
                 str(row.get("conviction", "low")),
                 str(row.get("macro_sensitivity", "medium")),
                 str(row.get("one_line_insight", "")),
                 str(row.get("signal_reason", "")),
                 int(bool(row.get("contrarian_flag", False))),
                 str(row.get("contrarian_reason", "")),
                 float(row.get("sentiment_num", 0)),
                 float(row.get("weighted_risk_score", 0)),
                 float(row.get("signal_decay", 1.0)),
                 float(row.get("recency_weighted_impact", 0)),
                 float(row.get("catalyst_weight", 0.7)),
                 float(row.get("z_score", 0)),
                 str(row.get("shock_status", "Normal")),
                 float(row.get("source_reliability", 1.0)))
            )


def get_latest_headlines():
    """Get headlines from the most recent pipeline run."""
    with get_db() as conn:
        run = conn.execute(
            "SELECT id FROM pipeline_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not run:
            return []
        rows = conn.execute(
            "SELECT * FROM headlines WHERE pipeline_run_id = ? ORDER BY impact_score DESC",
            (run["id"],)
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def search_stock(query):
    """Search headlines by company name."""
    with get_db() as conn:
        run = conn.execute(
            "SELECT id FROM pipeline_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not run:
            return []
        rows = conn.execute(
            """SELECT * FROM headlines 
               WHERE pipeline_run_id = ? 
               AND (affected_companies LIKE ? OR title LIKE ? OR second_order_beneficiaries LIKE ?)
               ORDER BY impact_score DESC""",
            (run["id"], f"%{query}%", f"%{query}%", f"%{query}%")
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


# ─── Sector Snapshots ───────────────────────────────────────────────

def save_sector_snapshots(sector_df, run_id, date_str):
    """Save sector data as a daily snapshot."""
    with get_db() as conn:
        for _, row in sector_df.iterrows():
            conn.execute(
                """INSERT INTO sector_snapshots 
                   (pipeline_run_id, date, sector, avg_weighted_risk, sentiment_nss,
                    impact_weighted_sentiment, confidence_weighted_sentiment,
                    composite_sentiment_index, sentiment_velocity, risk_level,
                    avg_impact, total_mentions, positive_count, negative_count,
                    neutral_count, momentum_score, divergence, divergence_flag,
                    govt_signals, contrarian_count, geopolitical_flags, valence,
                    arousal, sector_classification, investment_signal, dynamic_sector_weight)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, date_str, str(row.get("sector", "")),
                 float(row.get("avg_weighted_risk", 0)),
                 float(row.get("sentiment_nss", 0)),
                 float(row.get("impact_weighted_sentiment", 0)),
                 float(row.get("confidence_weighted_sentiment", 0)),
                 float(row.get("composite_sentiment_index", 0)),
                 float(row.get("sentiment_velocity", 0)),
                 str(row.get("risk_level", "LOW")),
                 float(row.get("avg_impact", 0)),
                 int(row.get("total_mentions", 0)),
                 int(row.get("positive_count", 0)),
                 int(row.get("negative_count", 0)),
                 int(row.get("neutral_count", 0)),
                 float(row.get("momentum_score", 0)),
                 float(row.get("divergence", 0)),
                 str(row.get("divergence_flag", "Normal")),
                 int(row.get("govt_signals", 0)),
                 int(row.get("contrarian_count", 0)),
                 int(row.get("geopolitical_flags", 0)),
                 float(row.get("valence", 0.5)),
                 float(row.get("arousal", 0.5)),
                 str(row.get("sector_classification", "")),
                 str(row.get("investment_signal", "NEUTRAL")),
                 float(row.get("dynamic_sector_weight", 1.0)))
            )


def get_latest_sectors():
    """Get sector data from the most recent pipeline run."""
    with get_db() as conn:
        run = conn.execute(
            "SELECT id FROM pipeline_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not run:
            return []
        rows = conn.execute(
            "SELECT * FROM sector_snapshots WHERE pipeline_run_id = ? ORDER BY avg_weighted_risk DESC",
            (run["id"],)
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def get_previous_sector_csi():
    """Get CSI from the second-most-recent pipeline run for velocity calc."""
    with get_db() as conn:
        runs = conn.execute(
            "SELECT id FROM pipeline_runs ORDER BY id DESC LIMIT 2"
        ).fetchall()
        if len(runs) < 2:
            return {}
        prev_run_id = runs[1]["id"]
        rows = conn.execute(
            "SELECT sector, composite_sentiment_index FROM sector_snapshots WHERE pipeline_run_id = ?",
            (prev_run_id,)
        ).fetchall()
        return {r["sector"]: r["composite_sentiment_index"] for r in rows}


def get_sector_history(sector=None, days=30):
    """Get historical sector snapshots for trend analysis."""
    with get_db() as conn:
        if sector:
            rows = conn.execute(
                """SELECT * FROM sector_snapshots 
                   WHERE sector = ? 
                   ORDER BY date DESC LIMIT ?""",
                (sector, days)
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT date, sector, composite_sentiment_index, avg_weighted_risk, 
                          investment_signal, sentiment_velocity
                   FROM sector_snapshots 
                   ORDER BY date DESC LIMIT ?""",
                (days * 12,)
            ).fetchall()
        return [_row_to_dict(r) for r in rows]


# ─── Source Reliability ──────────────────────────────────────────────

def get_source_reliability():
    """Get all source reliability scores."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM source_reliability ORDER BY reliability_score DESC"
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def get_source_reliability_map():
    """Get source → reliability_score map for pipeline use."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT source_name, reliability_score FROM source_reliability"
        ).fetchall()
        return {r["source_name"]: r["reliability_score"] for r in rows}


def update_source_reliability(source_name, was_correct):
    """Update a source's reliability based on prediction accuracy."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM source_reliability WHERE source_name = ?",
            (source_name,)
        ).fetchone()
        if not row:
            return
        total = row["total_headlines"] + 1
        correct = row["correct_direction"] + (1 if was_correct else 0)
        # Bayesian-smoothed reliability: base_rate * 0.3 + accuracy * 0.7
        base_rate = 1.4 if row["is_official"] else 1.0
        accuracy_rate = correct / total if total > 0 else 0.5
        new_score = round(base_rate * 0.3 + (accuracy_rate * 2.0) * 0.7, 3)
        new_score = max(0.5, min(2.0, new_score))
        conn.execute(
            """UPDATE source_reliability 
               SET total_headlines=?, correct_direction=?, reliability_score=?, last_calibrated=?
               WHERE source_name=?""",
            (total, correct, new_score, datetime.now(IST).isoformat(), source_name)
        )


# ─── Dynamic Weights ────────────────────────────────────────────────

def get_dynamic_weights(weight_type="sector"):
    """Get current dynamic weights."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT key, value FROM dynamic_weights WHERE weight_type = ?",
            (weight_type,)
        ).fetchall()
        if not rows:
            _seed_dynamic_weights(conn)
            rows = conn.execute(
                "SELECT key, value FROM dynamic_weights WHERE weight_type = ?",
                (weight_type,)
            ).fetchall()
        return {r["key"]: r["value"] for r in rows}


def update_dynamic_weight(weight_type, key, value, source="calibration"):
    """Update a single dynamic weight."""
    with get_db() as conn:
        conn.execute(
            """INSERT INTO dynamic_weights (weight_type, key, value, calibration_source, last_calibrated)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(weight_type, key) DO UPDATE SET 
               value=excluded.value, calibration_source=excluded.calibration_source, 
               last_calibrated=excluded.last_calibrated""",
            (weight_type, key, value, source, datetime.now(IST).isoformat())
        )


# ─── Predictions & Accuracy ─────────────────────────────────────────

def save_prediction(date_str, regime, nss, avg_risk, sector_signals):
    """Save today's prediction for later accuracy verification."""
    with get_db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO predictions 
               (date, predicted_regime, predicted_nss, predicted_avg_risk, sector_signals)
               VALUES (?, ?, ?, ?, ?)""",
            (date_str, regime, nss, avg_risk, json.dumps(sector_signals))
        )


def update_prediction_accuracy(date_str, nifty_open, nifty_close, change_pct, was_correct, sector_accuracy, overall):
    """Update a prediction with actual market data."""
    with get_db() as conn:
        conn.execute(
            """UPDATE predictions SET 
               actual_nifty_open=?, actual_nifty_close=?, actual_nifty_change_pct=?,
               was_regime_correct=?, sector_accuracy=?, overall_accuracy=?
               WHERE date=?""",
            (nifty_open, nifty_close, change_pct, int(was_correct),
             json.dumps(sector_accuracy), overall, date_str)
        )


def get_accuracy_stats(days=30):
    """Get prediction accuracy over a rolling window."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT * FROM predictions 
               WHERE was_regime_correct IS NOT NULL 
               ORDER BY date DESC LIMIT ?""",
            (days,)
        ).fetchall()
        if not rows:
            return {"total": 0, "correct": 0, "accuracy": 0, "predictions": []}
        
        total = len(rows)
        correct = sum(1 for r in rows if r["was_regime_correct"])
        accuracy = round((correct / total) * 100, 1) if total > 0 else 0
        
        return {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "predictions": [_row_to_dict(r) for r in rows]
        }


def get_latest_pipeline_info():
    """Get info about the most recent pipeline run."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM pipeline_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return _row_to_dict(row) if row else None


# ─── Helpers ─────────────────────────────────────────────────────────

def _row_to_dict(row):
    """Convert a sqlite3.Row to a dict, parsing JSON strings."""
    if row is None:
        return None
    d = dict(row)
    # Parse JSON fields
    for key in ["affected_companies", "second_order_beneficiaries", "sector_signals", "sector_accuracy"]:
        if key in d and isinstance(d[key], str):
            try:
                d[key] = json.loads(d[key])
            except (json.JSONDecodeError, TypeError):
                pass
    # Convert boolean ints back to bools
    for key in ["is_govt_source", "geopolitical_risk", "contrarian_flag", "is_official"]:
        if key in d:
            d[key] = bool(d[key])
    return d


# Initialize on import
init_db()
