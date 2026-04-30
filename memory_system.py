"""
Lightweight Agent Memory System (durable-ready)

Goal:
- Store discovered correlations, reasoning chains, and failure patterns
- Query them later to avoid repeating mistakes

Implementation notes:
- Uses the existing SQLite DB (no heavy vector DB deps on Render)
- Provides "good enough" similarity search via token overlap scoring
- Can be mirrored to GitHub snapshots (MarketPulseAIData) by the existing pipeline save_all() flow
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import database as db

IST = timezone(timedelta(hours=5, minutes=30))


def _tokenize(text: str) -> set[str]:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    toks = [t for t in text.split() if len(t) >= 3]
    return set(toks)


def _similarity(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / max(union, 1)


def _ensure_tables():
    with db.get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_correlations (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              trigger TEXT NOT NULL,
              effect TEXT NOT NULL,
              strength REAL DEFAULT 0.5,
              verified INTEGER DEFAULT 0,
              context_json TEXT,
              times_observed INTEGER DEFAULT 1,
              discovered_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_reasoning_chains (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              situation TEXT,
              thoughts_json TEXT,
              tools_json TEXT,
              conclusion TEXT,
              outcome TEXT,
              created_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_failure_patterns (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              date TEXT,
              predicted_regime TEXT,
              actual_direction TEXT,
              failure_type TEXT,
              narrative TEXT,
              created_at TEXT
            )
            """
        )


@dataclass
class Correlation:
    trigger: str
    effect: str
    strength: float
    verified: bool
    relevance_score: float
    times_observed: int


class AgentMemory:
    def __init__(self):
        _ensure_tables()

    def store_discovered_correlation(
        self,
        trigger: str,
        effect: str,
        strength: float,
        context: Dict[str, Any],
        outcome_verified: bool = False,
    ):
        now = datetime.now(IST).isoformat()
        with db.get_db() as conn:
            # If we already have a similar correlation, increment observation count.
            rows = conn.execute("SELECT id, trigger, effect, times_observed FROM agent_correlations").fetchall()
            best_id = None
            best_score = 0.0
            for r in rows:
                sc = _similarity(trigger, r["trigger"]) * 0.6 + _similarity(effect, r["effect"]) * 0.4
                if sc > best_score:
                    best_score = sc
                    best_id = r["id"]
            if best_id is not None and best_score >= 0.80:
                conn.execute(
                    "UPDATE agent_correlations SET times_observed = times_observed + 1, strength = ?, verified = ?, context_json = ?, discovered_at = ? WHERE id = ?",
                    (float(strength), int(bool(outcome_verified)), json.dumps(context, default=str), now, best_id),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO agent_correlations (trigger, effect, strength, verified, context_json, times_observed, discovered_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (trigger, effect, float(strength), int(bool(outcome_verified)), json.dumps(context, default=str), 1, now),
                )

    def query_similar_correlations(self, current_situation: str, top_k: int = 5) -> List[Dict[str, Any]]:
        with db.get_db() as conn:
            rows = conn.execute(
                "SELECT trigger, effect, strength, verified, times_observed FROM agent_correlations ORDER BY id DESC LIMIT 500"
            ).fetchall()
        scored: list[Correlation] = []
        for r in rows:
            combined = f"{r['trigger']} -> {r['effect']}"
            rel = _similarity(current_situation, combined)
            if rel <= 0:
                continue
            scored.append(
                Correlation(
                    trigger=r["trigger"],
                    effect=r["effect"],
                    strength=float(r["strength"] or 0.5),
                    verified=bool(r["verified"]),
                    relevance_score=float(rel),
                    times_observed=int(r["times_observed"] or 1),
                )
            )
        scored.sort(key=lambda x: (x.relevance_score, x.times_observed, x.strength), reverse=True)
        return [
            {
                "trigger": c.trigger,
                "effect": c.effect,
                "strength": c.strength,
                "verified": c.verified,
                "relevance_score": round(c.relevance_score, 3),
                "times_observed": c.times_observed,
            }
            for c in scored[:top_k]
        ]

    def store_reasoning_chain(
        self,
        situation: str,
        agent_thoughts: List[str],
        tool_calls: List[Dict[str, Any]],
        final_conclusion: str,
        outcome_if_known: Optional[str] = None,
    ):
        now = datetime.now(IST).isoformat()
        with db.get_db() as conn:
            conn.execute(
                """
                INSERT INTO agent_reasoning_chains (situation, thoughts_json, tools_json, conclusion, outcome, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    situation,
                    json.dumps(agent_thoughts, default=str),
                    json.dumps(tool_calls, default=str),
                    final_conclusion,
                    outcome_if_known,
                    now,
                ),
            )

    def store_failure_pattern(
        self,
        date: str,
        predicted_regime: str,
        actual_direction: str,
        failure_type: str,
        narrative: str,
    ):
        now = datetime.now(IST).isoformat()
        with db.get_db() as conn:
            conn.execute(
                """
                INSERT INTO agent_failure_patterns (date, predicted_regime, actual_direction, failure_type, narrative, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (date, predicted_regime, actual_direction, failure_type, narrative, now),
            )

    def get_recent_failures(self, days: int = 7) -> List[Dict[str, Any]]:
        # Simple recent failures (not strict by date range; DB has dates as string)
        with db.get_db() as conn:
            rows = conn.execute(
                "SELECT date, predicted_regime, actual_direction, failure_type, narrative FROM agent_failure_patterns ORDER BY id DESC LIMIT ?",
                (max(1, days * 5),),
            ).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "date": r["date"],
                    "predicted": r["predicted_regime"],
                    "actual": r["actual_direction"],
                    "failure_type": r["failure_type"],
                    "narrative": r["narrative"],
                }
            )
        return out[:days]


# Singleton
memory = AgentMemory()

