"""
Historical Analogy Agent

Searches agent memory for similar past conditions.
"Have we seen something like this before? What happened next?"
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


def _build_condition_signature(macro_result: Dict, headlines: List[Dict]) -> str:
    """Build a text signature of current conditions for similarity matching."""
    parts = []

    macro_data = macro_result.get("macro_data", {})
    flags = macro_result.get("risk_flags", [])
    mood = macro_result.get("mood", "")
    parts.append(f"mood:{mood}")

    for flag in flags:
        parts.append(flag)

    # Top sectors by mention
    sector_counts: Dict[str, int] = {}
    for h in headlines:
        s = h.get("sector", "Other")
        sector_counts[s] = sector_counts.get(s, 0) + 1
    top_sectors = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for s, c in top_sectors:
        sentiment_for_sector = [h.get("sentiment", "") for h in headlines if h.get("sector") == s]
        pos = sum(1 for x in sentiment_for_sector if x == "positive")
        neg = sum(1 for x in sentiment_for_sector if x == "negative")
        lean = "positive" if pos > neg else "negative" if neg > pos else "neutral"
        parts.append(f"{s}:{lean}")

    return " ".join(parts)


def run(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search memory for historical analogies.
    """
    macro_result = context.get("macro_result", {})
    headlines = context.get("headlines", [])

    signature = _build_condition_signature(macro_result, headlines)

    # Query memory for similar past conditions
    try:
        from memory_system import memory
        similar = memory.query_similar_correlations(signature, top_k=5)
    except Exception:
        similar = []

    # Also check recent failures to avoid repeating
    try:
        from memory_system import memory
        failures = memory.get_recent_failures(14)
    except Exception:
        failures = []

    # Build summary
    parts = []
    if similar:
        top = similar[0]
        parts.append(
            f"Historical pattern found: \"{top['trigger']}\" → \"{top['effect']}\" "
            f"(seen {top.get('times_observed', 1)} times, "
            f"relevance {top.get('relevance_score', 0):.0%})."
        )
    else:
        parts.append("No strong historical analog found for today's conditions.")

    if failures:
        recent = failures[0]
        parts.append(
            f"Recent miss to learn from: predicted {recent.get('predicted', '')} "
            f"but actual was {recent.get('actual', '')} "
            f"({recent.get('failure_type', '')})."
        )

    return {
        "agent": "historical_analog",
        "condition_signature": signature,
        "similar_patterns": similar[:5],
        "recent_failures": failures[:3],
        "summary": " ".join(parts),
    }
