"""
Noise Filter Agent

Most market information is garbage. This agent's job:
- Filter duplicate/near-duplicate narratives
- Detect source concentration (>50% from one source)
- Flag low-quality signals
- Compute signal-to-noise ratio
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List


def _tokenize(text: str) -> set:
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return {t for t in text.split() if len(t) >= 4}


def _near_duplicate_groups(headlines: List[Dict]) -> List[List[int]]:
    """Find groups of near-duplicate headlines (>60% token overlap)."""
    groups = []
    assigned = set()

    for i, h1 in enumerate(headlines):
        if i in assigned:
            continue
        group = [i]
        t1 = _tokenize(h1.get("title", ""))
        if not t1:
            continue
        for j, h2 in enumerate(headlines[i + 1:], start=i + 1):
            if j in assigned:
                continue
            t2 = _tokenize(h2.get("title", ""))
            if not t2:
                continue
            overlap = len(t1 & t2) / max(len(t1 | t2), 1)
            if overlap > 0.6:
                group.append(j)
                assigned.add(j)
        if len(group) > 1:
            groups.append(group)
            assigned.update(group)

    return groups


def _check_source_concentration(headlines: List[Dict]) -> Dict[str, Any]:
    """Check if too many headlines come from one source."""
    if not headlines:
        return {"concentrated": False}

    sources = [h.get("source", "unknown") for h in headlines]
    counter = Counter(sources)
    total = len(headlines)

    top_source, top_count = counter.most_common(1)[0]
    concentration_pct = round(top_count / total * 100, 1)

    if concentration_pct > 50:
        return {
            "concentrated": True,
            "top_source": top_source,
            "pct": concentration_pct,
            "warning": f"{concentration_pct:.0f}% of headlines come from {top_source}. This creates a narrative bias — cross-check with other sources.",
        }
    return {"concentrated": False, "top_source": top_source, "pct": concentration_pct}


def _compute_signal_quality(headlines: List[Dict]) -> Dict[str, Any]:
    """Overall signal quality score."""
    if not headlines:
        return {"quality": 0, "label": "no data"}

    # High-impact, high-confidence headlines = good signal
    high_impact = sum(1 for h in headlines if float(h.get("impact_score", 0)) >= 7)
    govt_sources = sum(1 for h in headlines if h.get("is_govt_source"))
    low_impact = sum(1 for h in headlines if float(h.get("impact_score", 0)) <= 3)
    total = len(headlines)

    signal_ratio = (high_impact + govt_sources * 0.5) / max(total, 1) * 100
    noise_ratio = low_impact / max(total, 1) * 100

    quality = min(100, max(0, int(signal_ratio * 1.5 - noise_ratio * 0.5)))

    if quality >= 70:
        label = "high"
    elif quality >= 40:
        label = "medium"
    else:
        label = "low"

    return {
        "quality": quality,
        "label": label,
        "high_impact_count": high_impact,
        "low_impact_count": low_impact,
        "govt_source_count": govt_sources,
    }


def run(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter noise and assess data quality.
    """
    headlines = context.get("headlines", [])

    dup_groups = _near_duplicate_groups(headlines)
    source_conc = _check_source_concentration(headlines)
    quality = _compute_signal_quality(headlines)

    duplicate_count = sum(len(g) - 1 for g in dup_groups)  # extra copies

    # Build summary
    parts = []
    if duplicate_count > 3:
        parts.append(f"Filtered {duplicate_count} near-duplicate headlines — same story, different sources.")
    if source_conc.get("concentrated"):
        parts.append(source_conc["warning"])

    if quality["label"] == "high":
        parts.append(f"Signal quality is strong today — {quality['high_impact_count']} high-impact headlines and {quality['govt_source_count']} verified government sources.")
    elif quality["label"] == "low":
        parts.append("Signal quality is weak today — mostly low-impact noise. Take the analysis with a grain of salt.")
    else:
        parts.append(f"Signal quality is decent — {quality['high_impact_count']} meaningful headlines out of {len(headlines)} total.")

    return {
        "agent": "noise_filter",
        "duplicate_groups": len(dup_groups),
        "duplicates_filtered": duplicate_count,
        "source_concentration": source_conc,
        "signal_quality": quality,
        "summary": " ".join(parts),
    }
