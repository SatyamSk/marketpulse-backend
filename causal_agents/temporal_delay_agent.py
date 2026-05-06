"""
Temporal Delay Agent

Identifies delayed market effects that haven't been priced yet.
The key insight: some events take days, weeks, or months to fully propagate.

"Railway capex announced today → transformer demand rises after months
→ copper demand rises later → small electrical equipment firms haven't moved yet."
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List


# Impact timelines by catalyst type
DELAY_MAP = {
    "policy_change": {"immediate": True, "days": True, "weeks": True, "months": True},
    "government_contract": {"immediate": True, "weeks": True, "months": True},
    "pib_announcement": {"immediate": True, "weeks": True},
    "rbi_action": {"immediate": True, "days": True, "weeks": True, "months": True},
    "sebi_action": {"immediate": True, "days": True},
    "capex_announcement": {"immediate": True, "months": True},
    "earnings": {"immediate": True, "days": True},
    "global_event": {"immediate": True, "days": True, "weeks": True},
    "fii_flow": {"immediate": True, "days": True},
    "supply_chain_disruption": {"days": True, "weeks": True, "months": True},
}


def _find_unpriced_effects(
    headlines: List[Dict],
    delayed_effects: List[Dict],
) -> List[Dict[str, Any]]:
    """
    From supply chain agent's delayed effects, identify which
    downstream entities haven't appeared in headlines yet (= not priced).
    """
    # Collect all entities/sectors mentioned in headlines
    mentioned = set()
    for h in headlines:
        mentioned.add(h.get("sector", "").lower())
        for comp in h.get("affected_companies", []) or []:
            if isinstance(comp, str):
                mentioned.add(comp.lower())

    unpriced = []
    for deff in delayed_effects:
        chain = deff.get("chain", [])
        for step in chain:
            entity = step.get("entity", "")
            delay = step.get("delay", "immediate")
            effect = step.get("effect", "")

            # If this entity is NOT mentioned in today's headlines
            # and the delay is > immediate, it's potentially unpriced
            if delay != "immediate" and entity.lower() not in mentioned:
                unpriced.append({
                    "trigger": deff.get("trigger", ""),
                    "entity": entity,
                    "expected_delay": delay,
                    "expected_effect": effect,
                    "is_mentioned": False,
                    "plain": f"{deff.get('trigger', '').replace('_', ' ').title()} → {effect} (expected in {delay}). Market hasn't connected this yet.",
                })

    return unpriced


def _categorize_by_timeline(headlines: List[Dict]) -> Dict[str, List[Dict]]:
    """Categorize headline effects by their time horizon."""
    categories = {
        "today": [],     # will play out today
        "this_week": [], # next few days
        "delayed": [],   # weeks or months
    }

    for h in headlines:
        horizon = h.get("time_horizon", "intraday")
        catalyst = h.get("catalyst_type", "other")
        impact = float(h.get("impact_score", 5))

        entry = {
            "title": h.get("title", ""),
            "sector": h.get("sector", "Other"),
            "impact": impact,
            "catalyst": catalyst,
        }

        if horizon == "intraday" or (impact >= 7 and catalyst in ("earnings", "fii_flow")):
            categories["today"].append(entry)
        elif horizon == "swing_2_5days":
            categories["this_week"].append(entry)
        else:
            categories["delayed"].append(entry)

    return categories


def run(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Identify delayed effects and unpriced opportunities.
    """
    headlines = context.get("headlines", [])
    delayed_effects = context.get("supply_chain_result", {}).get("delayed_effects", [])

    unpriced = _find_unpriced_effects(headlines, delayed_effects)
    timeline = _categorize_by_timeline(headlines)

    # Build summary
    parts = []
    if unpriced:
        top = unpriced[0]
        parts.append(top["plain"])
        if len(unpriced) > 1:
            parts.append(f"Plus {len(unpriced) - 1} more hidden connections the market hasn't made yet.")
    else:
        parts.append("No obvious delayed effects detected today.")

    delayed_count = len(timeline.get("delayed", []))
    if delayed_count > 0:
        parts.append(f"{delayed_count} headlines have effects that will play out over weeks, not today.")

    return {
        "agent": "temporal_delay",
        "unpriced_effects": unpriced[:8],
        "timeline": {k: len(v) for k, v in timeline.items()},
        "timeline_detail": timeline,
        "summary": " ".join(parts),
    }
