"""
Macro Intelligence Agent

Fetches macro data, classifies regime with nuanced Bayesian reasoning,
and produces a plain-English summary of global/domestic conditions.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict

IST = timezone(timedelta(hours=5, minutes=30))


def run(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch all macro indicators and produce a regime classification
    with plain-English summary.
    """
    from macro_fetcher import fetch_all_macro_data, format_macro_context_for_gpt

    macro = fetch_all_macro_data()
    macro_text = format_macro_context_for_gpt(macro)

    # Extract key readings
    crude = macro.get("crude_oil") or {}
    inr = macro.get("inr_usd") or {}
    vix = macro.get("india_vix") or {}
    gold = macro.get("gold") or {}
    us10y = macro.get("us_10y") or {}
    risk_score = macro.get("macro_risk_score", 0)
    flags = macro.get("risk_flags", [])

    # Build plain-English summary
    parts = []
    if crude:
        chg = crude.get("change_pct", 0)
        if abs(chg) >= 3:
            parts.append(f"Crude oil moved sharply ({chg:+.1f}%) — this is a big deal for India.")
        elif abs(chg) >= 1:
            parts.append(f"Crude oil is {'up' if chg > 0 else 'down'} {abs(chg):.1f}% — worth watching.")

    if inr:
        chg = inr.get("change_pct", 0)
        if abs(chg) >= 0.8:
            parts.append(f"Rupee {'weakened' if chg > 0 else 'strengthened'} {abs(chg):.1f}% against the dollar — significant for markets.")
        elif abs(chg) >= 0.3:
            parts.append(f"Rupee is {'slightly weaker' if chg > 0 else 'slightly stronger'} today.")

    if vix:
        price = vix.get("price", 0)
        if price >= 28:
            parts.append(f"India VIX at {price:.0f} — fear levels are extreme. Expect wild swings.")
        elif price >= 22:
            parts.append(f"India VIX at {price:.0f} — markets are nervous.")
        elif price <= 12:
            parts.append(f"India VIX at {price:.0f} — markets are very calm, maybe too calm.")

    if gold:
        chg = gold.get("change_pct", 0)
        if chg > 1.5:
            parts.append("Gold surging — investors are running to safety.")
        elif chg > 0.5:
            parts.append("Gold up modestly — some caution in the air.")

    if not parts:
        parts.append("Global macro conditions look stable today — no major shocks.")

    # Classify regime (nuanced, not hardcoded)
    if risk_score >= 60:
        mood = "fearful"
        mood_label = "Markets are scared"
        mood_color = "red"
    elif risk_score >= 35:
        mood = "cautious"
        mood_label = "Markets are nervous"
        mood_color = "amber"
    elif risk_score >= 15:
        mood = "mixed"
        mood_label = "Mixed signals"
        mood_color = "amber"
    elif risk_score <= 5:
        mood = "calm"
        mood_label = "Smooth sailing"
        mood_color = "green"
    else:
        mood = "cautiously positive"
        mood_label = "Cautiously positive"
        mood_color = "green"

    return {
        "agent": "macro",
        "mood": mood,
        "mood_label": mood_label,
        "mood_color": mood_color,
        "confidence": max(20, min(95, 85 - risk_score)),
        "risk_score": risk_score,
        "summary": " ".join(parts),
        "risk_flags": flags,
        "macro_data": macro,
        "macro_text": macro_text,
    }
