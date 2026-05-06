"""
Institutional Flow Agent

Estimates FII/DII positioning from proxy signals:
- INR direction (FII sell pressure proxy)
- Sector rotation patterns in headlines
- VIX + bond yield direction
"""

from __future__ import annotations

from typing import Any, Dict, List


def _estimate_fii_bias(macro_data: Dict) -> Dict[str, Any]:
    """Estimate FII bias from macro proxies."""
    inr = macro_data.get("inr_usd") or {}
    vix = macro_data.get("india_vix") or {}
    us10y = macro_data.get("us_10y") or {}
    crude = macro_data.get("crude_oil") or {}

    score = 0  # negative = FII selling, positive = FII buying
    reasons = []

    # INR weakening = FII selling pressure
    inr_chg = inr.get("change_pct", 0)
    if inr_chg > 0.5:
        score -= 2
        reasons.append(f"Rupee weakening ({inr_chg:+.1f}%) suggests FII outflows")
    elif inr_chg < -0.3:
        score += 1
        reasons.append(f"Rupee strengthening ({inr_chg:+.1f}%) hints at FII inflows")

    # High VIX = risk-off = FII may sell
    vix_price = vix.get("price", 15)
    if vix_price > 22:
        score -= 2
        reasons.append(f"High VIX ({vix_price:.0f}) indicates risk-off — FIIs likely reducing exposure")
    elif vix_price < 13:
        score += 1
        reasons.append(f"Low VIX ({vix_price:.0f}) — calm conditions favor FII allocation")

    # US 10Y rising = money goes back to US
    us10y_price = us10y.get("price", 4)
    us10y_chg = us10y.get("change_pct", 0)
    if us10y_price > 4.8 or us10y_chg > 2:
        score -= 1
        reasons.append("US yields elevated — capital may prefer US over EM")

    # Crude spiking = India's current account worsens = FII negative
    crude_chg = crude.get("change_pct", 0)
    if crude_chg > 3:
        score -= 1
        reasons.append("Crude spike worsens India's current account — FII headwind")

    if score <= -3:
        bias = "strong selling"
    elif score <= -1:
        bias = "mild selling"
    elif score >= 3:
        bias = "strong buying"
    elif score >= 1:
        bias = "mild buying"
    else:
        bias = "neutral"

    return {"bias": bias, "score": score, "reasons": reasons}


def _detect_sector_rotation(headlines: List[Dict]) -> Dict[str, Any]:
    """
    Detect sector rotation from headline patterns.
    If defensive sectors (FMCG, Pharma) are getting positive coverage
    while cyclicals (Banking, IT, Auto) are negative — rotation to safety.
    """
    defensive = {"FMCG", "Healthcare"}
    cyclical = {"Banking", "IT", "Manufacturing", "Energy", "Fintech"}

    def_score = 0
    cyc_score = 0

    for h in headlines:
        sector = h.get("sector", "Other")
        sent = h.get("sentiment", "neutral")
        val = 1 if sent == "positive" else -1 if sent == "negative" else 0

        if sector in defensive:
            def_score += val
        elif sector in cyclical:
            cyc_score += val

    if def_score > 2 and cyc_score < -2:
        return {
            "rotation": "risk_off",
            "summary": "Headlines show rotation into defensive sectors (FMCG, Pharma) and away from cyclicals — classic risk-off behavior.",
        }
    elif cyc_score > 2 and def_score < 0:
        return {
            "rotation": "risk_on",
            "summary": "Cyclical sectors (Banking, IT) dominating positive headlines — risk appetite is healthy.",
        }
    return {
        "rotation": "none",
        "summary": "No clear sector rotation pattern in today's headlines.",
    }


def run(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Estimate institutional flows and sector rotation.
    """
    macro_data = context.get("macro_result", {}).get("macro_data", {})
    headlines = context.get("headlines", [])

    fii = _estimate_fii_bias(macro_data)
    rotation = _detect_sector_rotation(headlines)

    parts = []
    if fii["bias"] in ("strong selling", "mild selling"):
        parts.append(f"FII positioning looks like {fii['bias']}.")
    elif fii["bias"] in ("strong buying", "mild buying"):
        parts.append(f"FII positioning looks like {fii['bias']}.")
    else:
        parts.append("No strong read on FII flows today.")

    if rotation["rotation"] != "none":
        parts.append(rotation["summary"])

    return {
        "agent": "flow",
        "fii_estimate": fii,
        "sector_rotation": rotation,
        "summary": " ".join(parts),
    }
