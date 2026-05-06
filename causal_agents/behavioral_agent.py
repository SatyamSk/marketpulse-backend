"""
Behavioral Psychology Agent

Detects crowd vs. smart money divergence, identifies traps,
FOMO patterns, and retail euphoria/panic.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _analyze_sentiment_distribution(headlines: List[Dict]) -> Dict[str, Any]:
    """Analyze the distribution pattern of sentiments."""
    if not headlines:
        return {"positive_pct": 0, "negative_pct": 0, "neutral_pct": 0, "total": 0}

    total = len(headlines)
    pos = sum(1 for h in headlines if h.get("sentiment") == "positive")
    neg = sum(1 for h in headlines if h.get("sentiment") == "negative")
    neu = total - pos - neg

    return {
        "positive_pct": round(pos / total * 100, 1) if total else 0,
        "negative_pct": round(neg / total * 100, 1) if total else 0,
        "neutral_pct": round(neu / total * 100, 1) if total else 0,
        "total": total,
    }


def _detect_trap(
    sentiment_dist: Dict[str, Any],
    macro_mood: str,
    vix_level: float,
    crude_change: float,
) -> Dict[str, Any]:
    """
    Detect sentiment traps:
    - Overwhelmingly bullish headlines + macro stress = BULL TRAP
    - Overwhelmingly bearish headlines + macro calm   = BEAR TRAP (contrarian buy)
    """
    pos_pct = sentiment_dist["positive_pct"]
    neg_pct = sentiment_dist["negative_pct"]
    trap = {"detected": False, "type": None, "reason": "", "probability": 0}

    # Bull trap: headlines euphoric but macro says danger
    if pos_pct > 70 and (vix_level > 20 or crude_change > 2 or macro_mood in ("fearful", "cautious")):
        trap = {
            "detected": True,
            "type": "bull_trap",
            "reason": (
                f"Headlines are {pos_pct:.0f}% positive, but macro signals "
                f"are stressed (VIX {vix_level:.0f}, crude {crude_change:+.1f}%). "
                "This looks like retail enthusiasm walking into institutional selling."
            ),
            "probability": min(85, int(pos_pct * 0.6 + vix_level * 1.5)),
        }
    # Bear trap: headlines fearful but macro is actually fine
    elif neg_pct > 60 and vix_level < 16 and abs(crude_change) < 1.5 and macro_mood in ("calm", "cautiously positive"):
        trap = {
            "detected": True,
            "type": "bear_trap",
            "reason": (
                f"Headlines are {neg_pct:.0f}% negative, but macro is calm "
                f"(VIX {vix_level:.0f}). This fear may be overdone. "
                "Smart money could be accumulating."
            ),
            "probability": min(75, int(neg_pct * 0.5 + (20 - vix_level) * 2)),
        }

    return trap


def _detect_source_divergence(headlines: List[Dict]) -> Dict[str, Any]:
    """
    Check if government/institutional sources disagree with market sources.
    Govt sources being negative while market headlines are positive = warning.
    """
    govt_headlines = [h for h in headlines if h.get("is_govt_source")]
    market_headlines = [h for h in headlines if not h.get("is_govt_source")]

    if not govt_headlines or not market_headlines:
        return {"divergence": False}

    govt_sentiment = sum(1 if h.get("sentiment") == "positive" else -1 if h.get("sentiment") == "negative" else 0 for h in govt_headlines)
    market_sentiment = sum(1 if h.get("sentiment") == "positive" else -1 if h.get("sentiment") == "negative" else 0 for h in market_headlines)

    govt_net = govt_sentiment / len(govt_headlines) if govt_headlines else 0
    market_net = market_sentiment / len(market_headlines) if market_headlines else 0

    if (govt_net < -0.3 and market_net > 0.3) or (govt_net > 0.3 and market_net < -0.3):
        return {
            "divergence": True,
            "govt_lean": "negative" if govt_net < 0 else "positive",
            "market_lean": "negative" if market_net < 0 else "positive",
            "warning": (
                f"Government sources lean {'negative' if govt_net < 0 else 'positive'} "
                f"while market headlines lean {'negative' if market_net < 0 else 'positive'}. "
                "Trust government sources more — they have information markets don't."
            ),
        }

    return {"divergence": False}


def _detect_sector_euphoria(headlines: List[Dict]) -> List[Dict[str, Any]]:
    """
    Detect sectors with unusually one-sided sentiment.
    >80% positive in a sector = potential euphoria / blow-off top risk.
    """
    sector_sentiments: Dict[str, List[str]] = {}
    for h in headlines:
        sector = h.get("sector", "Other")
        sentiment = h.get("sentiment", "neutral")
        sector_sentiments.setdefault(sector, []).append(sentiment)

    euphoric = []
    for sector, sents in sector_sentiments.items():
        if len(sents) < 3:
            continue
        pos_pct = sum(1 for s in sents if s == "positive") / len(sents) * 100
        if pos_pct > 80:
            euphoric.append({
                "sector": sector,
                "positive_pct": round(pos_pct, 0),
                "headline_count": len(sents),
                "warning": f"{sector} has {pos_pct:.0f}% positive headlines — when everyone agrees, be careful.",
            })

    return euphoric


def run(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect behavioral patterns: traps, divergence, euphoria.
    """
    headlines = context.get("headlines", [])
    macro_result = context.get("macro_result", {})

    macro_mood = macro_result.get("mood", "mixed")
    macro_data = macro_result.get("macro_data", {})
    vix = macro_data.get("india_vix", {})
    crude = macro_data.get("crude_oil", {})
    vix_level = vix.get("price", 15) if vix else 15
    crude_change = crude.get("change_pct", 0) if crude else 0

    sentiment_dist = _analyze_sentiment_distribution(headlines)
    trap = _detect_trap(sentiment_dist, macro_mood, vix_level, crude_change)
    source_div = _detect_source_divergence(headlines)
    sector_euphoria = _detect_sector_euphoria(headlines)

    # Build plain-English summary
    parts = []
    if trap["detected"]:
        parts.append(trap["reason"])
    if source_div.get("divergence"):
        parts.append(source_div["warning"])
    for se in sector_euphoria[:2]:
        parts.append(se["warning"])

    if not parts:
        crowd_label = "balanced"
        if sentiment_dist["positive_pct"] > 60:
            crowd_label = "optimistic"
        elif sentiment_dist["negative_pct"] > 60:
            crowd_label = "pessimistic"
        parts.append(f"Market crowd sentiment is {crowd_label} — no obvious traps detected.")

    return {
        "agent": "behavioral",
        "sentiment_distribution": sentiment_dist,
        "trap": trap,
        "source_divergence": source_div,
        "sector_euphoria": sector_euphoria,
        "summary": " ".join(parts),
    }
