"""
Self-Evaluation Agent

After every prediction: was I correct? Was timing right?
Was confidence justified? Feeds into the Bayesian weight system.
"""

from __future__ import annotations

from typing import Any, Dict


def run(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Self-evaluate the system's recent performance and calibrate confidence.
    """
    # Get recent accuracy stats
    try:
        from database import get_accuracy_stats
        stats_7 = get_accuracy_stats(7)
        stats_30 = get_accuracy_stats(30)
    except Exception:
        stats_7 = {"total": 0, "correct": 0, "accuracy": 0}
        stats_30 = {"total": 0, "correct": 0, "accuracy": 0}

    # Get recent failures
    failures = context.get("historical_result", {}).get("recent_failures", [])

    # Calculate confidence adjustment
    confidence_adjustment = 0
    reasons = []

    # If recent accuracy is low, reduce confidence
    if stats_7.get("total", 0) >= 3:
        acc_7 = stats_7.get("accuracy", 50)
        if acc_7 < 40:
            confidence_adjustment -= 15
            reasons.append(f"Last 7 days accuracy is only {acc_7}% — reducing confidence significantly.")
        elif acc_7 < 60:
            confidence_adjustment -= 5
            reasons.append(f"Last 7 days accuracy at {acc_7}% — slight confidence reduction.")
        elif acc_7 > 80:
            confidence_adjustment += 5
            reasons.append(f"Last 7 days accuracy at {acc_7}% — system is calibrated well.")

    # If contradiction agent found serious issues, reduce confidence
    contradiction_penalty = context.get("contradiction_result", {}).get("confidence_penalty", 0)
    if contradiction_penalty > 15:
        confidence_adjustment -= contradiction_penalty
        reasons.append(f"Contradiction analysis suggests reducing confidence by {contradiction_penalty}%.")

    # If signal quality is low, reduce confidence
    signal_quality = context.get("noise_result", {}).get("signal_quality", {}).get("quality", 50)
    if signal_quality < 30:
        confidence_adjustment -= 10
        reasons.append("Low signal quality today — data is mostly noise.")

    # Build summary
    parts = []
    if stats_7.get("total", 0) > 0:
        parts.append(
            f"System accuracy: {stats_7.get('accuracy', 0)}% over last 7 days "
            f"({stats_7.get('correct', 0)}/{stats_7.get('total', 0)} correct)."
        )
    if stats_30.get("total", 0) > 0:
        parts.append(f"30-day accuracy: {stats_30.get('accuracy', 0)}%.")

    if failures:
        parts.append(f"Learning from {len(failures)} recent misses.")

    if confidence_adjustment != 0:
        direction = "reducing" if confidence_adjustment < 0 else "boosting"
        parts.append(f"Net confidence adjustment: {direction} by {abs(confidence_adjustment)}%.")
    else:
        parts.append("No confidence adjustment needed.")

    return {
        "agent": "self_eval",
        "accuracy_7d": stats_7,
        "accuracy_30d": stats_30,
        "confidence_adjustment": confidence_adjustment,
        "adjustment_reasons": reasons,
        "summary": " ".join(parts) if parts else "No historical data to evaluate against yet.",
    }
