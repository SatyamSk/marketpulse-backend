"""
Post-market Learning Engine

Runs after market close:
- pulls today's prediction from DB
- fetches basic ground truth (Nifty move) via Yahoo Finance
- asks LLM to do a post-mortem (why right/wrong, what was missed)
- stores reflection into predictions.learning_reflection
- stores failure patterns / correlations into memory_system
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

import database as db
from memory_system import memory

IST = timezone(timedelta(hours=5, minutes=30))


def fetch_yahoo_daily(symbol: str) -> Optional[Dict[str, Any]]:
    try:
        import urllib.request
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=5d"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode())
        result = data["chart"]["result"][0]
        q = result["indicators"]["quote"][0]
        close = q.get("close") or []
        open_ = q.get("open") or []
        # latest valid index
        idx = None
        for i in range(len(close) - 1, -1, -1):
            if close[i] is not None:
                idx = i
                break
        if idx is None:
            return None
        c = float(close[idx])
        o = float(open_[idx] or c)
        change_pct = round(((c - o) / o) * 100, 2) if o else 0.0
        direction = "bullish" if change_pct > 0.15 else "bearish" if change_pct < -0.15 else "neutral"
        return {"symbol": symbol, "open": o, "close": c, "change_pct": change_pct, "direction": direction}
    except Exception:
        return None


def _llm_reflect(pred: Dict[str, Any], actual: Dict[str, Any], agent_reasoning: str = "") -> Dict[str, Any]:
    """
    Uses Anthropic if available; falls back to OpenAI.
    """
    import os
    prompt = f"""You are a 20-year experienced Indian intraday trader doing a brutal post-mortem.

PREDICTION:
{json.dumps(pred, indent=2, default=str)}

ACTUAL (ground truth proxy):
{json.dumps(actual, indent=2, default=str)}

AGENT REASONING (if available):
{agent_reasoning[:1500]}

Return strict JSON:
{{
  "was_correct": true/false,
  "accuracy_score": 0-100,
  "what_was_right": "...",
  "what_was_wrong": "...",
  "missed_signal": "...",
  "missed_correlation": {{"trigger": "...", "effect": "...", "why_missed": "..."}},
  "tomorrow_change": "one concrete procedure change",
  "confidence_calibration": "increase|decrease|maintain",
  "key_learning": "single sentence"
}}
"""

    # Anthropic preferred
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from anthropic import Anthropic

            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            resp = client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
                max_tokens=1200,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text if resp.content else ""
        except Exception as e:
            text = json.dumps({"error": f"anthropic_error: {str(e)[:120]}"})
    else:
        # OpenAI fallback
        try:
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            r = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL_REFLECT", "gpt-4o-mini"),
                temperature=0.2,
                max_tokens=900,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            text = r.choices[0].message.content or "{}"
        except Exception as e:
            text = json.dumps({"error": f"openai_error: {str(e)[:120]}"})

    # Parse JSON
    try:
        import re

        m = re.search(r"\{[\s\S]*\}", text)
        return json.loads(m.group()) if m else {"raw": text}
    except Exception:
        return {"raw": text}


def run_learning_reflection(date_str: Optional[str] = None) -> Dict[str, Any]:
    date_str = date_str or datetime.now(IST).strftime("%Y-%m-%d")

    with db.get_db() as conn:
        pred_row = conn.execute("SELECT * FROM predictions WHERE date = ?", (date_str,)).fetchone()
    if not pred_row:
        return {"error": f"No prediction found for {date_str}"}

    pred = dict(pred_row)
    # actual (proxy)
    nifty = fetch_yahoo_daily("^NSEI") or {}
    actual = {"nifty": nifty}

    was_correct = None
    if nifty:
        # coarse mapping: Risk On -> bullish, Risk Off/Panic -> bearish
        pr = (pred.get("predicted_regime") or "").lower()
        expected = "bullish" if "risk on" in pr else "bearish" if ("risk off" in pr or "panic" in pr) else "neutral"
        was_correct = expected == nifty.get("direction")

    reflection = _llm_reflect(
        pred={
            "date": pred.get("date"),
            "predicted_regime": pred.get("predicted_regime"),
            "predicted_nss": pred.get("predicted_nss"),
            "predicted_avg_risk": pred.get("predicted_avg_risk"),
            "sector_signals": pred.get("sector_signals"),
        },
        actual=actual,
        agent_reasoning="",
    )

    # Persist reflection into predictions table
    with db.get_db() as conn:
        conn.execute(
            "UPDATE predictions SET learning_reflection = ?, was_regime_correct = ? WHERE date = ?",
            (json.dumps(reflection, default=str), int(bool(was_correct)) if was_correct is not None else None, date_str),
        )

    # Store failure pattern and correlation to memory
    if reflection.get("missed_correlation"):
        mc = reflection["missed_correlation"]
        memory.store_discovered_correlation(
            trigger=str(mc.get("trigger", "")),
            effect=str(mc.get("effect", "")),
            strength=0.7,
            context={"date": date_str, "why_missed": mc.get("why_missed")},
            outcome_verified=True,
        )

    if reflection.get("was_correct") is False:
        memory.store_failure_pattern(
            date=date_str,
            predicted_regime=str(pred.get("predicted_regime", "")),
            actual_direction=str(nifty.get("direction", "")),
            failure_type="regime_mismatch",
            narrative=json.dumps(reflection, default=str)[:4000],
        )

    return {"date": date_str, "reflection": reflection, "was_correct_proxy": was_correct, "actual": actual}


if __name__ == "__main__":
    print(json.dumps(run_learning_reflection(), indent=2, default=str))

