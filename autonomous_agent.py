"""
Autonomous Agent (Trader-grade)

Design goals:
- Deterministic data collection (macro + RSS + full headline coverage) for reliability
- Claude (Anthropic) used for *reasoning + synthesis* (20-year intraday trader style)
- Memory-assisted (recent failures + correlations) to reduce repeat mistakes
- Produces a structured JSON result + stores reasoning chain

This intentionally avoids an always-on "iteration loop" for reliability and cost control.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from memory_system import memory
from macro_fetcher import fetch_all_macro_data, format_macro_context_for_gpt
from pipeline import calculate_metrics, calculate_market_stress_index, save_all
from agent import _exec_fetch_rss, _exec_analyze_batch, RSS_MAP, _log  # reuse implementations + logging

import database as db

IST = timezone(timedelta(hours=5, minutes=30))


def _anthropic_json(prompt: str) -> Dict[str, Any]:
    import re

    from anthropic import Anthropic

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
    resp = client.messages.create(
        model=model,
        max_tokens=2500,
        temperature=0.25,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text if resp.content else "{}"
    m = re.search(r"\{[\s\S]*\}", text)
    return json.loads(m.group()) if m else {"raw": text}


def _openai_json(prompt: str, model_override: str = None) -> Dict[str, Any]:
    from openai import OpenAI

    model = model_override or os.getenv("OPENAI_MODEL_AGENT", os.getenv("OPENAI_MODEL_SYNTH", "gpt-4o-mini"))
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    r = client.chat.completions.create(
        model=model,
        temperature=0.25,
        max_tokens=2500,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(r.choices[0].message.content or "{}")


def synthesize_trader_view(
    macro: Dict[str, Any],
    sectors_df: pd.DataFrame,
    top_headlines: List[Dict[str, Any]],
    recent_failures: List[Dict[str, Any]],
    similar_corrs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Claude-first synthesis. OpenAI fallback if Anthropic key missing.
    """
    macro_ctx = format_macro_context_for_gpt(macro)

    # compact sector snapshot
    sec_rows = []
    if not sectors_df.empty:
        for _, r in sectors_df.sort_values("avg_weighted_risk", ascending=False).head(12).iterrows():
            sec_rows.append(
                {
                    "sector": r.get("sector"),
                    "avg_weighted_risk": float(r.get("avg_weighted_risk", 0)),
                    "csi": float(r.get("composite_sentiment_index", 0)),
                    "velocity": float(r.get("sentiment_velocity", 0)),
                    "signal": r.get("investment_signal"),
                }
            )

    hl_rows = [
        {
            "title": h.get("title"),
            "sector": h.get("sector"),
            "sentiment": h.get("sentiment"),
            "impact": h.get("impact_score"),
            "shock": h.get("shock_status"),
            "url": h.get("url"),
        }
        for h in top_headlines[:25]
    ]

    prompt = f"""You are a 20-year experienced Indian intraday trader and market strategist.
You are NOT a generic AI. You are an AUTONOMOUS AGENT. Be ruthless, concrete, and causal.

## INSTITUTIONAL OPERATING PROTOCOLS

### TRAP_ALERT Detection
- If headlines show >70% bullish sentiment with VIX rising or crude spiking, FLAG AS TRAP.
- If 3+ sectors show sudden sentiment reversal from prior day, flag distribution risk.
- If government source contradicts market sentiment, trust government source more.

### Noise Filtration
- Ignore headlines with impact_score < 4 for regime classification.
- De-weight sectors with fewer than 3 headlines (insufficient sample).
- Flag if >50% of headlines come from a single source (source concentration risk).

### Autonomous Intelligence
- If you notice patterns NOT captured in the data (e.g., implied correlations between sectors,
  historical parallels, geopolitical spillover effects), INCLUDE THEM in your analysis.
- If macro data suggests a regime different from headline sentiment, explain the conflict.
- Identify SECOND-ORDER effects that the headline analysis may miss.
- Cross-reference sector signals for consistency — if Banking is bullish but Fintech is bearish, explain why.

### Confidence Scoring
- Be explicit about your confidence level (0-100) and list what would INVALIDATE your thesis.
- Higher confidence requires: macro alignment + headline consensus + historical pattern match.
- Never assign >85 confidence unless macro AND sentiment are unambiguously aligned.

MACRO SNAPSHOT (weight heavily — this is real money data):
{macro_ctx}

TOP SCORED SECTORS (risk/CSI/signal):
{json.dumps(sec_rows, indent=2, default=str)}

TOP HEADLINES (already scored/classified):
{json.dumps(hl_rows, indent=2, default=str)}

RECENT FAILURES (learn from these — do NOT repeat):
{json.dumps(recent_failures[:7], indent=2, default=str)}

PAST CORRELATIONS TO CONSIDER:
{json.dumps(similar_corrs[:5], indent=2, default=str)}

Output STRICT JSON:
{{
  "regime": "Risk On"|"Risk Off"|"Complacent"|"Panic",
  "regime_confidence": 0-100,
  "nifty_direction": "bullish"|"bearish"|"neutral",
  "macro_summary": "...",
  "risk_flags": ["..."],
  "trap_alerts": ["describe any detected traps"],
  "top_insight": "one paragraph — the single most actionable insight",
  "autonomous_observations": ["patterns you noticed that weren't in the data"],
  "invalidations": ["what would make this analysis wrong"],
  "sector_signals": {{
    "Banking": "BUY BIAS"|"AVOID"|"CAUTION"|"IMPROVING"|"CONTRARIAN WATCH"|"NEUTRAL",
    "IT": "...", "Energy": "...", "FMCG": "...", "Healthcare": "...",
    "Manufacturing": "...", "Fintech": "...", "Retail": "...",
    "Startup": "...", "Other": "..."
  }},
  "agent_reasoning_chain": ["step 1...", "step 2..."],
  "learning_notes": "what you learned / adjusted today based on failures/correlations",
  "noise_flags": ["any data quality concerns"]
}}
"""

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            return _anthropic_json(prompt)
        except Exception as e:
            _log(f"  Anthropic failed ({e}), falling back to OpenAI")
    return _openai_json(prompt)


def run_agent_pipeline(max_per_source: int = 14) -> Dict[str, Any]:
    selected_model = os.getenv("OPENAI_MODEL_AGENT", "gpt-4o-mini")
    _log(f"  Model: {selected_model}")

    # create pipeline run
    run_id = db.create_pipeline_run()
    _log(f"  Pipeline run #{run_id} (autonomous_agent)")

    # 1) Collect macro
    macro = fetch_all_macro_data()
    _log(f"  Macro risk score: {macro.get('macro_risk_score')} | flags: {len(macro.get('risk_flags', []))}")

    # 2) Collect RSS (full coverage)
    rss_payload = json.loads(_exec_fetch_rss(["ALL"], max_per_source=int(max_per_source)))
    rss_headlines = [h for h in rss_payload.get("headlines", []) if isinstance(h, dict) and h.get("title")]
    titles = [h["title"] for h in rss_headlines]
    _log(f"  RSS fetched: {len(titles)} headlines across {len(RSS_MAP)} sources (max_per_source={max_per_source})")

    # 3) Analyze ALL headlines in batches (alignment preserved)
    macro_context = format_macro_context_for_gpt(macro)
    analyzed_all: list[dict[str, Any]] = []
    for i in range(0, len(titles), 30):
        batch = titles[i : i + 30]
        _log(f"  Analyzing batch {i//30+1}: {len(batch)} items")
        payload = json.loads(_exec_analyze_batch(batch, macro_context))
        batch_analyzed = payload.get("analyzed", []) if isinstance(payload, dict) else []
        if not isinstance(batch_analyzed, list):
            batch_analyzed = []
        if len(batch_analyzed) < len(batch):
            batch_analyzed.extend([{}] * (len(batch) - len(batch_analyzed)))
        analyzed_all.extend(batch_analyzed[: len(batch)])
    analyzed_all = analyzed_all[: len(titles)]

    merged_rows: list[dict[str, Any]] = []
    now_pub = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
    for i, h in enumerate(rss_headlines[: len(analyzed_all)]):
        a = analyzed_all[i] if i < len(analyzed_all) else {}
        merged_rows.append(
            {
                "title": h.get("title", ""),
                "description": "",
                "source": h.get("source", ""),
                "source_url": "",
                "published": now_pub,
                "hours_old": float(h.get("hours_old", 0) or 0),
                "url": h.get("url", ""),
                "is_govt_source": bool(str(h.get("source", "")).upper() in ("PIB", "RBI", "SEBI")),
                "sector": a.get("sector", "Other"),
                "sentiment": a.get("sentiment", "neutral"),
                "sentiment_confidence": float(a.get("sentiment_confidence", 0.7) or 0.7),
                "impact_score": float(a.get("impact_score", 5) or 5),
                "valence": float(a.get("valence", 0.5) or 0.5),
                "arousal": float(a.get("arousal", 0.5) or 0.5),
                "geopolitical_risk": bool(a.get("geopolitical_risk", False)),
                "affected_companies": a.get("affected_companies", []),
                "second_order_beneficiaries": a.get("second_order_beneficiaries", []),
                "catalyst_type": a.get("catalyst_type", "other"),
                "price_direction": a.get("price_direction", "neutral"),
                "time_horizon": a.get("time_horizon", "intraday"),
                "conviction": a.get("conviction", "low"),
                "macro_sensitivity": a.get("macro_sensitivity", "medium"),
                "one_line_insight": a.get("one_line_insight", ""),
                "signal_reason": a.get("signal_reason", ""),
                "contrarian_flag": bool(a.get("contrarian_flag", False)),
                "contrarian_reason": a.get("contrarian_reason", ""),
                "source_reliability": 1.0,
            }
        )

    headlines_df = pd.DataFrame(merged_rows)
    scored_headlines_df, sector_df = calculate_metrics(headlines_df) if not headlines_df.empty else (headlines_df, pd.DataFrame())
    msi = calculate_market_stress_index(scored_headlines_df, sector_df) if not scored_headlines_df.empty else {"msi": 0, "level": "Low"}

    # save DB + GitHub snapshot (via pipeline.save_all)
    save_all(scored_headlines_df, sector_df, msi, run_id)

    # synthesize trader view
    recent_failures = memory.get_recent_failures(7)
    similar_corrs = memory.query_similar_correlations(" ".join(macro.get("risk_flags", [])) + " " + " ".join(titles[:10]), top_k=5)
    top_headlines = scored_headlines_df.sort_values("impact_score", ascending=False).to_dict(orient="records") if not scored_headlines_df.empty else []
    synthesis = synthesize_trader_view(macro, sector_df, top_headlines, recent_failures, similar_corrs)

    # store reasoning chain in memory
    memory.store_reasoning_chain(
        situation=f"Market analysis {datetime.now(IST).strftime('%Y-%m-%d')}",
        agent_thoughts=synthesis.get("agent_reasoning_chain", []) if isinstance(synthesis.get("agent_reasoning_chain"), list) else [],
        tool_calls=[{"tool": "fetch_macro_snapshot"}, {"tool": "fetch_rss_headlines"}, {"tool": "analyze_headlines_batch (batched)"}],
        final_conclusion=synthesis.get("top_insight", ""),
    )

    # save prediction (for learning_engine later)
    regime = synthesis.get("regime", "Risk Off")
    avg_nss = float(sector_df["composite_sentiment_index"].mean()) if not sector_df.empty else 0.0
    avg_risk = float(sector_df["avg_weighted_risk"].mean()) if not sector_df.empty else 0.0
    sector_signals = synthesis.get("sector_signals", {})
    today = datetime.now(IST).strftime("%Y-%m-%d")
    db.save_prediction(today, str(regime), float(avg_nss), float(avg_risk), sector_signals)

    # also dump as agent_result.json for UI
    result = {
        **synthesis,
        "timestamp": datetime.now(IST).isoformat(),
        "headlines_analyzed": int(len(scored_headlines_df)),
        "msi": msi,
        "macro": macro,
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_result.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    _log(f"  ✅ autonomous_agent complete. saved={len(scored_headlines_df)} sectors={len(sector_df)} regime={regime}")
    return result


if __name__ == "__main__":
    print(json.dumps(run_agent_pipeline(14), indent=2, default=str)[:2000])

