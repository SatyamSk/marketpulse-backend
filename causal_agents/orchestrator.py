"""
CausalEdge AI — Agent Orchestrator

The brain that coordinates 9 specialized agents + a meta-supervisor
that can spawn new specialist agents when it detects blind spots.
Produces two outputs:
  1. simple_thesis  — Plain-English for the "Today" page
  2. full_analysis   — Complete structured data for the deep dive
"""

from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from agent import _exec_fetch_rss, _exec_analyze_batch, RSS_MAP, _log
from macro_fetcher import format_macro_context_for_gpt
from pipeline import calculate_metrics, calculate_market_stress_index, save_all
from memory_system import memory
import database as db

IST = timezone(timedelta(hours=5, minutes=30))

# ── Structured Event Emitter (Claude/Gemini-style thinking) ────────

AGENT_DESCRIPTIONS = {
    "macro": ("Macro Intelligence", "Analyzes crude oil, VIX, INR, gold, and US10Y to classify the market regime"),
    "headlines": ("Headline Collection", "Fetches and AI-classifies headlines from 16 news sources"),
    "supply_chain": ("Supply Chain Graph", "Traces causal chains through 50+ entities in the knowledge graph"),
    "behavioral": ("Behavioral Psychology", "Detects bull/bear traps, FOMO patterns, and crowd vs. smart money divergence"),
    "flow": ("Institutional Flow", "Analyzes FII/DII flow patterns and institutional positioning"),
    "historical_analog": ("Historical Analogy", "Pattern-matches current conditions against past market episodes"),
    "temporal_delay": ("Temporal Delay", "Identifies effects that haven't hit markets yet — delayed propagation"),
    "noise_filter": ("Noise Filter", "Separates signal from noise — removes low-quality or recycled headlines"),
    "contradiction": ("Contradiction Detector", "Uses AI to find logical contradictions in the overall analysis"),
    "self_eval": ("Self-Evaluation", "Calibrates confidence and checks for overfit or blind spots"),
    "meta_supervisor": ("Meta-Supervisor", "Self-evolving intelligence — spawns new agents, runs anti-delusion checks"),
    "synthesis": ("Synthesis", "Combines all 10 agents into a unified plain-English thesis"),
}


def _emit_event(event_type: str, agent_key: str, step: int = 0, total: int = 12,
                message: str = "", confidence: Optional[float] = None,
                duration_ms: Optional[int] = None, data: Optional[Dict] = None):
    """
    Emit a structured thinking event. The SSE stream picks these up.
    Events are written to a shared file that the SSE endpoint reads.
    """
    name, desc = AGENT_DESCRIPTIONS.get(agent_key, (agent_key, ""))
    event = {
        "type": event_type,  # agent_start | agent_thinking | agent_done | agent_error | pipeline_done
        "agent": agent_key,
        "agent_name": name,
        "agent_description": desc,
        "step": step,
        "total_steps": total,
        "message": message,
        "confidence": confidence,
        "duration_ms": duration_ms,
        "data": data or {},
        "timestamp": datetime.now(IST).isoformat(),
    }
    # Write to the event log file (SSE endpoint reads this)
    event_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline_events.jsonl")
    try:
        with open(event_path, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except Exception:
        pass
    # Also emit to the existing log for backward compat
    prefix = f"[{step}/{total}]" if step > 0 else ""
    _log(f"  {prefix} {name}: {message}")


def _safe_run_agent(agent_module, context: Dict, agent_key: str,
                    step: int = 0, total: int = 12) -> Dict[str, Any]:
    """Run an agent with error handling and structured event emission."""
    _emit_event("agent_start", agent_key, step, total,
                message=f"Starting {AGENT_DESCRIPTIONS.get(agent_key, (agent_key, ''))[0]}...")
    t0 = time.time()
    try:
        result = agent_module.run(context)
        dur = int((time.time() - t0) * 1000)
        summary = result.get("summary", "")[:200]
        conf = result.get("confidence")
        _emit_event("agent_done", agent_key, step, total,
                    message=summary, confidence=conf, duration_ms=dur)
        return result
    except Exception as e:
        dur = int((time.time() - t0) * 1000)
        _emit_event("agent_error", agent_key, step, total,
                    message=f"Failed: {str(e)[:150]}", duration_ms=dur)
        return {
            "agent": agent_key,
            "summary": f"Agent failed: {str(e)[:100]}",
            "error": str(e),
        }



def _synthesize_narrative(
    macro: Dict, supply: Dict, behavioral: Dict, flow: Dict,
    historical: Dict, temporal: Dict, noise: Dict,
    contradiction: Dict, self_eval: Dict,
    sectors_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Synthesize all agent outputs into a unified narrative.
    Uses LLM for the final plain-English synthesis.
    """
    # Calculate base confidence
    base_confidence = macro.get("confidence", 60)
    penalty = contradiction.get("confidence_penalty", 0)
    adjustment = self_eval.get("confidence_adjustment", 0)
    final_confidence = max(10, min(95, base_confidence - penalty + adjustment))

    # Determine overall mood
    mood = macro.get("mood", "mixed")
    mood_color = macro.get("mood_color", "amber")

    # If behavioral agent detected a trap, override mood
    trap = behavioral.get("trap", {})
    if trap.get("detected") and trap.get("type") == "bull_trap":
        mood = "deceptively positive"
        mood_color = "amber"
        final_confidence = min(final_confidence, 55)
    elif trap.get("detected") and trap.get("type") == "bear_trap":
        mood = "oversold — contrarian opportunity"
        mood_color = "amber"

    # Build drivers list (the "What's Driving This" section)
    drivers = []

    # From supply chain
    for b in supply.get("beneficiaries", [])[:3]:
        drivers.append({
            "icon": "✅",
            "headline": b.get("reason", ""),
            "impact": "positive",
            "entity": b.get("entity", ""),
        })
    for v in supply.get("victims", [])[:2]:
        drivers.append({
            "icon": "⚠️",
            "headline": v.get("reason", ""),
            "impact": "negative",
            "entity": v.get("entity", ""),
        })

    # From macro
    for flag in macro.get("risk_flags", [])[:2]:
        drivers.append({
            "icon": "🔴" if "PANIC" in flag or "CRISIS" in flag else "⚠️",
            "headline": flag,
            "impact": "negative",
            "entity": "macro",
        })

    # Hidden signals from temporal delay
    hidden_signals = []
    for eff in temporal.get("unpriced_effects", [])[:3]:
        hidden_signals.append({
            "icon": "🔍",
            "text": eff.get("plain", ""),
            "entity": eff.get("entity", ""),
            "expected_delay": eff.get("expected_delay", ""),
        })

    # Behavioral hidden signals
    if trap.get("detected"):
        hidden_signals.append({
            "icon": "🎯",
            "text": trap.get("reason", ""),
            "entity": "behavioral",
            "expected_delay": "now",
        })

    # What could go wrong (from contradiction agent)
    risks = []
    if contradiction.get("plain_warning"):
        risks.append(contradiction["plain_warning"])
    for cond in contradiction.get("invalidation_conditions", [])[:2]:
        risks.append(cond)

    # Sector bars — simplified for the "Today" page
    sector_bars = []
    if not sectors_df.empty:
        for _, row in sectors_df.sort_values("composite_sentiment_index", ascending=False).iterrows():
            csi = float(row.get("composite_sentiment_index", 0))
            risk = float(row.get("avg_weighted_risk", 0))
            signal = row.get("investment_signal", "NEUTRAL")

            # Simple score 0-100 from CSI and risk
            score = max(0, min(100, int(50 + csi * 0.5 - risk * 0.3)))

            if signal == "BUY BIAS":
                label = "Strong"
            elif signal == "IMPROVING":
                label = "Good"
            elif signal == "AVOID":
                label = "Risky"
            elif signal == "CAUTION":
                label = "Careful"
            elif signal == "CONTRARIAN WATCH":
                label = "Contrarian"
            else:
                label = "Neutral"

            direction = "up" if csi > 10 else "down" if csi < -10 else "flat"
            if signal in ("CAUTION", "CONTRARIAN WATCH"):
                direction = "warning"

            sector_bars.append({
                "name": row.get("sector", ""),
                "score": score,
                "label": label,
                "direction": direction,
                "signal": signal,
                "csi": round(csi, 1),
                "risk": round(risk, 1),
            })

    # Now build the narrative using LLM
    narrative = _generate_narrative(
        macro, supply, behavioral, flow, contradiction,
        temporal, final_confidence, mood,
    )

    return {
        "mood": mood,
        "mood_color": mood_color,
        "confidence": final_confidence,
        "narrative": narrative,
        "drivers": drivers[:6],
        "sectors": sector_bars,
        "hidden_signals": hidden_signals[:4],
        "what_could_go_wrong": risks[:4],
        "invalidation": contradiction.get("invalidation_conditions", [""])[0] if contradiction.get("invalidation_conditions") else "",
        "signal_quality": noise.get("signal_quality", {}).get("label", "medium"),
    }


def _generate_narrative(
    macro: Dict, supply: Dict, behavioral: Dict, flow: Dict,
    contradiction: Dict, temporal: Dict,
    confidence: int, mood: str,
) -> str:
    """
    Generate the plain-English narrative for the Today page.
    3-5 sentences. No jargon. Like a smart friend telling you what's up.
    """
    prompt = f"""You are writing a brief market intelligence summary for someone who trades Indian stocks.

Write 3-5 sentences. Use plain English — NO financial jargon, no abbreviations like CSI or NSS.
Write like a smart, experienced friend sending a morning message. Be specific and actionable.

DATA:
- Macro mood: {mood} (confidence: {confidence}%)
- Macro summary: {macro.get('summary', '')}
- Supply chain: {supply.get('summary', '')}
- Behavioral: {behavioral.get('summary', '')}
- Flow: {flow.get('summary', '')}
- Contradiction: {contradiction.get('summary', '')}
- Hidden effects: {temporal.get('summary', '')}

Write ONLY the 3-5 sentence narrative. No headers, no bullet points, no JSON. Just the text."""

    try:
        if os.getenv("ANTHROPIC_API_KEY"):
            from anthropic import Anthropic
            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            resp = client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
                max_tokens=400,
                temperature=0.35,
                messages=[{"role": "user", "content": prompt}],
            )
            return (resp.content[0].text if resp.content else "").strip()
        else:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model = os.getenv("OPENAI_MODEL_AGENT", "gpt-4o-mini")
            r = client.chat.completions.create(
                model=model,
                temperature=0.35,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            return (r.choices[0].message.content or "").strip()
    except Exception as e:
        # Fallback: concatenate agent summaries
        parts = [macro.get("summary", ""), supply.get("summary", "")]
        if behavioral.get("trap", {}).get("detected"):
            parts.append(behavioral.get("summary", ""))
        return " ".join(p for p in parts if p)


def run_causal_pipeline(max_per_source: int = 14) -> Dict[str, Any]:
    """
    Run the full 9-agent causal reasoning pipeline.

    Returns a dict with:
      - "today": simplified output for the Today page
      - "full_analysis": detailed output for the Full Analysis page
      - All existing fields for backward compatibility with the old pipeline
    """
    now = datetime.now(IST)
    selected_model = os.getenv("OPENAI_MODEL_AGENT", "gpt-4o-mini")
    TOTAL = 12  # total pipeline steps

    # Clear previous events
    event_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline_events.jsonl")
    try:
        open(event_path, "w").close()
    except Exception:
        pass

    _emit_event("pipeline_start", "pipeline", 0, TOTAL,
                message=f"Starting CausalEdge AI pipeline · Model: {selected_model}")

    run_id = db.create_pipeline_run()

    # ── Agent 1: Macro Intelligence ──────────────────────────────
    from causal_agents import macro_agent
    macro_result = _safe_run_agent(macro_agent, {}, "macro", step=1, total=TOTAL)

    # ── Collect Headlines ─────────────────────────────────────────
    _emit_event("agent_start", "headlines", 2, TOTAL,
                message=f"Fetching headlines from {len(RSS_MAP)} sources...")
    t0 = time.time()
    rss_payload = json.loads(_exec_fetch_rss(["ALL"], max_per_source=int(max_per_source)))
    rss_headlines = [h for h in rss_payload.get("headlines", []) if isinstance(h, dict) and h.get("title")]
    titles = [h["title"] for h in rss_headlines]
    _emit_event("agent_thinking", "headlines", 2, TOTAL,
                message=f"Fetched {len(titles)} headlines · now AI-classifying each one...")


    # Analyze headlines in batches
    macro_text = format_macro_context_for_gpt(macro_result.get("macro_data", {}))
    analyzed_all: list = []
    for i in range(0, len(titles), 30):
        batch = titles[i:i + 30]
        _emit_event("agent_thinking", "headlines", 2, TOTAL,
                    message=f"Classifying batch {i // 30 + 1}: {len(batch)} headlines (sector, sentiment, impact, catalyst)...")
        payload = json.loads(_exec_analyze_batch(batch, macro_text))
        batch_analyzed = payload.get("analyzed", []) if isinstance(payload, dict) else []
        if not isinstance(batch_analyzed, list):
            batch_analyzed = []
        if len(batch_analyzed) < len(batch):
            batch_analyzed.extend([{}] * (len(batch) - len(batch_analyzed)))
        analyzed_all.extend(batch_analyzed[:len(batch)])
    analyzed_all = analyzed_all[:len(titles)]
    dur = int((time.time() - t0) * 1000)
    _emit_event("agent_done", "headlines", 2, TOTAL,
                message=f"Classified {len(titles)} headlines across {len(RSS_MAP)} sources",
                duration_ms=dur)

    # Merge analyzed results onto raw RSS headlines
    now_pub = now.strftime("%Y-%m-%d %H:%M:%S IST")
    merged_rows = []
    for i, h in enumerate(rss_headlines[:len(analyzed_all)]):
        a = analyzed_all[i] if i < len(analyzed_all) else {}
        merged_rows.append({
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
        })

    headlines_df = pd.DataFrame(merged_rows)
    scored_df, sector_df = calculate_metrics(headlines_df) if not headlines_df.empty else (headlines_df, pd.DataFrame())
    msi = calculate_market_stress_index(scored_df, sector_df) if not scored_df.empty else {"msi": 0, "level": "Low"}

    # Save to DB (backward compatible)
    save_all(scored_df, sector_df, msi, run_id)

    # Build context for remaining agents
    context = {
        "headlines": merged_rows,
        "macro_result": macro_result,
        "sector_data": sector_df.to_dict(orient="records") if not sector_df.empty else [],
        "scored_headlines": scored_df.to_dict(orient="records") if not scored_df.empty else [],
    }

    # ── Agent 2: Supply Chain ────────────────────────────────────
    from causal_agents import supply_chain_agent
    supply_result = _safe_run_agent(supply_chain_agent, context, "supply_chain", step=3, total=TOTAL)
    context["supply_chain_result"] = supply_result

    # ── Agent 3: Behavioral Psychology ───────────────────────────
    from causal_agents import behavioral_agent
    behavioral_result = _safe_run_agent(behavioral_agent, context, "behavioral", step=4, total=TOTAL)
    context["behavioral_result"] = behavioral_result

    # ── Agent 4: Institutional Flow ──────────────────────────────
    from causal_agents import flow_agent
    flow_result = _safe_run_agent(flow_agent, context, "flow", step=5, total=TOTAL)
    context["flow_result"] = flow_result

    # ── Agent 5: Historical Analogy ──────────────────────────────
    from causal_agents import historical_analog_agent
    historical_result = _safe_run_agent(historical_analog_agent, context, "historical_analog", step=6, total=TOTAL)
    context["historical_result"] = historical_result

    # ── Agent 6: Temporal Delay ──────────────────────────────────
    from causal_agents import temporal_delay_agent
    temporal_result = _safe_run_agent(temporal_delay_agent, context, "temporal_delay", step=7, total=TOTAL)
    context["temporal_result"] = temporal_result

    # ── Agent 7: Noise Filter ────────────────────────────────────
    from causal_agents import noise_filter_agent
    noise_result = _safe_run_agent(noise_filter_agent, context, "noise_filter", step=8, total=TOTAL)
    context["noise_result"] = noise_result

    # ── Agent 8: Contradiction (LLM call) ────────────────────────
    from causal_agents import contradiction_agent
    contradiction_result = _safe_run_agent(contradiction_agent, context, "contradiction", step=9, total=TOTAL)
    context["contradiction_result"] = contradiction_result

    # ── Agent 9: Self-Evaluation ─────────────────────────────────
    from causal_agents import self_eval_agent
    self_eval_result = _safe_run_agent(self_eval_agent, context, "self_eval", step=10, total=TOTAL)

    # ── Agent 10: Meta-Supervisor (Self-Evolving) ────────────────
    from causal_agents import meta_supervisor
    context["final_confidence"] = macro_result.get("confidence", 60)
    meta_result = _safe_run_agent(meta_supervisor, context, "meta_supervisor", step=11, total=TOTAL)

    # ── Synthesize Unified Thesis ────────────────────────────────
    _emit_event("agent_start", "synthesis", 12, TOTAL,
                message="Combining all 10 agents into a unified thesis...")
    t_synth = time.time()
    today_output = _synthesize_narrative(
        macro_result, supply_result, behavioral_result, flow_result,
        historical_result, temporal_result, noise_result,
        contradiction_result, self_eval_result,
        sector_df,
    )
    dur_synth = int((time.time() - t_synth) * 1000)
    _emit_event("agent_done", "synthesis", 12, TOTAL,
                message=f"Mood: {today_output.get('mood', 'mixed')} | Confidence: {today_output.get('confidence', 50)}%",
                confidence=today_output.get("confidence"),
                duration_ms=dur_synth)

    # Apply meta-supervisor confidence penalty (anti-delusion)
    meta_penalty = meta_result.get("confidence_penalty", 0)
    if meta_penalty > 0:
        today_output["confidence"] = max(10, today_output["confidence"] - meta_penalty)
        _log(f"    ⚠ Meta-supervisor applied -{meta_penalty}% confidence penalty")

    # Add dynamic agent insights to hidden signals
    for dyn in meta_result.get("dynamic_agent_results", []):
        out = dyn.get("output", {})
        if out.get("plain_summary") and out.get("confidence", 0) > 40:
            today_output.setdefault("hidden_signals", []).append({
                "icon": "🤖",
                "text": f"[{dyn['name']}] {out['plain_summary']}",
                "entity": dyn.get("domain", ""),
                "expected_delay": "",
            })

    # Add skepticism warnings to risks
    for warn in meta_result.get("skepticism", {}).get("warnings", []):
        if warn.get("severity") == "high":
            today_output.setdefault("what_could_go_wrong", []).append(warn["message"])

    today_output["last_updated"] = now.isoformat()
    _log(f"    Mood: {today_output['mood']} | Confidence: {today_output['confidence']}%")
    if meta_result.get("agents_spawned"):
        for s in meta_result["agents_spawned"]:
            _log(f"    🆕 Spawned: {s['name']} ({s['domain']})")

    # ── Store in memory ──────────────────────────────────────────
    try:
        memory.store_reasoning_chain(
            situation=f"Market analysis {now.strftime('%Y-%m-%d')}",
            agent_thoughts=[
                f"Macro: {macro_result.get('summary', '')}",
                f"Supply: {supply_result.get('summary', '')}",
                f"Behavioral: {behavioral_result.get('summary', '')}",
                f"Flow: {flow_result.get('summary', '')}",
                f"Historical: {historical_result.get('summary', '')}",
                f"Temporal: {temporal_result.get('summary', '')}",
                f"Contradiction: {contradiction_result.get('summary', '')}",
            ],
            tool_calls=[{"tool": "9-agent-pipeline"}],
            final_conclusion=today_output.get("narrative", ""),
        )
    except Exception:
        pass

    # ── Save prediction ──────────────────────────────────────────
    regime_map = {
        "fearful": "Panic",
        "cautious": "Risk Off",
        "mixed": "Risk Off",
        "calm": "Risk On",
        "cautiously positive": "Risk On",
        "deceptively positive": "Complacent",
        "oversold — contrarian opportunity": "Risk Off",
    }
    regime = regime_map.get(today_output["mood"], "Risk Off")
    avg_nss = float(sector_df["composite_sentiment_index"].mean()) if not sector_df.empty else 0.0
    avg_risk = float(sector_df["avg_weighted_risk"].mean()) if not sector_df.empty else 0.0

    sector_signals = {}
    for sb in today_output.get("sectors", []):
        sector_signals[sb["name"]] = sb.get("signal", "NEUTRAL")

    today_str = now.strftime("%Y-%m-%d")
    db.save_prediction(today_str, regime, avg_nss, avg_risk, sector_signals)

    # ── Build full analysis output ───────────────────────────────
    full_analysis = {
        "agents": {
            "macro": macro_result,
            "supply_chain": supply_result,
            "behavioral": behavioral_result,
            "flow": flow_result,
            "historical_analog": historical_result,
            "temporal_delay": temporal_result,
            "noise_filter": noise_result,
            "contradiction": contradiction_result,
            "self_eval": self_eval_result,
            "meta_supervisor": meta_result,
        },
        "regime": regime,
        "regime_confidence": today_output["confidence"],
        "msi": msi,
        "macro": macro_result.get("macro_data", {}),
        "headlines_analyzed": int(len(scored_df)),
        "sectors_count": int(len(sector_df)),
        "dynamic_agents": meta_result.get("registry", {}),
        "system_health": meta_result.get("skepticism", {}).get("health", "unknown"),
    }

    # ── Save combined result as JSON ─────────────────────────────
    result = {
        "today": today_output,
        "full_analysis": full_analysis,
        "timestamp": now.isoformat(),
        # Backward-compatible fields
        "regime": regime,
        "regime_confidence": today_output["confidence"],
        "nifty_direction": "bullish" if today_output["mood_color"] == "green" else "bearish" if today_output["mood_color"] == "red" else "neutral",
        "top_insight": today_output.get("narrative", ""),
        "risk_flags": macro_result.get("risk_flags", []),
        "sector_signals": sector_signals,
        "msi": msi,
        "macro": macro_result.get("macro_data", {}),
        "headlines_analyzed": int(len(scored_df)),
    }

    result_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "marketpulse-backend", "agent_result.json")
    # Fallback to same directory
    if not os.path.isdir(os.path.dirname(result_path)):
        result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent_result.json")
    try:
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent_result.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
    except Exception as e:
        _log(f"  [!] Failed to save agent_result.json: {e}")

    _emit_event("pipeline_done", "pipeline", TOTAL, TOTAL,
                message=f"Pipeline complete. {len(scored_df)} headlines | {len(sector_df)} sectors | regime={regime}")
    return result
