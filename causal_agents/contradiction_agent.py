"""
Contradiction Agent

The most important agent. Its job: DISAGREE with the emerging consensus.
Prevents confirmation bias by explicitly challenging the thesis.

"Why might this news NOT matter?"
"What if market already priced this?"
"What hidden negative exists?"
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict


def _build_contradiction_prompt(
    macro_summary: str,
    supply_chain_summary: str,
    behavioral_summary: str,
    flow_summary: str,
    sector_data: str,
) -> str:
    return f"""You are a senior contrarian analyst. Your ONLY job is to find what's WRONG with the current market thesis.

Do NOT agree. Do NOT be balanced. Find the holes.

CURRENT THESIS:
{macro_summary}

SUPPLY CHAIN VIEW:
{supply_chain_summary}

BEHAVIORAL SIGNALS:
{behavioral_summary}

FLOW SIGNALS:
{flow_summary}

SECTOR DATA:
{sector_data}

Answer in strict JSON:
{{
  "counter_thesis": "one paragraph — the strongest argument AGAINST the current view",
  "what_market_missed": "what negative signal is everyone ignoring?",
  "already_priced": "what part of the thesis is already reflected in prices?",
  "invalidation_conditions": ["condition 1 that breaks the thesis", "condition 2"],
  "confidence_penalty": 0-30 (how much should we reduce overall confidence because of these risks),
  "plain_warning": "one sentence a normal person would understand about what could go wrong"
}}"""


def run(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Challenge the emerging consensus from other agents.
    Uses LLM to find genuine counter-arguments.
    """
    macro_result = context.get("macro_result", {})
    supply_result = context.get("supply_chain_result", {})
    behavioral_result = context.get("behavioral_result", {})
    flow_result = context.get("flow_result", {})

    macro_summary = macro_result.get("summary", "No macro data.")
    sc_summary = supply_result.get("summary", "No supply chain analysis.")
    behav_summary = behavioral_result.get("summary", "No behavioral data.")
    flow_summary = flow_result.get("summary", "No flow data.")

    # Sector summary from scored data
    sectors = context.get("sector_data", [])
    sector_lines = []
    for s in sectors[:10]:
        if isinstance(s, dict):
            name = s.get("sector", "")
            signal = s.get("investment_signal", "NEUTRAL")
            risk = s.get("avg_weighted_risk", 0)
            csi = s.get("composite_sentiment_index", 0)
            sector_lines.append(f"{name}: signal={signal}, risk={risk}, sentiment_index={csi}")
    sector_text = "\n".join(sector_lines) if sector_lines else "No sector data."

    prompt = _build_contradiction_prompt(
        macro_summary, sc_summary, behav_summary, flow_summary, sector_text
    )

    # Try Anthropic first, then OpenAI
    result = {}
    try:
        if os.getenv("ANTHROPIC_API_KEY"):
            from anthropic import Anthropic
            import re

            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            resp = client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
                max_tokens=800,
                temperature=0.4,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text if resp.content else "{}"
            m = re.search(r"\{[\s\S]*\}", text)
            result = json.loads(m.group()) if m else {}
        else:
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model = os.getenv("OPENAI_MODEL_AGENT", "gpt-4o-mini")
            r = client.chat.completions.create(
                model=model,
                temperature=0.4,
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            result = json.loads(r.choices[0].message.content or "{}")
    except Exception as e:
        result = {
            "counter_thesis": f"Contradiction agent failed: {str(e)[:100]}",
            "confidence_penalty": 10,
            "invalidation_conditions": ["Unable to generate counter-thesis"],
            "plain_warning": "Could not verify the thesis — treat with extra caution.",
        }

    return {
        "agent": "contradiction",
        "counter_thesis": result.get("counter_thesis", ""),
        "what_market_missed": result.get("what_market_missed", ""),
        "already_priced": result.get("already_priced", ""),
        "invalidation_conditions": result.get("invalidation_conditions", []),
        "confidence_penalty": int(result.get("confidence_penalty", 10)),
        "plain_warning": result.get("plain_warning", ""),
        "summary": result.get("plain_warning", "No counter-thesis generated."),
    }
