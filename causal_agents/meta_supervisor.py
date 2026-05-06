"""
Meta-Supervisor Agent — The System That Watches The System

This is NOT a market analysis agent. Its job:
1. Detect blind spots — topics the system doesn't understand well enough
2. Spawn specialized micro-agents when novelty is detected
3. Track agent lifecycle — birth, evolution, retirement
4. Prevent hallucinated complexity via skepticism checks
5. Allocate "cognitive attention" to what matters most

The key insight: reality changes faster than predefined logic.
When the system encounters something it can't handle, it creates
new cognition rather than forcing old frameworks onto new problems.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import database as db

IST = timezone(timedelta(hours=5, minutes=30))

# ── How many dynamic agents can exist simultaneously ──
MAX_ACTIVE_AGENTS = 8
# ── Minimum novelty score to trigger agent creation ──
NOVELTY_THRESHOLD = 0.65
# ── Minimum times a topic must appear before spawning ──
MIN_APPEARANCE_COUNT = 3


def _ensure_tables():
    """Create tables for the dynamic agent registry."""
    with db.get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dynamic_agents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                domain TEXT NOT NULL,
                trigger_reason TEXT,
                system_prompt TEXT NOT NULL,
                focus_entities TEXT,
                status TEXT DEFAULT 'active',
                times_invoked INTEGER DEFAULT 0,
                times_useful INTEGER DEFAULT 0,
                confidence_avg REAL DEFAULT 0.5,
                created_at TEXT,
                last_invoked_at TEXT,
                retired_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS novelty_tracker (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                first_seen TEXT,
                times_seen INTEGER DEFAULT 1,
                confidence_score REAL DEFAULT 0.0,
                spawned_agent_id TEXT,
                context_json TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_performance_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                date TEXT,
                output_json TEXT,
                was_useful INTEGER,
                confidence REAL,
                created_at TEXT
            )
        """)


_ensure_tables()


# ═══════════════════════════════════════════════════════════════════
# 1. NOVELTY DETECTION — "What don't we understand?"
# ═══════════════════════════════════════════════════════════════════

def _load_known_entities() -> set:
    """Load all entity names/aliases from the supply chain graph."""
    try:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        with open(os.path.join(data_dir, "supply_chain_graph.json"), "r") as f:
            graph = json.load(f)
        known = set()
        for entity in graph.get("entities", []):
            known.add(entity["name"].lower())
            for alias in entity.get("aliases", []):
                known.add(alias.lower())
        return known
    except Exception:
        return set()


def detect_novelty(headlines: List[Dict], agent_outputs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Scan for topics/entities the system doesn't understand well.
    
    Returns a list of novelty signals:
    {topic, novelty_score, reason, appearances, suggested_domain}
    """
    known_entities = _load_known_entities()
    
    # Extract all unique terms from headlines that are NOT in our knowledge
    topic_counts: Dict[str, int] = {}
    topic_contexts: Dict[str, List[str]] = {}
    
    for h in headlines:
        title = h.get("title", "")
        # Extract capitalized phrases (likely proper nouns / entities)
        phrases = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', title)
        # Also extract quoted terms and technical terms
        quoted = re.findall(r'"([^"]+)"', title)
        phrases.extend(quoted)
        
        for phrase in phrases:
            pl = phrase.lower().strip()
            if len(pl) < 4 or pl in known_entities:
                continue
            # Skip common words
            if pl in {"india", "indian", "market", "stock", "share", "price", 
                      "report", "today", "says", "news", "update", "govt",
                      "minister", "company", "billion", "crore", "rupee",
                      "growth", "profit", "revenue", "quarter", "year",
                      "budget", "policy", "trade", "global", "world"}:
                continue
            topic_counts[pl] = topic_counts.get(pl, 0) + 1
            topic_contexts.setdefault(pl, []).append(title[:100])
    
    # Check which agent outputs show low confidence
    low_confidence_domains = []
    for agent_name, output in agent_outputs.items():
        if isinstance(output, dict):
            conf = output.get("confidence", output.get("confidence_avg", 70))
            if isinstance(conf, (int, float)) and conf < 40:
                low_confidence_domains.append({
                    "agent": agent_name,
                    "confidence": conf,
                    "summary": output.get("summary", "")[:200],
                })
    
    # Score novelty
    novelties = []
    for topic, count in topic_counts.items():
        if count < 2:
            continue
        
        # Higher count + not in knowledge = higher novelty
        novelty_score = min(1.0, (count / 10) * 0.4 + 0.6)
        
        # Check if we've seen this before in the tracker
        with db.get_db() as conn:
            existing = conn.execute(
                "SELECT times_seen, spawned_agent_id FROM novelty_tracker WHERE topic = ?",
                (topic,)
            ).fetchone()
        
        if existing:
            total_seen = existing["times_seen"] + count
            already_spawned = existing["spawned_agent_id"]
            with db.get_db() as conn:
                conn.execute(
                    "UPDATE novelty_tracker SET times_seen = ?, context_json = ? WHERE topic = ?",
                    (total_seen, json.dumps(topic_contexts.get(topic, []), default=str), topic),
                )
            if already_spawned:
                continue  # Already handled
            novelty_score = min(1.0, novelty_score + (total_seen / 20))
        else:
            with db.get_db() as conn:
                conn.execute(
                    "INSERT INTO novelty_tracker (topic, first_seen, times_seen, confidence_score, context_json) VALUES (?, ?, ?, ?, ?)",
                    (topic, datetime.now(IST).isoformat(), count, novelty_score,
                     json.dumps(topic_contexts.get(topic, []), default=str)),
                )
        
        if novelty_score >= NOVELTY_THRESHOLD and count >= MIN_APPEARANCE_COUNT:
            novelties.append({
                "topic": topic,
                "novelty_score": round(novelty_score, 2),
                "appearances": count,
                "reason": f"Topic '{topic}' appeared {count} times but is not in our knowledge graph",
                "contexts": topic_contexts.get(topic, [])[:3],
                "suggested_domain": _suggest_domain(topic, topic_contexts.get(topic, [])),
            })
    
    # Sort by novelty score
    novelties.sort(key=lambda x: x["novelty_score"], reverse=True)
    return novelties[:5]


def _suggest_domain(topic: str, contexts: List[str]) -> str:
    """Suggest what domain a novel topic belongs to."""
    combined = (topic + " " + " ".join(contexts)).lower()
    domain_keywords = {
        "technology": ["ai", "quantum", "blockchain", "crypto", "chip", "semiconductor", "compute", "gpu", "cloud"],
        "energy": ["nuclear", "solar", "wind", "hydrogen", "battery", "grid", "power", "reactor", "uranium"],
        "geopolitics": ["sanctions", "tariff", "war", "conflict", "treaty", "embargo", "diplomatic"],
        "regulation": ["sebi", "rbi", "regulation", "compliance", "ban", "license", "framework"],
        "supply_chain": ["shortage", "supply", "logistics", "import", "export", "manufacturing"],
        "finance": ["bond", "yield", "credit", "derivative", "futures", "option", "forex"],
        "healthcare": ["drug", "vaccine", "clinical", "trial", "fda", "pharma", "biotech"],
    }
    scores = {}
    for domain, keywords in domain_keywords.items():
        scores[domain] = sum(1 for kw in keywords if kw in combined)
    if not scores or max(scores.values()) == 0:
        return "emerging"
    return max(scores, key=lambda k: scores[k])


# ═══════════════════════════════════════════════════════════════════
# 2. DYNAMIC AGENT CREATION — "Build new cognition"
# ═══════════════════════════════════════════════════════════════════

def spawn_agent(
    topic: str,
    domain: str,
    trigger_reason: str,
    contexts: List[str],
) -> Optional[Dict[str, Any]]:
    """
    Create a new specialized micro-agent for an emerging topic.
    
    This doesn't create code — it creates a specialized LLM prompt
    configuration that gets invoked during the pipeline.
    """
    # Check if we have too many active agents
    with db.get_db() as conn:
        active_count = conn.execute(
            "SELECT COUNT(*) as c FROM dynamic_agents WHERE status = 'active'"
        ).fetchone()["c"]
    
    if active_count >= MAX_ACTIVE_AGENTS:
        # Retire the least useful agent
        _retire_weakest_agent()
    
    agent_id = f"dyn_{domain}_{topic.replace(' ', '_')[:20]}_{datetime.now(IST).strftime('%m%d')}"
    
    # Build specialized system prompt
    system_prompt = f"""You are a specialized market intelligence agent focused on: {topic}

Your domain: {domain}
Your job: Analyze news and events related to "{topic}" and identify:
1. Which Indian market sectors and companies are affected
2. Whether the impact is immediate or delayed
3. What causal chain connects this topic to stock prices
4. What most people are missing about this topic
5. Your confidence level (0-100%) in your analysis

Recent context that triggered your creation:
{chr(10).join(f'- {c}' for c in contexts[:5])}

You must output strict JSON:
{{
    "topic": "{topic}",
    "relevance_today": "high/medium/low",
    "affected_sectors": ["sector1", "sector2"],
    "affected_companies": ["company1", "company2"],
    "causal_chain": "X leads to Y which causes Z",
    "immediate_impact": "what happens now",
    "delayed_impact": "what happens in weeks/months",
    "what_market_misses": "the non-obvious insight",
    "confidence": 0-100,
    "plain_summary": "one paragraph for a non-expert"
}}"""
    
    now = datetime.now(IST).isoformat()
    
    with db.get_db() as conn:
        try:
            conn.execute(
                """INSERT INTO dynamic_agents 
                   (agent_id, name, domain, trigger_reason, system_prompt, focus_entities, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'active', ?)""",
                (agent_id, f"{topic.title()} Agent", domain, trigger_reason,
                 system_prompt, json.dumps(contexts[:5], default=str), now),
            )
        except Exception:
            return None  # Already exists
    
    # Mark the novelty tracker
    with db.get_db() as conn:
        conn.execute(
            "UPDATE novelty_tracker SET spawned_agent_id = ? WHERE topic = ?",
            (agent_id, topic),
        )
    
    return {
        "agent_id": agent_id,
        "name": f"{topic.title()} Agent",
        "domain": domain,
        "status": "active",
        "created_at": now,
    }


def _retire_weakest_agent():
    """Retire the least useful active dynamic agent."""
    with db.get_db() as conn:
        weakest = conn.execute(
            """SELECT agent_id FROM dynamic_agents 
               WHERE status = 'active' 
               ORDER BY times_useful ASC, times_invoked ASC 
               LIMIT 1"""
        ).fetchone()
        if weakest:
            conn.execute(
                "UPDATE dynamic_agents SET status = 'dormant', retired_at = ? WHERE agent_id = ?",
                (datetime.now(IST).isoformat(), weakest["agent_id"]),
            )


# ═══════════════════════════════════════════════════════════════════
# 3. DYNAMIC AGENT EXECUTION — "Run the spawned cognition"
# ═══════════════════════════════════════════════════════════════════

def run_dynamic_agents(headlines: List[Dict]) -> List[Dict[str, Any]]:
    """
    Run all active dynamic agents against today's headlines.
    Each dynamic agent is an LLM call with a specialized prompt.
    """
    with db.get_db() as conn:
        agents = conn.execute(
            "SELECT * FROM dynamic_agents WHERE status = 'active'"
        ).fetchall()
    
    if not agents:
        return []
    
    # Filter headlines relevant to each agent's domain
    results = []
    for agent in agents:
        agent_dict = dict(agent)
        focus = json.loads(agent_dict.get("focus_entities", "[]")) if agent_dict.get("focus_entities") else []
        
        # Find relevant headlines
        relevant = []
        for h in headlines:
            title_lower = h.get("title", "").lower()
            if any(f.lower() in title_lower for f in focus):
                relevant.append(h.get("title", ""))
            elif agent_dict["domain"].lower() in title_lower:
                relevant.append(h.get("title", ""))
        
        if not relevant:
            continue
        
        # Run the agent (LLM call)
        output = _execute_dynamic_agent(
            agent_dict["system_prompt"],
            relevant[:10],
            agent_dict["agent_id"],
        )
        
        if output:
            results.append({
                "agent_id": agent_dict["agent_id"],
                "name": agent_dict["name"],
                "domain": agent_dict["domain"],
                "output": output,
                "headlines_analyzed": len(relevant),
            })
            
            # Update invocation stats
            was_useful = output.get("confidence", 0) > 30 and output.get("relevance_today") != "low"
            with db.get_db() as conn:
                conn.execute(
                    """UPDATE dynamic_agents 
                       SET times_invoked = times_invoked + 1,
                           times_useful = times_useful + ?,
                           last_invoked_at = ?
                       WHERE agent_id = ?""",
                    (1 if was_useful else 0, datetime.now(IST).isoformat(), agent_dict["agent_id"]),
                )
            
            # Log performance
            with db.get_db() as conn:
                conn.execute(
                    """INSERT INTO agent_performance_log (agent_id, date, output_json, was_useful, confidence, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (agent_dict["agent_id"], datetime.now(IST).strftime("%Y-%m-%d"),
                     json.dumps(output, default=str), int(was_useful),
                     output.get("confidence", 0), datetime.now(IST).isoformat()),
                )
    
    return results


def _execute_dynamic_agent(system_prompt: str, headlines: List[str], agent_id: str) -> Optional[Dict]:
    """Execute a dynamic agent via LLM."""
    user_msg = f"""Today's relevant headlines:
{chr(10).join(f'- {h}' for h in headlines)}

Analyze these and provide your specialized assessment."""

    try:
        if os.getenv("ANTHROPIC_API_KEY"):
            from anthropic import Anthropic
            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            resp = client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
                max_tokens=600,
                temperature=0.3,
                system=system_prompt,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = resp.content[0].text if resp.content else "{}"
        else:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            r = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL_AGENT", "gpt-4o-mini"),
                temperature=0.3,
                max_tokens=600,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
            )
            text = r.choices[0].message.content or "{}"
        
        m = re.search(r"\{[\s\S]*\}", text)
        return json.loads(m.group()) if m else None
    except Exception as e:
        return {"error": str(e)[:100], "confidence": 0}


# ═══════════════════════════════════════════════════════════════════
# 4. ANTI-DELUSION SYSTEM — "Are we fooling ourselves?"
# ═══════════════════════════════════════════════════════════════════

def run_skepticism_check(
    agent_outputs: Dict[str, Any],
    dynamic_results: List[Dict],
    final_confidence: int,
) -> Dict[str, Any]:
    """
    The immune system. Checks for:
    - Recursive confidence spirals (agents reinforcing each other)
    - Fake causality (correlations masquerading as causation)
    - Narrative hallucination (convincing but unfounded stories)
    - Overconfidence relative to data quality
    """
    warnings = []
    confidence_penalty = 0
    
    # 1. Check for confidence spiral
    confidences = []
    for name, output in agent_outputs.items():
        if isinstance(output, dict):
            c = output.get("confidence", output.get("confidence_avg"))
            if isinstance(c, (int, float)):
                confidences.append(c)
    
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        all_agree = all(c > 60 for c in confidences)
        
        if all_agree and avg_conf > 75:
            warnings.append({
                "type": "confidence_spiral",
                "severity": "high",
                "message": f"All agents agree with avg {avg_conf:.0f}% confidence. When everyone agrees, something is usually wrong. Reducing confidence.",
            })
            confidence_penalty += 15
    
    # 2. Check signal quality vs confidence mismatch
    noise_quality = agent_outputs.get("noise_filter", {}).get("signal_quality", {})
    quality_score = noise_quality.get("quality", 50) if isinstance(noise_quality, dict) else 50
    
    if quality_score < 30 and final_confidence > 60:
        warnings.append({
            "type": "quality_mismatch",
            "severity": "high",
            "message": f"Signal quality is only {quality_score}/100 but confidence is {final_confidence}%. Can't be confident with bad data.",
        })
        confidence_penalty += 20
    
    # 3. Check for narrative without evidence
    supply_chains = agent_outputs.get("supply_chain", {}).get("chains", [])
    if not supply_chains and final_confidence > 70:
        warnings.append({
            "type": "weak_evidence",
            "severity": "medium",
            "message": "High confidence but no concrete supply chain evidence found. Thesis may be narrative-driven, not data-driven.",
        })
        confidence_penalty += 10
    
    # 4. Check for disagreement being suppressed
    contradiction = agent_outputs.get("contradiction", {})
    contra_penalty = contradiction.get("confidence_penalty", 0) if isinstance(contradiction, dict) else 0
    if contra_penalty > 20 and final_confidence > 65:
        warnings.append({
            "type": "suppressed_dissent",
            "severity": "high",
            "message": "Contradiction agent raised serious concerns but final confidence remains high. The counter-thesis deserves more weight.",
        })
        confidence_penalty += 10
    
    # 5. Check dynamic agents for conflicting signals
    if dynamic_results:
        dyn_sectors = set()
        dyn_directions = set()
        for dr in dynamic_results:
            out = dr.get("output", {})
            for s in out.get("affected_sectors", []):
                dyn_sectors.add(s)
            impact = out.get("immediate_impact", "")
            if "negative" in impact.lower() or "risk" in impact.lower():
                dyn_directions.add("negative")
            elif "positive" in impact.lower() or "benefit" in impact.lower():
                dyn_directions.add("positive")
        
        if len(dyn_directions) > 1:
            warnings.append({
                "type": "dynamic_conflict",
                "severity": "medium",
                "message": "Spawned specialist agents disagree on direction. Emerging domains show mixed signals.",
            })
            confidence_penalty += 5
    
    # Build overall assessment
    if confidence_penalty > 30:
        health = "unstable"
        health_message = "System is showing signs of overconfidence or weak reasoning. Trust outputs less today."
    elif confidence_penalty > 15:
        health = "cautious"
        health_message = "Some reasoning inconsistencies detected. Outputs are usable but treat with extra skepticism."
    elif warnings:
        health = "minor_concerns"
        health_message = "Small issues detected but overall reasoning appears sound."
    else:
        health = "healthy"
        health_message = "No major reasoning issues detected. System appears well-calibrated."
    
    return {
        "health": health,
        "health_message": health_message,
        "warnings": warnings,
        "confidence_penalty": confidence_penalty,
        "checks_run": 5,
        "issues_found": len(warnings),
    }


# ═══════════════════════════════════════════════════════════════════
# 5. AGENT REGISTRY — "What agents exist?"
# ═══════════════════════════════════════════════════════════════════

def get_agent_registry() -> Dict[str, Any]:
    """Get the full registry of dynamic agents."""
    with db.get_db() as conn:
        active = conn.execute(
            "SELECT agent_id, name, domain, status, times_invoked, times_useful, created_at, last_invoked_at FROM dynamic_agents WHERE status = 'active'"
        ).fetchall()
        dormant = conn.execute(
            "SELECT agent_id, name, domain, status, times_invoked, times_useful, created_at, retired_at FROM dynamic_agents WHERE status = 'dormant'"
        ).fetchall()
    
    return {
        "active_agents": [dict(a) for a in active],
        "dormant_agents": [dict(d) for d in dormant],
        "total_active": len(active),
        "total_dormant": len(dormant),
        "max_allowed": MAX_ACTIVE_AGENTS,
    }


def reactivate_agent(agent_id: str) -> bool:
    """Reactivate a dormant agent (old regimes can return)."""
    with db.get_db() as conn:
        conn.execute(
            "UPDATE dynamic_agents SET status = 'active', retired_at = NULL WHERE agent_id = ?",
            (agent_id,),
        )
    return True


# ═══════════════════════════════════════════════════════════════════
# 6. MAIN ENTRY POINT — Called by the orchestrator
# ═══════════════════════════════════════════════════════════════════

def run(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Meta-supervisor entry point. Called after all 9 fixed agents.
    
    1. Detect novelty in today's headlines
    2. Spawn new agents if needed
    3. Run existing dynamic agents
    4. Run skepticism checks
    5. Return meta-intelligence
    """
    headlines = context.get("headlines", [])
    agent_outputs = {}
    for key in ["macro_result", "supply_chain_result", "behavioral_result",
                 "flow_result", "historical_result", "temporal_result",
                 "noise_result", "contradiction_result"]:
        if key in context:
            agent_outputs[key.replace("_result", "")] = context[key]
    
    # 1. Detect novelty
    novelties = detect_novelty(headlines, agent_outputs)
    
    # 2. Spawn agents for novel topics
    spawned = []
    for nov in novelties:
        if nov["novelty_score"] >= NOVELTY_THRESHOLD:
            result = spawn_agent(
                topic=nov["topic"],
                domain=nov["suggested_domain"],
                trigger_reason=nov["reason"],
                contexts=nov.get("contexts", []),
            )
            if result:
                spawned.append(result)
    
    # 3. Run existing dynamic agents
    dynamic_results = run_dynamic_agents(headlines)
    
    # 4. Run skepticism checks
    final_confidence = context.get("final_confidence", 60)
    skepticism = run_skepticism_check(agent_outputs, dynamic_results, final_confidence)
    
    # 5. Get registry status
    registry = get_agent_registry()
    
    # Build summary
    parts = []
    if spawned:
        names = [s["name"] for s in spawned]
        parts.append(f"Created {len(spawned)} new specialist agent(s): {', '.join(names)}.")
    if dynamic_results:
        parts.append(f"{len(dynamic_results)} specialist agent(s) contributed insights today.")
    if novelties:
        topics = [n["topic"] for n in novelties[:3]]
        parts.append(f"Emerging topics detected: {', '.join(topics)}.")
    if skepticism["warnings"]:
        parts.append(f"⚠ {len(skepticism['warnings'])} reasoning concern(s) flagged.")
    if not parts:
        parts.append("No emerging topics or reasoning concerns detected. System is stable.")
    
    return {
        "agent": "meta_supervisor",
        "novelties_detected": novelties,
        "agents_spawned": spawned,
        "dynamic_agent_results": dynamic_results,
        "skepticism": skepticism,
        "registry": registry,
        "confidence_penalty": skepticism["confidence_penalty"],
        "summary": " ".join(parts),
    }
