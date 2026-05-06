"""
Supply Chain Graph Agent

Knows who supplies whom. When a headline triggers an entity node,
propagates the impact through the causal graph and identifies
second/third-order effects that most people miss.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _load_graph() -> Dict[str, Any]:
    path = os.path.join(_DATA_DIR, "supply_chain_graph.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _entity_match(text: str, entity: Dict[str, Any]) -> bool:
    """Check if an entity is mentioned in text (case-insensitive)."""
    text_lower = text.lower()
    if entity["name"].lower() in text_lower:
        return True
    for alias in entity.get("aliases", []):
        if alias.lower() in text_lower:
            return True
    return False


def _find_triggered_entities(headlines: List[str], entities: List[Dict]) -> List[str]:
    """Find which entities are mentioned across all headlines."""
    triggered = set()
    combined_text = " ".join(headlines).lower()
    for entity in entities:
        if _entity_match(combined_text, entity):
            triggered.add(entity["id"])
    return list(triggered)


def _propagate(
    triggered: List[str],
    relationships: List[Dict],
    depth: int = 3,
) -> List[Dict[str, Any]]:
    """
    Propagate effects through the graph.
    Returns a list of causal chains.
    """
    chains = []
    visited = set()

    def _walk(source_id: str, current_depth: int, path: List[str]):
        if current_depth > depth or source_id in visited:
            return
        visited.add(source_id)
        for rel in relationships:
            if rel["source"] == source_id:
                target = rel["target"]
                chain_entry = {
                    "from": source_id,
                    "to": target,
                    "type": rel["type"],
                    "direction": rel["direction"],
                    "strength": rel["strength"],
                    "delay_hours": rel.get("delay_hours", 0),
                    "plain": rel.get("plain", ""),
                    "depth": current_depth,
                    "path": path + [target],
                }
                chains.append(chain_entry)
                _walk(target, current_depth + 1, path + [target])
        visited.discard(source_id)

    for entity_id in triggered:
        _walk(entity_id, 1, [entity_id])

    return chains


def _find_delayed_effects(headlines: List[str], graph: Dict) -> List[Dict[str, Any]]:
    """Check if any delayed-effect templates match current headlines."""
    templates = graph.get("delayed_effect_templates", [])
    combined = " ".join(headlines).lower()
    matches = []

    for tpl in templates:
        for kw in tpl.get("keywords", []):
            if kw.lower() in combined:
                matches.append({
                    "trigger": tpl["trigger"],
                    "matched_keyword": kw,
                    "chain": tpl["chain"],
                })
                break

    return matches


def run(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze headlines for supply-chain and causal propagation effects.
    """
    headlines = [h.get("title", "") for h in context.get("headlines", [])]
    if not headlines:
        return {
            "agent": "supply_chain",
            "chains": [],
            "delayed_effects": [],
            "summary": "No headlines to analyze for supply chain effects.",
            "beneficiaries": [],
            "victims": [],
        }

    graph = _load_graph()
    entities = graph.get("entities", [])
    relationships = graph.get("relationships", [])

    # Find which entities are mentioned
    triggered = _find_triggered_entities(headlines, entities)

    # Propagate through graph
    chains = _propagate(triggered, relationships, depth=3)

    # Find delayed effects
    delayed = _find_delayed_effects(headlines, graph)

    # Identify beneficiaries (positive effects) and victims (negative effects)
    entity_names = {e["id"]: e["name"] for e in entities}
    beneficiaries = []
    victims = []
    seen_b = set()
    seen_v = set()

    for c in chains:
        target_name = entity_names.get(c["to"], c["to"])
        if "negative" in c["direction"] and c["to"] not in seen_v:
            victims.append({
                "entity": target_name,
                "reason": c["plain"],
                "strength": c["strength"],
                "delay": c["delay_hours"],
            })
            seen_v.add(c["to"])
        elif "positive" in c["direction"] and c["to"] not in seen_b:
            beneficiaries.append({
                "entity": target_name,
                "reason": c["plain"],
                "strength": c["strength"],
                "delay": c["delay_hours"],
            })
            seen_b.add(c["to"])

    # Sort by strength
    beneficiaries.sort(key=lambda x: x["strength"], reverse=True)
    victims.sort(key=lambda x: x["strength"], reverse=True)

    # Build plain-English summary
    parts = []
    if beneficiaries:
        top_b = [b["entity"] for b in beneficiaries[:3]]
        parts.append(f"Likely beneficiaries: {', '.join(top_b)}.")
    if victims:
        top_v = [v["entity"] for v in victims[:3]]
        parts.append(f"Under pressure: {', '.join(top_v)}.")
    if delayed:
        for d in delayed[:2]:
            last_step = d["chain"][-1] if d["chain"] else None
            if last_step:
                parts.append(
                    f"Hidden signal: {d['trigger'].replace('_', ' ')} → "
                    f"{last_step['effect']} ({last_step['delay']})."
                )

    # Format chains for the frontend as simple "X → Y → Z" narratives
    simple_chains = []
    for c in chains[:10]:
        source_name = entity_names.get(c["from"], c["from"])
        target_name = entity_names.get(c["to"], c["to"])
        simple_chains.append({
            "from_name": source_name,
            "to_name": target_name,
            "effect": c["plain"],
            "direction": c["direction"],
            "delay_hours": c["delay_hours"],
            "strength": c["strength"],
        })

    return {
        "agent": "supply_chain",
        "triggered_entities": triggered,
        "chains": simple_chains,
        "delayed_effects": delayed,
        "beneficiaries": beneficiaries[:8],
        "victims": victims[:8],
        "summary": " ".join(parts) if parts else "No significant supply chain effects detected today.",
    }
