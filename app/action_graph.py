"""Minimal action-graph representation (item 22).

Nodes:  actions in the plan.
Edges:  sequential (action N precedes N+1) and conditional (if this action
        doesn't resolve, try the next action at the same risk tier).

The graph lives in the response meta block, not in the Pydantic schema,
so the competition contract is preserved.
"""
from typing import Any, Dict, List


def build_graph(actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    nodes = []
    edges = []

    for i, a in enumerate(actions):
        node_id = f"a{i}"
        nodes.append({
            "id": node_id,
            "actionName": a.get("actionName", ""),
            "category": a.get("category", "manual"),
        })

    # Sequential edges.
    for i in range(len(nodes) - 1):
        edges.append({
            "from": nodes[i]["id"],
            "to": nodes[i + 1]["id"],
            "kind": "sequential",
        })

    # Conditional edges between same-tier actions: if action i fails,
    # try i+1. Only auto-tier, since manual/critical are terminal.
    prev_auto = None
    for i, n in enumerate(nodes):
        if n["category"] == "auto":
            if prev_auto is not None:
                edges.append({
                    "from": prev_auto,
                    "to": n["id"],
                    "kind": "fallback",
                })
            prev_auto = n["id"]

    return {
        "nodes": nodes,
        "edges": edges,
        "tiers": _tier_index(nodes),
    }


def _tier_index(nodes: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    tiers: Dict[str, List[str]] = {"auto": [], "manual": [], "critical": []}
    for n in nodes:
        tiers.setdefault(n["category"], []).append(n["id"])
    return tiers