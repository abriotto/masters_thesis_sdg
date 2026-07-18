import argparse
import copy
import hashlib
import json
import os
import random
import re
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from openai import OpenAI


# ============================================================
# Configuration
# ============================================================

NODES_ONLY_PROMPT_PATH = Path("src/prompts/argument_mining_nodes_only_v1.txt")
RELATIONS_ONLY_PROMPT_PATH = Path("src/prompts/argument_mining_relations_only_v1.txt")
REASONING_TYPE_ONLY_PROMPT_PATH = Path(
    "src/prompts/argument_mining_reasoning_type_only_v1.txt"
)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# V4 Pro across all three stages. The original mixed-tier allocation (flash
# for nodes/reasoning, pro for relations) was a cost-driven compromise from
# when the alternative was Gemini 3.5 Flash at real money. At DeepSeek's
# pricing, running Pro everywhere costs an estimated ~$2.50 worst-case (no
# caching) for a 382-row run -- cheap enough that there's no reason left to
# under-provision any stage. Override any stage independently via CLI if
# you ever want to test a cheaper tier again.
DEFAULT_NODES_MODEL = "deepseek-v4-pro"
DEFAULT_RELATIONS_MODEL = "deepseek-v4-pro"
DEFAULT_REASONING_MODEL = "deepseek-v4-pro"
DEFAULT_STAGE_SLEEP_SECONDS = 0.0
# NOT USED: per DeepSeek's own documentation, in thinking mode (which this
# runner always enables via extra_body={"thinking": {"type": "enabled"}}),
# temperature, top_p, presence_penalty, and frequency_penalty are ignored
# by the API entirely, regardless of what value is passed. DeepSeek's
# published use-case temperature table (Coding/Math=0.0, Data
# Cleaning=1.0, etc.) applies to non-thinking-mode calls, not this
# configuration. These are intentionally NOT sent to the API below. If you
# ever disable thinking mode, reinstate temperature/top_p here and consult
# that table -- argument-graph extraction is closest to the "Coding /
# Math" case (0.0), for the reasons discussed when this was last relevant.
DEFAULT_GENERATION_SEED = 42
DEFAULT_MAX_TOKENS = 8192
DEFAULT_MAX_TOKENS_ON_LENGTH_CAP = 32768
# DeepSeek's documented reasoning-depth control (client.chat.completions.create
# accepts reasoning_effort the same as OpenAI's o-series models). Confirmed
# valid value from DeepSeek's own docs example: "high". Other values
# (e.g. "low", "medium") are plausible by analogy to OpenAI's convention but
# not directly confirmed for DeepSeek -- verify against their docs if you
# change this.
DEFAULT_REASONING_EFFORT = "high"

# NOTE: DeepSeek's thinking mode is a default-ON binary toggle (per their
# own model table: "Supports both non-thinking and thinking (default)
# modes"), not Gemini's four graduated levels (minimal/low/medium/high).
# There is no direct equivalent of --thinking-level here -- simply not
# disabling thinking mode gets you the reasoning-capable behavior by
# default. If DeepSeek later exposes a graduated control, this is the
# place to wire it in.

# DeepSeek's caching is automatic and prefix-based (no explicit
# cache-creation call, unlike Gemini's client.caches.create()) -- it
# activates automatically whenever a request's leading tokens exactly
# match a recently processed prefix (your stable system prompts). There is
# nothing to configure here beyond keeping each prompt file byte-for-byte
# identical across calls, which the existing schema_version hash already
# lets you verify.

ALLOWED_ARGUMENT_ROLES = {"Premise", "Conclusion"}
ALLOWED_RELATIONS = {"supports", "attacks"}
ALLOWED_REASONING_TYPES = {
    "Game-Mechanical",
    "Claim-Consistency",
    "Behavioral-Credibility",
    "Social-Consensus",
    "Individual-Testimony",
    "Epistemic-Uncertainty",
    "Generic-Heuristic",
}

REPAIR_POLICY_VERSION = "lossless_normalization_v1"

# This layer follows the same general motivation as Pirozelli et al.'s
# well-formedness steps: make graph outputs consistent and inspectable.
# Unlike their optional semantic stages, these repairs never infer, delete,
# reconnect, or reinterpret argumentative content. The exact model output is
# always retained in `raw_graph`.
REPAIR_POLICY = {
    "version": REPAIR_POLICY_VERSION,
    "applied_repairs": [
        "trim outer whitespace in string fields",
        "remove exact duplicate node records",
        "renumber unique node IDs by node order and update references",
        "remove exact duplicate edge records",
        "renumber inference-group IDs by first edge appearance",
        "copy each group's sources, target, and relation from the fixed edges",
        "remove classifications for inference groups that have no edges",
        "remove exact duplicate inference-group classification records",
    ],
    "never_applied": [
        "add, remove, merge, split, or rewrite argumentative propositions",
        "attach disconnected nodes",
        "delete transitive edges",
        "break cycles",
        "delete outgoing edges from the Conclusion",
        "change support into attack or attack into support",
        "invent a missing inference group or reasoning type",
        "default a missing reasoning type to Generic-Heuristic",
    ],
}


# ============================================================
# JSON schemas
#
# IMPORTANT DIFFERENCE FROM THE GEMINI VERSION: Gemini's
# response_json_schema enforces these enums at the API level -- an invalid
# argument_role/relation/reasoning_type literally cannot be returned. This
# script uses DeepSeek's response_format={"type": "json_object"} instead,
# which is confirmed supported ("Json Output" in DeepSeek's model table)
# but only guarantees *valid JSON*, not schema/enum conformance. Whether
# DeepSeek's endpoint also supports the stricter OpenAI
# response_format={"type": "json_schema", ...} mode is UNVERIFIED as of
# writing -- worth testing directly against DeepSeek's docs before relying
# on it. Until confirmed, these schema dicts are kept here for reference
# and are embedded into the user message as an explicit instruction, but
# are NOT passed as an enforced response_format. The existing
# validate_graph() below is your real safety net now: expect a higher rate
# of invalid-enum flags than you saw with Gemini, since nothing upstream is
# preventing them.
# ============================================================

NODES_ONLY_SCHEMA = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "argument_role": {
                        "type": "string",
                        "enum": sorted(ALLOWED_ARGUMENT_ROLES),
                    },
                    "text": {"type": "string"},
                },
                "required": ["id", "argument_role", "text"],
            },
        },
    },
    "required": ["nodes"],
}

RELATIONS_ONLY_SCHEMA = {
    "type": "object",
    "properties": {
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "relation": {
                        "type": "string",
                        "enum": sorted(ALLOWED_RELATIONS),
                    },
                    "inference_group": {"type": "string"},
                },
                "required": [
                    "source",
                    "target",
                    "relation",
                    "inference_group",
                ],
            },
        },
    },
    "required": ["edges"],
}

REASONING_TYPE_ONLY_SCHEMA = {
    "type": "object",
    "properties": {
        "inference_groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "inference_group": {"type": "string"},
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "target": {"type": "string"},
                    "relation": {
                        "type": "string",
                        "enum": sorted(ALLOWED_RELATIONS),
                    },
                    "reasoning_type": {
                        "type": "string",
                        "enum": sorted(ALLOWED_REASONING_TYPES),
                    },
                },
                "required": [
                    "inference_group",
                    "sources",
                    "target",
                    "relation",
                    "reasoning_type",
                ],
            },
        },
    },
    "required": ["inference_groups"],
}


def schema_reminder_text(schema):
    """
    Since response_format={"type": "json_object"} does not enforce the
    schema the way Gemini's response_json_schema does, restate it
    explicitly in the user message as a belt-and-braces measure on top of
    the schema already described in each prompt's own OUTPUT FORMAT
    section. Cheap insurance, not a substitute for verifying whether
    DeepSeek's json_schema mode is actually available.
    """
    return (
        "\n\nYour entire response must be a single JSON object matching "
        "exactly this structure (no extra text, no markdown fences):\n"
        f"{json.dumps(schema, indent=2)}"
    )


# ============================================================
# General helpers (unchanged from the Gemini version -- none of this is
# provider-specific)
# ============================================================

def safe_str(value, default=""):
    if pd.isna(value):
        return default
    return str(value).strip()


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
        "correct",
    }


def normalize_chosen_vote(value):
    raw_value = safe_str(value, default="Unknown")
    normalized = raw_value.lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())

    no_werewolf_variants = {
        "no werewolf",
        "no werewolves",
        "none",
        "no one",
        "nobody",
        "no player",
        "no player is werewolf",
        "no player is the werewolf",
        "no current werewolf",
        "no werewolf present",
        "no werewolf in play",
    }
    return "No Werewolf" if normalized in no_werewolf_variants else raw_value


def prompt_content_hash(*prompts):
    combined = "\x00".join(prompts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:12]


def slugify(value):
    value = safe_str(value, default="unknown")
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value.strip("._-") or "unknown"


def normalize_prompt_version(value):
    """Accept values such as '4', 'v4', or 'prompt_v4'."""
    normalized = safe_str(value)
    if not normalized:
        return normalized
    if normalized.startswith("prompt_"):
        return normalized
    if normalized.startswith("v"):
        return f"prompt_{normalized}"
    return f"prompt_v{normalized}"


def infer_metadata_from_path(csv_path):
    parts = csv_path.parts
    inferred_model_name = "unknown_model"
    source_mode = "unknown_mode"
    prompt_version = "unknown_prompt"

    if "voting" in parts:
        voting_idx = parts.index("voting")
        if voting_idx + 1 < len(parts):
            prompt_version = parts[voting_idx + 1]
        if voting_idx - 1 >= 0:
            source_mode = parts[voting_idx - 1]
        if voting_idx - 2 >= 0:
            inferred_model_name = parts[voting_idx - 2]

    return inferred_model_name, source_mode, prompt_version


def resolve_input_csv(args):
    """Resolve an explicit CSV or infer it from the experiment scope."""
    explicit_paths = [
        path for path in (args.csv_path, args.csv_path_override)
        if path is not None
    ]
    if len(explicit_paths) > 1:
        raise ValueError(
            "Specify the input CSV either positionally or with --csv-path, not both."
        )
    if explicit_paths:
        return explicit_paths[0].expanduser().resolve()

    missing = [
        name for name, value in (
            ("--source-model", args.source_model),
            ("--mode", args.mode),
            ("--prompt-version", args.prompt_version),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "No CSV path was supplied. To infer it, provide "
            + ", ".join(missing)
            + "."
        )

    prompt_version = normalize_prompt_version(args.prompt_version)
    return (
        args.analysis_root
        / args.source_model
        / args.mode
        / "voting"
        / prompt_version
        / "vote_stability"
        / "tables"
        / "llm_vote_file_level.csv"
    ).expanduser().resolve()


def resolve_prompt_root(csv_path):
    """Return the .../voting/<prompt_version> directory for standard inputs."""
    if (
        csv_path.parent.name == "tables"
        and csv_path.parent.parent.name == "vote_stability"
    ):
        return csv_path.parent.parent.parent
    return csv_path.parent


def parse_requested_run_labels(raw_values):
    labels = []
    for raw in raw_values or []:
        labels.extend(part.strip() for part in raw.split(",") if part.strip())
    return list(dict.fromkeys(labels))


def sample_rows_mixed(df, sample_size, seed):
    """
    Sample across run labels while preferring distinct game IDs.

    The original DataFrame index is preserved so resume keys and metadata
    remain stable.
    """
    if sample_size <= 0:
        return df.iloc[0:0].copy()

    rng = random.Random(seed)
    shuffled_indices = list(df.index)
    rng.shuffle(shuffled_indices)
    shuffled = df.loc[shuffled_indices]

    run_labels = list(shuffled["run_label"].astype(str).unique())
    rng.shuffle(run_labels)

    queues = {
        label: list(shuffled[shuffled["run_label"].astype(str) == label].index)
        for label in run_labels
    }

    selected = []
    selected_games = set()

    # First pass: mix runs and avoid repeated games when possible.
    while len(selected) < sample_size:
        progressed = False
        for label in run_labels:
            queue = queues[label]
            chosen_position = None
            for pos, idx in enumerate(queue):
                game_id = safe_str(df.loc[idx, "game_id"])
                if game_id not in selected_games:
                    chosen_position = pos
                    break

            if chosen_position is None:
                continue

            idx = queue.pop(chosen_position)
            selected.append(idx)
            selected_games.add(safe_str(df.loc[idx, "game_id"]))
            progressed = True

            if len(selected) >= sample_size:
                break

        if not progressed:
            break

    # Fill any remaining slots without the distinct-game constraint.
    if len(selected) < sample_size:
        remaining = [idx for idx in shuffled_indices if idx not in set(selected)]
        selected.extend(remaining[: sample_size - len(selected)])

    return df.loc[selected]


def parse_json_response(response_text):
    if response_text is None:
        raise ValueError("Model response content is None")

    text = response_text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


# ============================================================
# Lossless normalization and graph validation
# (entirely unchanged from the Gemini version -- none of this depends on
# which provider produced the graph)
# ============================================================

def _record_repair(repairs, repair_type, description, **details):
    record = {
        "type": repair_type,
        "description": description,
    }
    record.update(details)
    repairs.append(record)


def _trim_string(value):
    return value.strip() if isinstance(value, str) else value


def normalize_graph_losslessly(raw_graph):
    """
    Produce an analysis-ready copy without changing argumentative meaning.

    The original model output remains untouched in `raw_graph`. This function
    only performs identifier/format normalization and removes exact duplicates.
    Semantic problems are left in place and reported by validate_graph().
    """
    graph = copy.deepcopy(raw_graph)
    repairs = []

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    groups = graph.get("inference_groups", [])

    if not isinstance(nodes, list) or not isinstance(edges, list) or not isinstance(groups, list):
        # Structured output should make this unreachable for Gemini; for
        # DeepSeek's looser json_object mode, this guard is more likely to
        # actually fire in practice -- worth watching how often it does.
        return graph, repairs

    # 1. Strip outer whitespace only. Internal text is never rewritten.
    trimmed_fields = []
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        for field in ("id", "argument_role", "text"):
            old = node.get(field)
            new = _trim_string(old)
            if new != old:
                node[field] = new
                trimmed_fields.append(f"nodes[{index}].{field}")

    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            continue
        for field in ("source", "target", "relation", "inference_group"):
            old = edge.get(field)
            new = _trim_string(old)
            if new != old:
                edge[field] = new
                trimmed_fields.append(f"edges[{index}].{field}")

    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            continue
        for field in ("inference_group", "target", "relation", "reasoning_type"):
            old = group.get(field)
            new = _trim_string(old)
            if new != old:
                group[field] = new
                trimmed_fields.append(f"inference_groups[{index}].{field}")
        sources = group.get("sources")
        if isinstance(sources, list):
            new_sources = [_trim_string(source) for source in sources]
            if new_sources != sources:
                group["sources"] = new_sources
                trimmed_fields.append(f"inference_groups[{index}].sources")

    if trimmed_fields:
        _record_repair(
            repairs,
            "outer_whitespace_trimmed",
            "Removed outer whitespace only; internal wording was unchanged.",
            count=len(trimmed_fields),
            fields=trimmed_fields,
        )

    # 2. Remove byte-for-byte equivalent node records. Nodes with different
    # IDs or content are never merged.
    unique_nodes = []
    seen_node_records = set()
    removed_nodes = []
    for node in nodes:
        if not isinstance(node, dict):
            unique_nodes.append(node)
            continue
        key = json.dumps(node, sort_keys=True, ensure_ascii=False)
        if key in seen_node_records:
            removed_nodes.append(copy.deepcopy(node))
            continue
        seen_node_records.add(key)
        unique_nodes.append(node)
    nodes = unique_nodes
    graph["nodes"] = nodes
    if removed_nodes:
        _record_repair(
            repairs,
            "exact_duplicate_node_removed",
            "Removed repeated copies of an identical node record; no proposition was merged.",
            count=len(removed_nodes),
            removed=removed_nodes,
        )

    # 3. Canonicalize node IDs only when every current ID is non-empty and
    # unique. All references are updated through the same bijective mapping.
    current_node_ids = [
        node.get("id") for node in nodes if isinstance(node, dict)
    ]
    can_renumber_nodes = (
        len(current_node_ids) == len(nodes)
        and all(isinstance(node_id, str) and node_id for node_id in current_node_ids)
        and len(current_node_ids) == len(set(current_node_ids))
    )
    node_id_map = {}
    if can_renumber_nodes:
        node_id_map = {
            old_id: f"n{index}"
            for index, old_id in enumerate(current_node_ids, start=1)
        }
        if any(old != new for old, new in node_id_map.items()):
            for node in nodes:
                node["id"] = node_id_map[node["id"]]
            for edge in edges:
                if not isinstance(edge, dict):
                    continue
                if edge.get("source") in node_id_map:
                    edge["source"] = node_id_map[edge["source"]]
                if edge.get("target") in node_id_map:
                    edge["target"] = node_id_map[edge["target"]]
            for group in groups:
                if not isinstance(group, dict):
                    continue
                if group.get("target") in node_id_map:
                    group["target"] = node_id_map[group["target"]]
                if isinstance(group.get("sources"), list):
                    group["sources"] = [
                        node_id_map.get(source, source)
                        for source in group["sources"]
                    ]
            _record_repair(
                repairs,
                "node_ids_renumbered",
                "Renumbered unique node IDs by node order and updated every reference; graph content and topology were unchanged.",
                mapping=node_id_map,
            )

    # 4. Remove exact duplicate edges only. Different relations or group IDs
    # for the same pair are retained and flagged as semantic conflicts.
    unique_edges = []
    seen_edge_records = set()
    removed_edges = []
    for edge in edges:
        if not isinstance(edge, dict):
            unique_edges.append(edge)
            continue
        key = json.dumps(edge, sort_keys=True, ensure_ascii=False)
        if key in seen_edge_records:
            removed_edges.append(copy.deepcopy(edge))
            continue
        seen_edge_records.add(key)
        unique_edges.append(edge)
    edges = unique_edges
    graph["edges"] = edges
    if removed_edges:
        _record_repair(
            repairs,
            "exact_duplicate_edge_removed",
            "Removed repeated copies of an identical edge; no relation was inferred or deleted selectively.",
            count=len(removed_edges),
            removed=removed_edges,
        )

    # 5. Renumber non-empty group IDs by first appearance in the edge list.
    old_group_order = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        group_id = edge.get("inference_group")
        if isinstance(group_id, str) and group_id and group_id not in old_group_order:
            old_group_order.append(group_id)
    group_id_map = {
        old_id: f"g{index}"
        for index, old_id in enumerate(old_group_order, start=1)
    }
    if group_id_map and any(old != new for old, new in group_id_map.items()):
        for edge in edges:
            if isinstance(edge, dict) and edge.get("inference_group") in group_id_map:
                edge["inference_group"] = group_id_map[edge["inference_group"]]
        for group in groups:
            if isinstance(group, dict) and group.get("inference_group") in group_id_map:
                group["inference_group"] = group_id_map[group["inference_group"]]
        _record_repair(
            repairs,
            "inference_group_ids_renumbered",
            "Renumbered inference-group identifiers by first edge appearance; group membership was unchanged.",
            mapping=group_id_map,
        )

    # 6. Edges are authoritative for group membership. Call 3 only assigns
    # reasoning_type, so its copied sources/target/relation fields are safely
    # canonicalized from the fixed edge list.
    expected_groups, _ = build_expected_inference_groups(edges)
    expected_group_ids = set(expected_groups)
    canonical_groups = []
    canonicalized_group_ids = []
    removed_extra_groups = []

    for group in groups:
        if not isinstance(group, dict):
            canonical_groups.append(group)
            continue
        group_id = group.get("inference_group")
        if group_id not in expected_group_ids:
            removed_extra_groups.append(copy.deepcopy(group))
            continue
        expected = expected_groups[group_id]
        canonical_group = copy.deepcopy(group)
        old_structure = {
            "sources": canonical_group.get("sources"),
            "target": canonical_group.get("target"),
            "relation": canonical_group.get("relation"),
        }
        new_structure = {
            "sources": list(expected["sources"]),
            "target": expected["target"],
            "relation": expected["relation"],
        }
        canonical_group.update(new_structure)
        if old_structure != new_structure:
            canonicalized_group_ids.append(group_id)
        canonical_groups.append(canonical_group)

    if canonicalized_group_ids:
        _record_repair(
            repairs,
            "inference_group_structure_canonicalized",
            "Copied sources, target, and relation from the fixed edges. The reasoning_type label was not changed.",
            groups=list(dict.fromkeys(canonicalized_group_ids)),
        )

    if removed_extra_groups:
        _record_repair(
            repairs,
            "orphan_inference_group_classification_removed",
            "Removed call-3 classifications whose inference_group had no edge. No graph relation was removed.",
            count=len(removed_extra_groups),
            removed=removed_extra_groups,
        )

    # 7. Remove exact duplicate classification records after canonicalization.
    deduplicated_groups = []
    seen_group_records = set()
    removed_group_records = []
    for group in canonical_groups:
        if not isinstance(group, dict):
            deduplicated_groups.append(group)
            continue
        key = json.dumps(group, sort_keys=True, ensure_ascii=False)
        if key in seen_group_records:
            removed_group_records.append(copy.deepcopy(group))
            continue
        seen_group_records.add(key)
        deduplicated_groups.append(group)

    if removed_group_records:
        _record_repair(
            repairs,
            "exact_duplicate_inference_group_removed",
            "Removed repeated copies of an identical inference-group classification; the reasoning label was unchanged.",
            count=len(removed_group_records),
            removed=removed_group_records,
        )

    # Preserve edge-defined group order while retaining any conflicting
    # duplicate labels so validation can surface them instead of choosing one.
    groups_by_id = OrderedDict()
    malformed_groups = []
    for group in deduplicated_groups:
        if not isinstance(group, dict) or not group.get("inference_group"):
            malformed_groups.append(group)
            continue
        groups_by_id.setdefault(group["inference_group"], []).append(group)

    ordered_groups = []
    for group_id in expected_groups:
        ordered_groups.extend(groups_by_id.get(group_id, []))
    ordered_groups.extend(malformed_groups)
    graph["inference_groups"] = ordered_groups

    return graph, repairs


def detect_cycle(nodes, edges):
    node_ids = {
        node.get("id")
        for node in nodes
        if isinstance(node, dict) and node.get("id")
    }
    adjacency = {node_id: [] for node_id in node_ids}

    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = edge.get("source")
        target = edge.get("target")
        if source in adjacency and target in node_ids:
            adjacency[source].append(target)

    white, gray, black = 0, 1, 2
    color = {node_id: white for node_id in node_ids}
    cycle_path = []

    def dfs(node_id, path):
        color[node_id] = gray
        path.append(node_id)

        for neighbor in adjacency.get(node_id, []):
            if color[neighbor] == gray:
                start = path.index(neighbor)
                cycle_path.extend(path[start:] + [neighbor])
                return True
            if color[neighbor] == white and dfs(neighbor, path):
                return True

        path.pop()
        color[node_id] = black
        return False

    for node_id in node_ids:
        if color[node_id] == white and dfs(node_id, []):
            break

    return cycle_path


def build_expected_inference_groups(edges):
    """
    Build the inference groups implied by the edge list, preserving edge order.

    Returns:
        OrderedDict[group_id] = {
            "sources": [...],
            "target": ...,
            "relation": ...,
        }
        issues = [...]
    """
    groups = OrderedDict()
    issues = []

    for edge in edges:
        if not isinstance(edge, dict):
            continue

        group_id = edge.get("inference_group")
        source = edge.get("source")
        target = edge.get("target")
        relation = edge.get("relation")

        if not group_id:
            continue

        if group_id not in groups:
            groups[group_id] = {
                "sources": [source],
                "target": target,
                "relation": relation,
            }
            continue

        group = groups[group_id]
        if group["target"] != target:
            issues.append(
                f"inference group '{group_id}' mixes targets "
                f"'{group['target']}' and '{target}'"
            )
        if group["relation"] != relation:
            issues.append(
                f"inference group '{group_id}' mixes relations "
                f"'{group['relation']}' and '{relation}'"
            )
        if source not in group["sources"]:
            group["sources"].append(source)

    return groups, issues


def validate_graph(graph):
    issues = []

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    inference_groups = graph.get("inference_groups", [])

    if not isinstance(nodes, list):
        return ["nodes is not a list"]
    if not isinstance(edges, list):
        return ["edges is not a list"]
    if not isinstance(inference_groups, list):
        return ["inference_groups is not a list"]

    node_ids = []
    for node in nodes:
        if not isinstance(node, dict):
            issues.append("a node entry is not an object")
            continue

        node_id = node.get("id")
        node_ids.append(node_id)

        if not node_id:
            issues.append("a node has an empty id")
        if node.get("argument_role") not in ALLOWED_ARGUMENT_ROLES:
            issues.append(
                f"node '{node_id}' has invalid argument_role "
                f"'{node.get('argument_role')}'"
            )
        if not safe_str(node.get("text")):
            issues.append(f"node '{node_id}' has empty text")

    if len(node_ids) != len(set(node_ids)):
        issues.append("duplicate node IDs")

    expected_ids = [f"n{i}" for i in range(1, len(nodes) + 1)]
    if node_ids != expected_ids:
        issues.append(
            f"node IDs are not sequential in input order: "
            f"expected {expected_ids}, got {node_ids}"
        )

    node_id_set = set(node_ids)
    seen_edges = set()
    seen_pairs = set()

    for edge in edges:
        if not isinstance(edge, dict):
            issues.append("an edge entry is not an object")
            continue

        source = edge.get("source")
        target = edge.get("target")
        relation = edge.get("relation")
        group_id = edge.get("inference_group")

        if source not in node_id_set:
            issues.append(f"edge source '{source}' does not exist")
        if target not in node_id_set:
            issues.append(f"edge target '{target}' does not exist")
        if source == target:
            issues.append(f"self-edge on '{source}'")
        if relation not in ALLOWED_RELATIONS:
            issues.append(f"edge relation '{relation}' is not allowed")
        if not safe_str(group_id):
            issues.append(f"edge {source}->{target} has no inference_group")

        edge_key = (source, target, relation, group_id)
        if edge_key in seen_edges:
            issues.append(f"duplicate edge: {edge_key}")
        seen_edges.add(edge_key)

        pair_key = (source, target)
        if pair_key in seen_pairs:
            issues.append(
                f"source-target pair {source}->{target} appears more than once"
            )
        seen_pairs.add(pair_key)

    conclusion_ids = [
        node.get("id")
        for node in nodes
        if isinstance(node, dict)
        and node.get("argument_role") == "Conclusion"
    ]

    if len(conclusion_ids) > 1:
        issues.append(f"multiple Conclusion nodes: {conclusion_ids}")

    for conclusion_id in conclusion_ids:
        outgoing = [
            edge
            for edge in edges
            if isinstance(edge, dict)
            and edge.get("source") == conclusion_id
        ]
        if outgoing:
            issues.append(
                f"Conclusion '{conclusion_id}' has {len(outgoing)} outgoing edge(s)"
            )

    cycle = detect_cycle(nodes, edges)
    if cycle:
        issues.append(f"graph contains a cycle: {' -> '.join(cycle)}")

    expected_groups, group_structure_issues = build_expected_inference_groups(edges)
    issues.extend(group_structure_issues)

    returned_groups = OrderedDict()
    for group in inference_groups:
        if not isinstance(group, dict):
            issues.append("an inference-group entry is not an object")
            continue

        group_id = group.get("inference_group")
        if group_id in returned_groups:
            issues.append(f"duplicate inference-group classification: '{group_id}'")
            continue

        returned_groups[group_id] = group

        reasoning_type = group.get("reasoning_type")
        if reasoning_type not in ALLOWED_REASONING_TYPES:
            issues.append(
                f"inference group '{group_id}' has invalid reasoning_type "
                f"'{reasoning_type}'"
            )

    missing_groups = [
        group_id for group_id in expected_groups if group_id not in returned_groups
    ]
    extra_groups = [
        group_id for group_id in returned_groups if group_id not in expected_groups
    ]

    if missing_groups:
        issues.append(
            f"call 3 omitted inference group(s): {missing_groups}"
        )
    if extra_groups:
        issues.append(
            f"call 3 returned unknown inference group(s): {extra_groups}"
        )

    for group_id, expected in expected_groups.items():
        returned = returned_groups.get(group_id)
        if returned is None:
            continue

        if returned.get("target") != expected["target"]:
            issues.append(
                f"inference group '{group_id}' has target "
                f"'{returned.get('target')}', expected '{expected['target']}'"
            )
        if returned.get("relation") != expected["relation"]:
            issues.append(
                f"inference group '{group_id}' has relation "
                f"'{returned.get('relation')}', expected '{expected['relation']}'"
            )

        returned_sources = returned.get("sources")
        if returned_sources != expected["sources"]:
            issues.append(
                f"inference group '{group_id}' has sources "
                f"{returned_sources}, expected {expected['sources']}"
            )

    if not edges and inference_groups:
        issues.append("inference_groups must be empty when edges is empty")

    return issues


# ============================================================
# DeepSeek calls (OpenAI-compatible client)
# ============================================================

def build_generation_config(
    schema,
    *,
    generation_seed=DEFAULT_GENERATION_SEED,
    reasoning_effort=DEFAULT_REASONING_EFFORT,
):
    """
    Returns a plain dict of kwargs for client.chat.completions.create(),
    mirroring build_generation_config()'s role in the Gemini version.
    There is no cache object to attach here -- DeepSeek's caching is
    automatic and requires no per-call configuration.

    temperature and top_p are deliberately NOT included: DeepSeek's docs
    state both are ignored by the API in thinking mode. reasoning_effort
    and extra_body={"thinking": {"type": "enabled"}} are DeepSeek's
    documented, explicit controls for reasoning depth -- using them makes
    the reasoning configuration an explicit, auditable choice (recorded in
    the manifest) rather than relying on an assumed default.
    """
    return {
        "seed": generation_seed,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "_max_tokens_on_length_cap": DEFAULT_MAX_TOKENS_ON_LENGTH_CAP,
        "response_format": {"type": "json_object"},
        "reasoning_effort": reasoning_effort,
        "extra_body": {"thinking": {"type": "enabled"}},
        "_schema_reminder": schema_reminder_text(schema),
    }


def is_retryable_error(error):
    text = str(error).lower()
    markers = [
        "429",
        "rate limit",
        "500",
        "502",
        "503",
        "504",
        "deadline",
        "timeout",
        "temporarily",
        "overloaded",
    ]
    return any(marker in text for marker in markers)


class RetryableModelOutputError(RuntimeError):
    """The API returned a response, but its final JSON content was unusable."""

    def __init__(self, message, *, length_limited=False):
        super().__init__(message)
        self.length_limited = length_limited


def call_model(
    client,
    model,
    stage,
    system_prompt,
    user_message,
    config,
    max_retries,
    retry_sleep_seconds,
):
    """
    Call DeepSeek and return JSON text.

    Empty content and malformed JSON are validated inside the retry loop.
    When a response ends because it reached max_tokens, the next attempt
    doubles the ceiling up to the configured cap. Other transient failures
    retry with the same ceiling.
    """
    schema_reminder = config.get("_schema_reminder", "")
    max_tokens_cap = int(
        config.get(
            "_max_tokens_on_length_cap",
            DEFAULT_MAX_TOKENS_ON_LENGTH_CAP,
        )
    )

    call_kwargs = {
        key: value
        for key, value in config.items()
        if not key.startswith("_")
    }

    current_max_tokens = int(
        call_kwargs.get("max_tokens", DEFAULT_MAX_TOKENS)
    )

    # Put all stable text before the variable input to maximize prefix caching.
    messages = [
        {
            "role": "system",
            "content": system_prompt + schema_reminder,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

    last_error = None

    for attempt in range(max_retries + 1):
        attempt_kwargs = dict(call_kwargs)
        attempt_kwargs["max_tokens"] = current_max_tokens

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                **attempt_kwargs,
            )

            if not response.choices:
                raise RetryableModelOutputError(
                    f"{stage}: API response contained no choices"
                )

            choice = response.choices[0]
            message = choice.message
            content = message.content or ""
            finish_reason = getattr(choice, "finish_reason", None)
            reasoning_content = (
                getattr(message, "reasoning_content", None) or ""
            )

            usage = getattr(response, "usage", None)
            completion_tokens = (
                getattr(usage, "completion_tokens", None)
                if usage is not None
                else None
            )

            length_limited = finish_reason == "length"

            if not content.strip():
                raise RetryableModelOutputError(
                    f"{stage}: empty model content "
                    f"(finish_reason={finish_reason!r}, "
                    f"max_tokens={current_max_tokens}, "
                    f"completion_tokens={completion_tokens!r}, "
                    f"reasoning_chars={len(reasoning_content)})",
                    length_limited=length_limited,
                )

            try:
                # Validate here so malformed output is retried at this stage.
                parse_json_response(content)
            except (json.JSONDecodeError, ValueError, TypeError) as error:
                preview = content[:300].replace("\n", "\\n")
                raise RetryableModelOutputError(
                    f"{stage}: invalid JSON "
                    f"(finish_reason={finish_reason!r}, "
                    f"max_tokens={current_max_tokens}, "
                    f"completion_tokens={completion_tokens!r}, "
                    f"preview={preview!r}): {error}",
                    length_limited=length_limited,
                ) from error

            return content

        except Exception as error:
            last_error = error
            retryable = (
                isinstance(error, RetryableModelOutputError)
                or is_retryable_error(error)
            )

            if attempt >= max_retries or not retryable:
                raise

            next_max_tokens = current_max_tokens

            if (
                isinstance(error, RetryableModelOutputError)
                and error.length_limited
                and current_max_tokens < max_tokens_cap
            ):
                next_max_tokens = min(
                    current_max_tokens * 2,
                    max_tokens_cap,
                )

            if next_max_tokens > current_max_tokens:
                print(
                    f"    Retrying {stage} after attempt "
                    f"{attempt + 1}/{max_retries + 1}: {error}"
                )
                print(
                    f"    Increasing max_tokens for {stage}: "
                    f"{current_max_tokens} -> {next_max_tokens}"
                )
                current_max_tokens = next_max_tokens
            else:
                print(
                    f"    Retrying {stage} after attempt "
                    f"{attempt + 1}/{max_retries + 1}: {error}"
                )

            time.sleep(retry_sleep_seconds)

    raise last_error


def run_three_call(
    client,
    nodes_model,
    relations_model,
    reasoning_model,
    justification,
    nodes_prompt,
    relations_prompt,
    reasoning_prompt,
    nodes_config,
    relations_config,
    reasoning_type_config,
    max_retries,
    retry_sleep_seconds,
    stage_sleep_seconds,
):
    # Call 1: nodes and argument roles.
    message_1 = (
        "INPUT_JSON:\n"
        f"{json.dumps({'justification': justification}, ensure_ascii=False, indent=2)}\n\n"
        "OUTPUT_JSON:\n"
    )
    text_1 = call_model(
        client,
        nodes_model,
        "nodes",
        nodes_prompt,
        message_1,
        nodes_config,
        max_retries,
        retry_sleep_seconds,
    )
    nodes = parse_json_response(text_1).get("nodes", [])

    if stage_sleep_seconds > 0:
        time.sleep(stage_sleep_seconds)

    # Call 2: fixed graph structure.
    message_2 = (
        "INPUT_JSON:\n"
        f"{json.dumps({'justification': justification, 'nodes': nodes}, ensure_ascii=False, indent=2)}\n\n"
        "OUTPUT_JSON:\n"
    )
    text_2 = call_model(
        client,
        relations_model,
        "relations",
        relations_prompt,
        message_2,
        relations_config,
        max_retries,
        retry_sleep_seconds,
    )
    edges = parse_json_response(text_2).get("edges", [])

    if stage_sleep_seconds > 0:
        time.sleep(stage_sleep_seconds)

    # Call 3: one reasoning type per fixed inference group. It receives the
    # exact outputs of Calls 1 and 2; normalization happens only afterward.
    message_3 = (
        "INPUT_JSON:\n"
        f"{json.dumps({'justification': justification, 'nodes': nodes, 'edges': edges}, ensure_ascii=False, indent=2)}\n\n"
        "OUTPUT_JSON:\n"
    )
    text_3 = call_model(
        client,
        reasoning_model,
        "reasoning_types",
        reasoning_prompt,
        message_3,
        reasoning_type_config,
        max_retries,
        retry_sleep_seconds,
    )
    inference_groups = parse_json_response(text_3).get("inference_groups", [])

    if stage_sleep_seconds > 0:
        time.sleep(stage_sleep_seconds)

    raw_graph = {
        "nodes": nodes,
        "edges": edges,
        "inference_groups": inference_groups,
    }
    raw_validation_flags = validate_graph(raw_graph)

    # Analysis graph: only lossless, fully auditable normalization.
    graph, repairs = normalize_graph_losslessly(raw_graph)
    validation_flags = validate_graph(graph)

    return raw_graph, graph, repairs, raw_validation_flags, validation_flags


# ============================================================
# CLI, output scoping, and resume
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract argument graphs with a three-call DeepSeek pipeline "
            "(OpenAI-compatible API). All three stages use one DeepSeek API key."
        )
    )

    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "Optional path to llm_vote_file_level.csv. If omitted, the path "
            "is inferred from --source-model, --mode, and --prompt-version."
        ),
    )
    parser.add_argument(
        "--csv-path",
        dest="csv_path_override",
        type=Path,
        default=None,
        help="Explicit CSV path override. Do not combine with the positional path.",
    )
    parser.add_argument(
        "--source-model",
        default=None,
        help=(
            "Directory name of the voting model whose justifications are being "
            "analysed, for example unsloth_gemma-4-31B-it-unsloth-bnb-4bit."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["base", "finetuned"],
        default=None,
        help="Source-model mode used in the analysis directory.",
    )
    parser.add_argument(
        "--prompt-version",
        default=None,
        help="Voting prompt version, for example 4, v4, or prompt_v4.",
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=Path("analysis"),
        help="Root analysis directory used when the CSV path is inferred.",
    )
    parser.add_argument(
        "--nodes-only-prompt",
        type=Path,
        default=NODES_ONLY_PROMPT_PATH,
    )
    parser.add_argument(
        "--relations-only-prompt",
        type=Path,
        default=RELATIONS_ONLY_PROMPT_PATH,
    )
    parser.add_argument(
        "--reasoning-type-only-prompt",
        type=Path,
        default=REASONING_TYPE_ONLY_PROMPT_PATH,
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Backward-compatible override: use one DeepSeek model for all "
            "three stages. Stage-specific options take precedence."
        ),
    )
    parser.add_argument(
        "--nodes-model",
        default=None,
        help=f"DeepSeek model for node extraction. Default: {DEFAULT_NODES_MODEL}.",
    )
    parser.add_argument(
        "--relations-model",
        default=None,
        help=(
            "DeepSeek model for relation and inference-group extraction. "
            f"Default: {DEFAULT_RELATIONS_MODEL}."
        ),
    )
    parser.add_argument(
        "--reasoning-model",
        default=None,
        help=f"DeepSeek model for reasoning-type classification. Default: {DEFAULT_REASONING_MODEL}.",
    )
    parser.add_argument(
        "--run-label",
        action="append",
        default=[],
        help=(
            "Exact run_label to process. Repeat the option or pass a "
            "comma-separated list. For a full extraction, exactly one run "
            "must remain after filtering."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Optional explicit JSONL path. By default, mixed samples and "
            "full runs are saved in separate scoped folders."
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help=(
            "Run a mixed pilot sample of N rows. Without --run-label, rows "
            "are mixed across all available run labels."
        ),
    )
    parser.add_argument(
        "--sample-seed",
        "--seed",
        dest="sample_seed",
        type=int,
        default=42,
        help="Random seed used only for pilot sampling.",
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=None,
        help=(
            "Fail before extraction unless the selected run contains exactly "
            "this many rows. Use 191 for the full one-run extraction."
        ),
    )
    # No --temperature / --top-p flags: DeepSeek ignores both in thinking
    # mode (always enabled by this runner), so exposing them would be
    # misleading -- they would silently do nothing.
    parser.add_argument(
        "--generation-seed",
        type=int,
        default=DEFAULT_GENERATION_SEED,
        help=(
            "Best-effort generation seed, same caveat as the Gemini version: "
            "OpenAI-compatible APIs document seed as best-effort, not "
            "guaranteed bit-for-bit reproducibility."
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        default=DEFAULT_REASONING_EFFORT,
        help=(
            "DeepSeek reasoning_effort value (confirmed valid: 'high'; "
            "other values are unverified by analogy to OpenAI's convention)."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--stage-sleep-seconds",
        type=float,
        default=float(
            os.environ.get(
                "ARGMINING_STAGE_SLEEP_SECONDS",
                str(DEFAULT_STAGE_SLEEP_SECONDS),
            )
        ),
        help="Optional delay after each successful stage. Default: 0 seconds.",
    )
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-sleep-seconds", type=float, default=30.0)

    args = parser.parse_args()

    # Resolve stage models. A legacy --model applies to all stages,
    # while explicit stage-specific options override it.
    args.nodes_model = (
        args.nodes_model
        or args.model
        or os.environ.get("ARGMINING_NODES_MODEL")
        or DEFAULT_NODES_MODEL
    )
    args.relations_model = (
        args.relations_model
        or args.model
        or os.environ.get("ARGMINING_RELATIONS_MODEL")
        or DEFAULT_RELATIONS_MODEL
    )
    args.reasoning_model = (
        args.reasoning_model
        or args.model
        or os.environ.get("ARGMINING_REASONING_MODEL")
        or DEFAULT_REASONING_MODEL
    )

    if args.resume and args.overwrite:
        parser.error("Use either --resume or --overwrite, not both.")
    if args.csv_path is not None and args.csv_path_override is not None:
        parser.error(
            "Specify the input CSV either positionally or with --csv-path, not both."
        )
    uses_inferred_path = args.csv_path is None and args.csv_path_override is None
    if uses_inferred_path:
        missing_scope = [
            option for option, value in (
                ("--source-model", args.source_model),
                ("--mode", args.mode),
                ("--prompt-version", args.prompt_version),
            )
            if not value
        ]
        if missing_scope:
            parser.error(
                "When no CSV path is supplied, provide "
                + ", ".join(missing_scope)
                + "."
            )
    if args.sample_size is not None and args.sample_size <= 0:
        parser.error("--sample-size must be greater than zero.")
    if args.expected_rows is not None and args.expected_rows <= 0:
        parser.error("--expected-rows must be greater than zero.")
    if args.stage_sleep_seconds < 0:
        parser.error("--stage-sleep-seconds cannot be negative.")

    # Safe default: append to an existing compatible output.
    if not args.overwrite:
        args.resume = True

    return args


def resolve_default_output_path(csv_path, df, args):
    root = resolve_prompt_root(csv_path) / "justification_analysis" / "three_call_deepseek"
    run_labels = sorted(df["run_label"].astype(str).unique())

    if args.sample_size is not None:
        scope = "mixed" if len(run_labels) > 1 else slugify(run_labels[0])
        folder = (
            root
            / "samples"
            / f"{scope}_n{args.sample_size}_seed{args.sample_seed}"
        )
        return folder / "graphs.jsonl"

    if len(run_labels) != 1:
        raise ValueError(
            "A full extraction must contain exactly one run_label. "
            f"Selected labels: {run_labels}. Pass --run-label <label>."
        )

    return root / "runs" / slugify(run_labels[0]) / "graphs.jsonl"


def make_row_key(row_index, game_id, run_label, chosen_vote):
    return (
        str(row_index),
        safe_str(game_id),
        safe_str(run_label),
        normalize_chosen_vote(chosen_vote),
    )


def load_completed_rows(output_path):
    completed_keys = set()
    stats = {
        "lines": 0,
        "complete": 0,
        "errors": 0,
        "malformed": 0,
    }

    if not output_path.exists():
        return completed_keys, stats

    with output_path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            line = line.strip()
            if not line:
                continue

            stats["lines"] += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["malformed"] += 1
                continue

            metadata = record.get("metadata", {})
            has_error = "error" in metadata or "error" in record
            has_graph = isinstance(record.get("graph"), dict)

            if has_error or not has_graph:
                stats["errors"] += 1
                continue

            completed_keys.add(
                make_row_key(
                    metadata.get("row_index"),
                    metadata.get("game_id", ""),
                    metadata.get("run_label", ""),
                    metadata.get("chosen_vote", ""),
                )
            )
            stats["complete"] += 1

    return completed_keys, stats


def write_jsonl(output_file, record):
    output_file.write(
        json.dumps(record, ensure_ascii=False, default=str) + "\n"
    )
    output_file.flush()


def build_manifest(
    *,
    csv_path,
    output_path,
    run_labels,
    row_count_before_sampling,
    selected_row_count,
    args,
    inferred_model_name,
    source_mode,
    prompt_version,
    schema_version,
    prompt_paths,
):
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(csv_path),
        "output_jsonl": str(output_path),
        "source_model": inferred_model_name,
        "source_mode": source_mode,
        "source_prompt_version": prompt_version,
        "selected_run_labels": run_labels,
        "row_count_before_sampling": row_count_before_sampling,
        "selected_row_count": selected_row_count,
        "sample_size": args.sample_size,
        "sample_seed": args.sample_seed if args.sample_size is not None else None,
        "api_provider": "deepseek",
        "extractor_model": (
            args.nodes_model
            if args.nodes_model == args.relations_model == args.reasoning_model
            else "mixed-stage-models"
        ),
        "extractor_models": {
            "nodes": args.nodes_model,
            "relations": args.relations_model,
            "reasoning_types": args.reasoning_model,
        },
        "stage_sleep_seconds": {
            "nodes": args.stage_sleep_seconds,
            "relations": args.stage_sleep_seconds,
            "reasoning_types": args.stage_sleep_seconds,
        },
        "generation": {
            # temperature/top_p are deliberately absent: DeepSeek ignores
            # both in thinking mode, so recording a value here would
            # misrepresent what the API actually did. Decoding behavior in
            # thinking mode is governed by reasoning_effort instead.
            "decoding": "thinking_mode (temperature/top_p not applicable)",
            "seed": args.generation_seed,
            # Explicitly enabled via extra_body={"thinking": {"type":
            # "enabled"}}, per DeepSeek's documented example, rather than
            # relying on an assumed default.
            "thinking_mode": "explicitly enabled (extra_body)",
            "reasoning_effort": args.reasoning_effort,
            # UNCONFIRMED whether DeepSeek enforces this schema at the API
            # level the way Gemini's response_json_schema does, or only
            # guarantees valid JSON via response_format=json_object. See
            # the comment above NODES_ONLY_SCHEMA. Recorded honestly as
            # "requested", not "enforced".
            "structured_output_requested": True,
            "structured_output_mode": "json_object (schema not API-enforced; see prompt + validate_graph)",
        },
        "schema_version": schema_version,
        "prompt_paths": [str(path) for path in prompt_paths],
        "post_processing": REPAIR_POLICY,
    }


def manifest_signature(manifest):
    return {
        "input_csv": manifest.get("input_csv"),
        "source_model": manifest.get("source_model"),
        "source_mode": manifest.get("source_mode"),
        "source_prompt_version": manifest.get("source_prompt_version"),
        "selected_run_labels": manifest.get("selected_run_labels"),
        "sample_size": manifest.get("sample_size"),
        "sample_seed": manifest.get("sample_seed"),
        "api_provider": manifest.get("api_provider"),
        "extractor_model": manifest.get("extractor_model"),
        "extractor_models": manifest.get("extractor_models"),
        "stage_sleep_seconds": manifest.get("stage_sleep_seconds"),
        "generation": manifest.get("generation"),
        "schema_version": manifest.get("schema_version"),
        "post_processing_version": (
            manifest.get("post_processing", {}).get("version")
        ),
    }


def write_or_check_manifest(manifest_path, manifest, overwrite):
    if manifest_path.exists() and not overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_signature(existing) != manifest_signature(manifest):
            raise ValueError(
                "The existing output manifest uses different prompts, scope, "
                "or generation settings. Use --overwrite or a different "
                "--output-path."
            )
        return

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    csv_path = resolve_input_csv(args)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {csv_path}\n"
            "Check --source-model, --mode, --prompt-version, or pass --csv-path."
        )

    prompt_paths = [
        args.nodes_only_prompt.resolve(),
        args.relations_only_prompt.resolve(),
        args.reasoning_type_only_prompt.resolve(),
    ]
    for prompt_path in prompt_paths:
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    nodes_prompt = prompt_paths[0].read_text(encoding="utf-8")
    relations_prompt = prompt_paths[1].read_text(encoding="utf-8")
    reasoning_prompt = prompt_paths[2].read_text(encoding="utf-8")

    schema_version = (
        f"{prompt_paths[0].name}+{prompt_paths[1].name}+"
        f"{prompt_paths[2].name}:"
        f"{prompt_content_hash(nodes_prompt, relations_prompt, reasoning_prompt)}"
    )

    inferred_model_name, source_mode, prompt_version = infer_metadata_from_path(csv_path)

    df = pd.read_csv(csv_path)
    required_columns = {
        "game_id",
        "run_label",
        "chosen_vote",
        "justification",
        "is_correct",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Input CSV is missing required columns: {sorted(missing_columns)}"
        )

    requested_run_labels = parse_requested_run_labels(args.run_label)
    available_run_labels = sorted(df["run_label"].astype(str).unique())

    if requested_run_labels:
        missing_labels = sorted(
            set(requested_run_labels) - set(available_run_labels)
        )
        if missing_labels:
            raise ValueError(
                f"Unknown run_label value(s): {missing_labels}. "
                f"Available: {available_run_labels}"
            )

        df = df[
            df["run_label"].astype(str).isin(requested_run_labels)
        ].copy()

    row_count_before_sampling = len(df)

    if args.expected_rows is not None and row_count_before_sampling != args.expected_rows:
        raise ValueError(
            f"Expected {args.expected_rows} selected rows, "
            f"found {row_count_before_sampling}."
        )

    if args.sample_size is not None:
        df = sample_rows_mixed(
            df,
            min(args.sample_size, len(df)),
            args.sample_seed,
        )

    selected_run_labels = sorted(df["run_label"].astype(str).unique())

    output_path = (
        args.output_path.resolve()
        if args.output_path is not None
        else resolve_default_output_path(csv_path, df, args)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path.parent / "manifest.json"

    manifest = build_manifest(
        csv_path=csv_path,
        output_path=output_path,
        run_labels=selected_run_labels,
        row_count_before_sampling=row_count_before_sampling,
        selected_row_count=len(df),
        args=args,
        inferred_model_name=inferred_model_name,
        source_mode=source_mode,
        prompt_version=prompt_version,
        schema_version=schema_version,
        prompt_paths=prompt_paths,
    )
    write_or_check_manifest(manifest_path, manifest, args.overwrite)

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing DEEPSEEK_API_KEY environment variable.")

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    print("--- Initialization ---")
    print(f"API provider                 : DeepSeek ({DEEPSEEK_BASE_URL})")
    print(f"Source model                 : {inferred_model_name}")
    print(f"Source mode                  : {source_mode}")
    print(f"Source prompt version        : {prompt_version}")
    print(f"Input CSV                    : {csv_path}")
    print(f"Selected run labels          : {selected_run_labels}")
    print(f"Rows before sampling         : {row_count_before_sampling}")
    print(f"Rows selected                : {len(df)}")
    print(f"Output JSONL                 : {output_path}")
    print(f"Manifest                     : {manifest_path}")
    print(f"Nodes model                  : {args.nodes_model}")
    print(f"Relations model              : {args.relations_model}")
    print(f"Reasoning model              : {args.reasoning_model}")
    print(f"Decoding                     : thinking_mode (temperature/top_p ignored by DeepSeek)")
    print(f"Generation seed              : {args.generation_seed} (best-effort)")
    print(f"Reasoning effort              : {args.reasoning_effort}")
    print(
        "Output-token policy           : "
        f"{DEFAULT_MAX_TOKENS} initial; double on length up to "
        f"{DEFAULT_MAX_TOKENS_ON_LENGTH_CAP}"
    )
    print(f"Thinking mode                 : explicitly enabled")
    print(f"Structured output mode       : json_object (NOT API-enforced schema -- see comments)")
    print(f"Caching                      : automatic prefix-based (no config needed)")
    print(f"Schema version               : {schema_version}")
    print(f"Repair policy                : {REPAIR_POLICY_VERSION}")
    print(f"Resume                       : {args.resume}")
    print(f"Overwrite                    : {args.overwrite}")
    print(f"Stage sleep seconds          : {args.stage_sleep_seconds}")
    print()

    config_kwargs = {
        "generation_seed": args.generation_seed,
        "reasoning_effort": args.reasoning_effort,
    }
    nodes_config = build_generation_config(NODES_ONLY_SCHEMA, **config_kwargs)
    relations_config = build_generation_config(RELATIONS_ONLY_SCHEMA, **config_kwargs)
    reasoning_config = build_generation_config(REASONING_TYPE_ONLY_SCHEMA, **config_kwargs)

    if args.overwrite:
        output_mode = "w"
        completed_keys = set()
        print("Overwrite requested: previous output will be replaced.\n")
    else:
        output_mode = "a"
        completed_keys, resume_stats = load_completed_rows(output_path)
        print("--- Resume scan ---")
        print(f"Existing JSONL lines    : {resume_stats['lines']}")
        print(f"Completed rows detected : {resume_stats['complete']}")
        print(f"Error rows detected     : {resume_stats['errors']}")
        print(f"Malformed lines ignored : {resume_stats['malformed']}")
        print("Only successful rows are skipped; error rows are retried.\n")

    processed_this_run = 0
    skipped_existing = 0
    failed_rows = 0
    flagged_rows = 0
    repaired_rows = 0

    with output_path.open(output_mode, encoding="utf-8") as output_file:
        for position, (index, row) in enumerate(df.iterrows(), start=1):
            if (
                args.max_rows is not None
                and processed_this_run >= args.max_rows
            ):
                print(
                    f"Reached --max-rows={args.max_rows}. Stopping early."
                )
                break

            game_id = safe_str(row["game_id"])
            run_label = safe_str(row["run_label"])
            chosen_vote = normalize_chosen_vote(row["chosen_vote"])
            justification = safe_str(row["justification"])

            row_key = make_row_key(
                index,
                game_id,
                run_label,
                chosen_vote,
            )

            if args.resume and row_key in completed_keys:
                skipped_existing += 1
                print(
                    f"Skipping row {index} ({position}/{len(df)}): "
                    "already completed."
                )
                continue

            print(
                f"Processing row {index} ({position}/{len(df)}): "
                f"{game_id} | {run_label} | vote={chosen_vote}"
            )

            try:
                (
                    raw_graph,
                    graph,
                    repairs,
                    raw_validation_flags,
                    validation_flags,
                ) = run_three_call(
                    client,
                    args.nodes_model,
                    args.relations_model,
                    args.reasoning_model,
                    justification,
                    nodes_prompt,
                    relations_prompt,
                    reasoning_prompt,
                    nodes_config,
                    relations_config,
                    reasoning_config,
                    args.max_retries,
                    args.retry_sleep_seconds,
                    args.stage_sleep_seconds,
                )

                if repairs:
                    repaired_rows += 1
                    print(
                        f"  Lossless repairs ({len(repairs)}): "
                        f"{[repair['type'] for repair in repairs]}"
                    )

                if validation_flags:
                    flagged_rows += 1
                    print(
                        f"  Flagged after normalization ({len(validation_flags)}): "
                        f"{validation_flags}"
                    )

                record = {
                    "metadata": {
                        "row_index": int(index),
                        "game_id": game_id,
                        "run_label": run_label,
                        "chosen_vote": chosen_vote,
                        "is_correct": parse_bool(row["is_correct"]),
                        "prompt_version": prompt_version,
                        "model_name": inferred_model_name,
                        "source_mode": source_mode,
                        "api_provider": "deepseek",
                        "extractor_model": manifest["extractor_model"],
                        "extractor_models": manifest["extractor_models"],
                        "stage_sleep_seconds": manifest["stage_sleep_seconds"],
                        "schema_version": schema_version,
                        "repair_policy_version": REPAIR_POLICY_VERSION,
                        "raw_validation_flags": raw_validation_flags,
                        "validation_flags": validation_flags,
                        "generation_config": manifest["generation"],
                    },
                    "input": {
                        "justification": justification,
                    },
                    "raw_graph": raw_graph,
                    "graph": graph,
                    "repairs": repairs,
                }
                write_jsonl(output_file, record)

                processed_this_run += 1
                completed_keys.add(row_key)

            except KeyboardInterrupt:
                print("\nInterrupted by user. Partial output has been saved.")
                break

            except Exception as error:
                failed_rows += 1
                print(f"Error on row {index}: {error}")

                error_record = {
                    "metadata": {
                        "row_index": int(index),
                        "game_id": game_id,
                        "run_label": run_label,
                        "chosen_vote": chosen_vote,
                        "is_correct": parse_bool(row["is_correct"]),
                        "prompt_version": prompt_version,
                        "model_name": inferred_model_name,
                        "source_mode": source_mode,
                        "api_provider": "deepseek",
                        "extractor_model": manifest["extractor_model"],
                        "extractor_models": manifest["extractor_models"],
                        "stage_sleep_seconds": manifest["stage_sleep_seconds"],
                        "schema_version": schema_version,
                        "generation_config": manifest["generation"],
                        "error": str(error),
                        "error_type": type(error).__name__,
                    },
                    "input": {
                        "justification": justification,
                    },
                }
                write_jsonl(output_file, error_record)

    print(f"\nExtraction complete: {output_path}")
    print(f"Rows processed this run : {processed_this_run}")
    print(f"Rows skipped by resume  : {skipped_existing}")
    print(f"Rows failed this run    : {failed_rows}")
    print(f"Rows with repairs       : {repaired_rows}")
    print(f"Rows flagged for review : {flagged_rows}")


if __name__ == "__main__":
    main()