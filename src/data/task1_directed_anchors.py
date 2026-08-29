from __future__ import annotations

import json
from pathlib import Path

from .directed_anchors import (
    IDENTIFIER,
    IGNORED_IDENTIFIERS,
    target_member_names,
)


def extract_method_body(source: str, method_name: str) -> str | None:
    """Return the balanced-brace body of a Java method when it can be found."""
    import re

    signature = re.compile(
        rf"\b{re.escape(method_name)}\s*\([^)]*\)\s*"
        rf"(?:throws\s+[^{{]+)?\{{"
    )
    match = signature.search(source)
    if match is None:
        return None
    opening = source.find("{", match.start())
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening:index + 1]
    return None


def class_orientation(metadata: dict) -> tuple[int, int]:
    """Resolve source and target positions in a Task-1 class-pair sample."""
    source = Path(metadata["source_class"]).stem
    target = Path(metadata["target_class"]).stem
    graph1 = Path(metadata["graph1_path"]).stem
    graph2 = Path(metadata["graph2_path"]).stem
    if graph1 == source and graph2 == target:
        return 1, 2
    if graph2 == source and graph1 == target:
        return 2, 1
    return 0, 0


def sample_task1_accessed_members(
    metadata: dict,
) -> tuple[int, tuple[str, ...], str]:
    """Extract moved-method accesses and the source→target graph orientation."""
    source_graph, target_graph = class_orientation(metadata)
    if not source_graph:
        return 0, (), "orientation_unresolved"
    graph1_path = Path(metadata["graph1_path"])
    graph2_path = Path(metadata["graph2_path"])
    source_path = graph1_path if source_graph == 1 else graph2_path
    target_path = graph1_path if target_graph == 1 else graph2_path
    info_path = graph1_path.parent / "dataItemInfo.json"
    if not source_path.exists() or not target_path.exists():
        return source_graph, (), "java_source_missing"
    if not info_path.exists():
        return source_graph, (), "item_info_missing"
    source = source_path.read_text(encoding="utf-8", errors="ignore")
    target = target_path.read_text(encoding="utf-8", errors="ignore")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    body = extract_method_body(source, str(metadata["moved_method"]))
    if body is None:
        return source_graph, (), "method_body_unresolved"
    identifiers = set(IDENTIFIER.findall(body)) - IGNORED_IDENTIFIERS
    members = target_member_names(target, info, target_path.stem)
    accessed = tuple(sorted(identifiers & members))
    return (
        source_graph,
        accessed,
        "resolved" if accessed else "no_accessed_target_member",
    )


def build_task1_directed_anchor_index(
    metadata: dict[str, dict],
    vocab: dict[str, int],
) -> tuple[dict[str, dict], dict]:
    """Build source-oriented, method-conditioned anchors for Task 1.

    Each entry identifies which graph contains the source class and restricts
    exact local matches to target members referenced by the moved method.
    """
    anchors: dict[str, dict] = {}
    status_counts: dict[str, int] = {}
    orientation_counts = {"source_graph_1": 0, "source_graph_2": 0}
    positive_samples = 0
    positive_oriented = 0
    positive_with_anchor = 0
    positive_anchor_tokens = 0
    negative_with_anchor = 0
    for line, row in metadata.items():
        source_graph, names, status = sample_task1_accessed_members(row)
        token_ids = tuple(
            sorted({int(vocab[name]) for name in names if name in vocab})
        )
        anchors[line] = {
            "source_graph": source_graph,
            "anchor_token_ids": token_ids,
            "anchor_names": names,
            "status": status,
        }
        status_counts[status] = status_counts.get(status, 0) + 1
        if source_graph:
            orientation_counts[f"source_graph_{source_graph}"] += 1
        if int(row["label"]) == 1:
            positive_samples += 1
            positive_oriented += int(bool(source_graph))
            positive_with_anchor += int(bool(token_ids))
            positive_anchor_tokens += len(token_ids)
        else:
            negative_with_anchor += int(bool(token_ids))
    return anchors, {
        "samples": len(metadata),
        "positive_samples": positive_samples,
        "positive_oriented": positive_oriented,
        "positive_orientation_coverage": (
            positive_oriented / positive_samples if positive_samples else 0.0
        ),
        "positive_with_anchor": positive_with_anchor,
        "positive_anchor_coverage": (
            positive_with_anchor / positive_samples if positive_samples else 0.0
        ),
        "mean_positive_anchor_tokens": (
            positive_anchor_tokens / positive_with_anchor
            if positive_with_anchor else 0.0
        ),
        "negative_with_anchor": negative_with_anchor,
        "orientation_counts": orientation_counts,
        "status_counts": dict(sorted(status_counts.items())),
    }


def orient_source_target(
    features_by_level,
    graph1_tokens,
    graph2_tokens,
    source_graph: int,
):
    """Normalize an arbitrary class-pair order to source class → target class."""
    if source_graph == 1:
        return features_by_level, graph1_tokens, graph2_tokens
    if source_graph != 2:
        raise ValueError("source_graph must be 1 or 2")
    oriented = {
        level: {
            "graph1": values["graph2"],
            "graph2": values["graph1"],
        }
        for level, values in features_by_level.items()
    }
    return oriented, graph2_tokens, graph1_tokens
