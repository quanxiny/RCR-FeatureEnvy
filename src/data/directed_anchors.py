from __future__ import annotations

import json
from pathlib import Path
import re


IDENTIFIER = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
FIELD_DECLARATION = re.compile(
    r"^(?:(?:public|protected|private|static|final|transient|volatile)\s+)*"
    r"(?:[\w$.[\]<>?,]+\s+)+([A-Za-z_$][A-Za-z0-9_$]*)"
    r"\s*(?:=|;|,)"
)
IGNORED_IDENTIFIERS = {
    "false",
    "null",
    "super",
    "this",
    "true",
}


def class_field_names(source: str) -> set[str]:
    """Extract class-level fields without treating method locals as members."""
    fields = set()
    depth = 0
    for raw_line in source.splitlines():
        line = re.sub(r"//.*$", "", raw_line).strip()
        if depth == 1:
            match = FIELD_DECLARATION.match(line)
            if match and "(" not in line.split("=", 1)[0]:
                fields.add(match.group(1))
        depth += line.count("{") - line.count("}")
    return fields


def target_member_names(class_source: str, info: dict, class_stem: str) -> set[str]:
    if class_stem == str(info.get("tagClassName", "")):
        methods = info.get("MethodNamesInTag", [])
    elif class_stem == str(info.get("srcClassName", "")):
        methods = info.get("MethodNamesInSrc", [])
    else:
        methods = []
    return (
        {str(name) for name in methods}
        | class_field_names(class_source)
    ) - IGNORED_IDENTIFIERS


def sample_accessed_members(metadata: dict) -> set[str]:
    method_path = Path(metadata["graph1_path"])
    class_path = Path(metadata["graph2_path"])
    info_path = Path(metadata["item_path"])
    if not info_path.is_absolute():
        info_path = class_path.parents[2] / info_path
    info_path = info_path / "dataItemInfo.json"
    if not method_path.exists() or not class_path.exists() or not info_path.exists():
        return set()
    method_source = method_path.read_text(encoding="utf-8", errors="ignore")
    class_source = class_path.read_text(encoding="utf-8", errors="ignore")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    identifiers = set(IDENTIFIER.findall(method_source)) - IGNORED_IDENTIFIERS
    members = target_member_names(class_source, info, class_path.stem)
    return identifiers & members


def build_directed_anchor_index(
    metadata: dict[str, dict], vocab: dict[str, int]
) -> tuple[dict[str, tuple[int, ...]], dict]:
    anchors = {}
    positive_samples = 0
    positive_with_anchor = 0
    negative_with_anchor = 0
    anchor_count = 0
    for line, row in metadata.items():
        names = sample_accessed_members(row)
        token_ids = tuple(sorted({int(vocab[name]) for name in names if name in vocab}))
        anchors[line] = token_ids
        if int(row["label"]) == 1:
            positive_samples += 1
            positive_with_anchor += int(bool(token_ids))
            anchor_count += len(token_ids)
        else:
            negative_with_anchor += int(bool(token_ids))
    return anchors, {
        "samples": len(metadata),
        "positive_samples": positive_samples,
        "positive_with_anchor": positive_with_anchor,
        "positive_anchor_coverage": (
            positive_with_anchor / positive_samples if positive_samples else 0.0
        ),
        "mean_positive_anchor_tokens": (
            anchor_count / positive_with_anchor if positive_with_anchor else 0.0
        ),
        "negative_with_anchor": negative_with_anchor,
    }
