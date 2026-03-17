from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .util import tokenize


def _name_tokens(name: str) -> list[str]:
    # Split snake_case/kebab-case into meaningful tokens.
    return [t for t in tokenize(name.replace("-", " ").replace("_", " ")) if t]


def _query_has_path_hint(q: str) -> bool:
    return any(x in q for x in ("./", "../", "/", "\\"))


def _query_file_extension_tokens(q: str) -> set[str]:
    # Very small heuristic: capture .ext occurrences as tokens ("md", "py", ...)
    out: set[str] = set()
    for raw in q.replace("\\", "/").split():
        if "." in raw:
            tail = raw.split(".")[-1].strip(" '\"),.;:")
            if 1 <= len(tail) <= 8 and tail.isalnum():
                out.add(tail.lower())
    return out


def _tool_kind_hints(name_tokens: set[str]) -> dict[str, bool]:
    return {
        "is_media": bool(name_tokens & {"media", "image", "audio"}),
        "is_text": bool(name_tokens & {"text"}),
        "is_allowed_dirs": bool(name_tokens & {"allowed", "directories", "directory"}) and bool(
            name_tokens & {"allowed"}
        ),
    }


def _infer_tool_operation_from_name(name: str) -> str:
    toks = _name_tokens(name)
    if not toks:
        return "unknown"
    # common MCP-ish tool verbs
    verbs = {
        "read",
        "get",
        "fetch",
        "list",
        "search",
        "find",
        "query",
        "create",
        "add",
        "new",
        "update",
        "edit",
        "patch",
        "set",
        "write",
        "put",
        "delete",
        "remove",
        "drop",
        "move",
        "rename",
        "copy",
        "merge",
        "close",
    }
    if toks[0] in verbs:
        return toks[0]
    for t in toks:
        if t in verbs:
            return t
    return "unknown"


def _tool_is_mutating(op: str) -> bool:
    return op in {
        "create",
        "add",
        "new",
        "update",
        "edit",
        "patch",
        "set",
        "write",
        "put",
        "delete",
        "remove",
        "drop",
        "move",
        "rename",
        "copy",
        "merge",
        "close",
    }


def _infer_query_intent(tokens: set[str]) -> dict[str, Any]:
    """
    Very small heuristic intent extractor from query tokens.
    Returns:
      - desired_ops: set[str]
      - wants_mutation: bool
    """
    desired_ops: set[str] = set()
    create_words = {"create", "add", "new", "make"}
    edit_words = {"update", "edit", "modify", "change", "replace", "patch", "write"}
    delete_words = {"delete", "remove", "drop"}
    move_words = {"rename", "move", "copy"}
    merge_words = {"merge", "close"}
    mut_words = create_words | edit_words | delete_words | move_words | merge_words | {"assign", "label"}
    read_words = {"read", "open", "view", "show", "display", "contents", "file"}
    list_words = {"list", "browse", "ls", "directory", "folder"}
    search_words = {"search", "find", "locate", "grep", "match", "pattern", "glob", "query"}

    # Be specific: infer likely operation family from the words present.
    if tokens & create_words:
        desired_ops |= {"create", "add", "new"}
    if tokens & edit_words:
        desired_ops |= {"edit", "update", "patch", "write"}
    if tokens & delete_words:
        desired_ops |= {"delete", "remove", "drop"}
    if tokens & move_words:
        desired_ops |= {"move", "rename", "copy"}
    if tokens & merge_words:
        desired_ops |= {"merge", "close"}
    if tokens & read_words:
        desired_ops |= {"read", "get", "fetch"}
    if tokens & list_words:
        desired_ops |= {"list"}
    if tokens & search_words:
        desired_ops |= {"search", "find", "query"}

    prefers_selective_edit = bool(tokens & {"replace", "patch"}) or ("old" in tokens and "new" in tokens)
    return {
        "desired_ops": desired_ops,
        "wants_mutation": bool(tokens & mut_words),
        "prefers_selective_edit": prefers_selective_edit,
    }


def _extract_schema_terms(schema: Any, *, depth: int = 0, max_depth: int = 3) -> set[str]:
    if depth > max_depth or not isinstance(schema, dict):
        return set()
    out: set[str] = set()
    props = schema.get("properties")
    if isinstance(props, dict):
        for k, v in props.items():
            if isinstance(k, str):
                out |= set(tokenize(k))
            if isinstance(v, dict):
                desc = v.get("description")
                if isinstance(desc, str):
                    out |= set(tokenize(desc))
                enum = v.get("enum")
                if isinstance(enum, list):
                    out |= set(tokenize(" ".join(str(x) for x in enum[:10])))
                out |= _extract_schema_terms(v, depth=depth + 1, max_depth=max_depth)
    items = schema.get("items")
    if isinstance(items, dict):
        out |= _extract_schema_terms(items, depth=depth + 1, max_depth=max_depth)
    return out


def _extract_tool_terms(payload_json: str) -> dict[str, Any]:
    raw = json.loads(payload_json)
    name = str(raw.get("name", ""))
    desc = str(raw.get("description", ""))
    schema = raw.get("input_schema", {}) or {}
    metadata = raw.get("metadata", {}) or {}
    tags = metadata.get("tags", [])
    tag_text = " ".join(str(t) for t in tags) if isinstance(tags, list) else str(tags)

    op = _infer_tool_operation_from_name(name)
    return {
        "name_tokens": set(_name_tokens(name)),
        "desc_tokens": set(tokenize(desc)),
        "schema_tokens": _extract_schema_terms(schema),
        "tag_tokens": set(tokenize(tag_text)),
        "operation": op,
        "mutating": _tool_is_mutating(op),
    }


def _overlap_ratio(q: set[str], t: set[str]) -> float:
    if not q or not t:
        return 0.0
    return len(q & t) / len(q)


@dataclass(frozen=True)
class RerankWeights:
    # All weights are small adjustments added to the base embedding similarity.
    name_overlap: float = 0.06
    desc_overlap: float = 0.03
    schema_overlap: float = 0.05
    tag_overlap: float = 0.03
    op_match: float = 0.08
    side_effect_match: float = 0.05
    selective_edit_match: float = 0.08
    # Heuristics to break ties in dense real catalogs (filesystem).
    filetype_match: float = 0.06
    meta_tool_penalty: float = -0.08


def rerank_candidates(
    *,
    user_query: str,
    base_candidates: list[dict[str, Any]],
    tool_payloads: dict[str, str],
    weights: RerankWeights,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    base_candidates items must contain:
      - tool_id
      - score (base embedding similarity)
      - server, name, version

    Returns:
      - reranked candidate dicts (with final_score and contributions)
      - trace dict containing original order and per-tool explanations
    """
    q_tokens = set(tokenize(user_query))
    q_raw = user_query
    q_exts = _query_file_extension_tokens(q_raw)
    q_has_path = _query_has_path_hint(q_raw)
    q_intent = _infer_query_intent(q_tokens)
    desired_ops: set[str] = set(q_intent["desired_ops"])
    wants_mutation: bool = bool(q_intent["wants_mutation"])
    prefers_selective_edit: bool = bool(q_intent.get("prefers_selective_edit"))

    explanations: dict[str, Any] = {}
    reranked: list[dict[str, Any]] = []
    for c in base_candidates:
        tool_id = str(c["tool_id"])
        payload_json = tool_payloads.get(tool_id)
        if not payload_json:
            # If payload is missing, keep base score.
            new = dict(c)
            new["final_score"] = float(c["score"])
            new["rerank"] = {"reason": "missing_payload"}
            reranked.append(new)
            continue

        terms = _extract_tool_terms(payload_json)
        name_ov = _overlap_ratio(q_tokens, terms["name_tokens"])
        desc_ov = _overlap_ratio(q_tokens, terms["desc_tokens"])
        schema_ov = _overlap_ratio(q_tokens, terms["schema_tokens"])
        tag_ov = _overlap_ratio(q_tokens, terms["tag_tokens"])

        op = str(terms["operation"])
        op_match = 1.0 if (desired_ops and (op in desired_ops)) else 0.0

        mut = bool(terms["mutating"])
        side_match = 0.0
        if desired_ops:
            # If user language implies mutation, prefer mutating tools; otherwise prefer read-only.
            if wants_mutation and mut:
                side_match = 1.0
            elif (not wants_mutation) and (not mut):
                side_match = 1.0

        # Filetype/tool-type hints (helps read_text_file vs read_media_file).
        kind = _tool_kind_hints(set(terms["name_tokens"]))
        filetype_match = 0.0
        if q_exts:
            # If query mentions a typical text extension, prefer text/non-media readers.
            text_exts = {"md", "txt", "py", "toml", "json", "yaml", "yml", "js", "ts", "tsx", "jsx", "css", "html", "xml"}
            media_exts = {"png", "jpg", "jpeg", "gif", "webp", "mp3", "wav", "m4a", "flac", "mp4", "mov", "webm"}
            if q_exts & text_exts:
                if kind["is_media"]:
                    filetype_match = -1.0
                else:
                    filetype_match = 1.0 if (kind["is_text"] or ("read" in terms["name_tokens"])) else 0.5
            elif q_exts & media_exts:
                filetype_match = 1.0 if kind["is_media"] else -0.5

        # Penalize "meta" tools like list_allowed_directories when user asked about a specific path.
        meta_penalty = 0.0
        if q_has_path and kind["is_allowed_dirs"]:
            meta_penalty = 1.0

        # Prefer selective edit tools for "replace/patch" style queries.
        selective_edit = 0.0
        if prefers_selective_edit and wants_mutation:
            nt = set(terms["name_tokens"])
            if "edit" in nt or "patch" in nt:
                selective_edit = 1.0
            elif "write" in nt:
                selective_edit = -1.0

        contrib = {
            "base_embedding": float(c["score"]),
            "name_overlap": weights.name_overlap * name_ov,
            "desc_overlap": weights.desc_overlap * desc_ov,
            "schema_overlap": weights.schema_overlap * schema_ov,
            "tag_overlap": weights.tag_overlap * tag_ov,
            "op_match": weights.op_match * op_match,
            "side_effect_match": weights.side_effect_match * side_match,
            "selective_edit_match": weights.selective_edit_match * selective_edit,
            "filetype_match": weights.filetype_match * filetype_match,
            "meta_tool_penalty": weights.meta_tool_penalty * meta_penalty,
        }
        final = sum(contrib.values())

        new = dict(c)
        new["final_score"] = float(final)
        new["rerank"] = {
            "operation": op,
            "mutating": mut,
            "intent_desired_ops": sorted(desired_ops),
            "intent_wants_mutation": wants_mutation,
            "raw_features": {
                "name_overlap_ratio": name_ov,
                "desc_overlap_ratio": desc_ov,
                "schema_overlap_ratio": schema_ov,
                "tag_overlap_ratio": tag_ov,
                "op_match": op_match,
                "side_effect_match": side_match,
                "selective_edit_match": selective_edit,
                "filetype_match": filetype_match,
                "meta_tool_penalty": meta_penalty,
            },
            "contrib": contrib,
        }
        reranked.append(new)
        explanations[tool_id] = new["rerank"]

    reranked.sort(key=lambda x: float(x["final_score"]), reverse=True)
    trace = {
        "original": [{"tool_id": c["tool_id"], "score": c["score"]} for c in base_candidates],
        "reranked": [{"tool_id": c["tool_id"], "final_score": c["final_score"]} for c in reranked],
        "per_tool": explanations,
        "weights": weights.__dict__,
    }
    return reranked, trace

