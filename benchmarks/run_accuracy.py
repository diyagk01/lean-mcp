from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from mcp_tool_router import (
    Client,
    HashingEmbeddingProvider,
    RouterConfig,
    SentenceTransformersEmbeddingProvider,
)


@dataclass(frozen=True)
class BenchCase:
    size: int
    servers: int
    queries: int


def _read_labels(path: Path) -> list[dict[str, Any]]:
    """
    Supported formats:
    - JSON: a list of objects
    - JSONL: one JSON object per line

    Each example supports:
      - query: str (required)
      - expected_tool_id: str (optional)  e.g. "github:search_issues:1"
      - expected_tool_name: str (optional; less specific than tool_id)
      - server: str (optional; helps disambiguate tool_name and can be used as server_filter)
      - expected_abstain: bool (optional)
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        raw = json.loads(text)
        if not isinstance(raw, list):
            raise ValueError("JSON label file must be a list of objects")
        return [dict(x) for x in raw]

    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(dict(json.loads(line)))
    return out


def _count_tools_in_descriptor_root(root: Path) -> int:
    n = 0
    for server_dir in root.iterdir():
        if not server_dir.is_dir():
            continue
        tool_dir = server_dir / "tools"
        if tool_dir.exists():
            n += len(list(tool_dir.glob("*.json")))
    return n


def _mk_descriptor_root(base: Path, *, size: int, servers: int, seed: int) -> Path:
    """
    Creates a synthetic descriptor tree:
      root/srv_i/tools/*.json

    Tools are designed to be distinguishable by:
    - natural-language topic keywords (better proxy for real MCP usage)
    - schema property names (indexed by this SDK)
    """
    rng = random.Random(seed)
    root = base / f"desc_{size}_{servers}"
    for s in range(servers):
        tool_dir = root / f"srv{s}" / "tools"
        tool_dir.mkdir(parents=True, exist_ok=True)

    # A fixed vocabulary to simulate real tool domains/terms.
    vocab = [
        "issues",
        "pull request",
        "repository",
        "channel",
        "message",
        "email",
        "calendar",
        "invoice",
        "payment",
        "weather",
        "flight",
        "hotel",
        "kubernetes",
        "docker",
        "logs",
        "metrics",
        "alerts",
        "database",
        "migration",
        "backup",
        "search",
        "news",
        "summarize",
        "translate",
        "token refresh",
        "authentication",
        "users",
        "billing",
        "analytics",
        "export",
    ]

    # Distribute tools across servers.
    for i in range(size):
        server = f"srv{rng.randrange(servers)}"
        topic = vocab[i % len(vocab)]
        intent = f"{topic} task {i}"
        tool = {
            "name": f"tool_{i}",
            "description": f"Handle {topic} related requests. Specializes in {intent}.",
            # Use a mix of schema field variants to mirror the wild.
            "arguments": {
                "type": "object",
                "properties": {
                    f"param_{i}": {"type": "string", "description": f"Key input for {topic}."},
                    "limit": {"type": "integer", "enum": [10, 25, 50]},
                },
                "required": [f"param_{i}"],
            },
            "version": "1",
            "metadata": {"tags": [server, topic]},
        }
        p = root / server / "tools" / f"tool_{i}.json"
        p.write_text(json.dumps(tool, indent=2, sort_keys=True), encoding="utf-8")
    return root


def _generate_queries(*, size: int, n: int, seed: int) -> list[tuple[str, str]]:
    """
    Returns (query, expected_tool_name).
    Query includes topic words and required param name.
    """
    rng = random.Random(seed + 999)
    vocab = [
        "issues",
        "pull request",
        "repository",
        "channel",
        "message",
        "email",
        "calendar",
        "invoice",
        "payment",
        "weather",
        "flight",
        "hotel",
        "kubernetes",
        "docker",
        "logs",
        "metrics",
        "alerts",
        "database",
        "migration",
        "backup",
        "search",
        "news",
        "summarize",
        "translate",
        "token refresh",
        "authentication",
        "users",
        "billing",
        "analytics",
        "export",
    ]
    qs: list[tuple[str, str]] = []
    for _ in range(n):
        i = rng.randrange(size)
        topic = vocab[i % len(vocab)]
        q = (
            f"I need help with {topic}. Please run the tool for this request. "
            f"The {f'param_{i}'} is abc123."
        )
        qs.append((q, f"tool_{i}"))
    return qs


def _generate_ambiguous_queries(*, n: int) -> list[str]:
    return [
        "Help me do something.",
        "I need assistance with a task.",
        "Please run the command.",
        "Can you take care of this for me?",
    ][:n]


def _pick_embedder(name: str, model_name: str | None):
    if name == "hash":
        return HashingEmbeddingProvider()
    if name == "st":
        if not model_name:
            raise SystemExit("--model-name is required when --embedder=st")
        return SentenceTransformersEmbeddingProvider(model_name=model_name, device="cpu")
    raise SystemExit(f"Unknown embedder: {name}")


def run_case(case: BenchCase, *, embedder: str, model_name: str | None) -> dict:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        desc_root = _mk_descriptor_root(td_path, size=case.size, servers=case.servers, seed=123)
        db_path = str(td_path / "bench.sqlite3")

        cfg = RouterConfig(rerank_enabled=not bool(os.environ.get("MCP_TOOL_ROUTER_NO_RERANK")))
        client = Client(db_path=db_path, embedder=_pick_embedder(embedder, model_name), config=cfg)
        client.sync_descriptors(root=str(desc_root), delete_stale=True)

        labeled = _generate_queries(size=case.size, n=case.queries, seed=123)
        correct = 0
        abstained = 0
        for q, expected in labeled:
            d = client.route(q, top_k=25, min_confidence=0.0, min_margin=0.0)
            if d.best_tool is None:
                abstained += 1
                continue
            if d.best_tool.name == expected:
                correct += 1

        # Ambiguity: expect abstention when thresholds are strict.
        ambiguous = _generate_ambiguous_queries(n=4)
        amb_abstained = 0
        for q in ambiguous:
            d = client.route(q, top_k=25, min_confidence=0.30, min_margin=0.05)
            if d.best_tool is None:
                amb_abstained += 1

        acc = correct / max(1, len(labeled))
        amb_rate = amb_abstained / max(1, len(ambiguous))

        # Pass criteria: high top-1 accuracy on unambiguous set + strong abstention on ambiguous set
        passed = acc >= 0.90 and amb_rate >= 0.75

        return {
            "size": case.size,
            "servers": case.servers,
            "queries": case.queries,
            "embedder": embedder,
            "model_name": model_name,
            "top1_accuracy": round(acc, 4),
            "unambiguous_abstain_rate": round(abstained / max(1, len(labeled)), 4),
            "ambiguous_abstain_rate": round(amb_rate, 4),
            "passed": passed,
        }


def _tool_key_from_candidate(cand) -> tuple[Optional[str], str]:
    return (cand.server, cand.name)


def run_real_catalog(
    *,
    descriptor_root: Path,
    labels_path: Path,
    embedder: str,
    model_name: str | None,
    top_k: int,
    min_confidence: float,
    min_margin: float,
    debug: bool = False,
    debug_top_k: Optional[int] = None,
    server: Optional[str] = None,
) -> dict[str, Any]:
    labels = _read_labels(labels_path)
    if not labels:
        raise SystemExit("No labeled examples found in label file.")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        db_path = str(td_path / "bench.sqlite3")
        cfg = RouterConfig(rerank_enabled=not bool(os.environ.get("MCP_TOOL_ROUTER_NO_RERANK")))
        client = Client(db_path=db_path, embedder=_pick_embedder(embedder, model_name), config=cfg)
        client.sync_descriptors(root=str(descriptor_root), delete_stale=True)

        total_tools = _count_tools_in_descriptor_root(descriptor_root)

        total = 0
        correct_top1 = 0
        hit_topk = 0
        abstained = 0

        # Abstention precision/recall only computed when expected_abstain is labeled.
        abstain_labeled_total = 0
        abstain_expected_true = 0
        abstain_pred_true = 0
        abstain_true_pos = 0

        per_server: dict[str, dict[str, int]] = {}

        for ex in labels:
            q = ex.get("query")
            if not isinstance(q, str) or not q.strip():
                continue

            expected_abstain = ex.get("expected_abstain", None)
            if expected_abstain is not None:
                abstain_labeled_total += 1
                if bool(expected_abstain):
                    abstain_expected_true += 1

            server_filter = ex.get("server")
            if server_filter is not None and not isinstance(server_filter, str):
                server_filter = None
            if server:
                # benchmark-per-server override
                server_filter = server

            d = client.route(
                q,
                top_k=top_k,
                server_filter=server_filter,
                min_confidence=min_confidence,
                min_margin=min_margin,
            )
            total += 1
            if d.best_tool is None:
                abstained += 1

            if expected_abstain is not None:
                pred_abstain = d.best_tool is None
                if pred_abstain:
                    abstain_pred_true += 1
                if bool(expected_abstain) and pred_abstain:
                    abstain_true_pos += 1

            # Abstain-labeled example: correctness is abstaining.
            if bool(ex.get("expected_abstain", False)):
                srv = str(ex.get("server") or "unknown")
                per_server.setdefault(
                    srv, {"total": 0, "top1": 0, "topk": 0, "abstained": 0}
                )
                per_server[srv]["total"] += 1
                if d.best_tool is None:
                    correct_top1 += 1
                    hit_topk += 1
                    per_server[srv]["top1"] += 1
                    per_server[srv]["topk"] += 1
                    per_server[srv]["abstained"] += 1
                else:
                    if debug:
                        print("")
                        print("FAIL (expected abstain, predicted tool)")
                        print(f"query: {q}")
                        print("expected: abstain=true")
                        print(f"predicted: {d.best_tool.server}:{d.best_tool.name}:{d.best_tool.version} score={d.best_tool.score:.4f}")
                        print(f"router_abstained: {d.abstained} confidence={d.confidence:.4f} margin={d.margin:.4f}")
                        kshow = debug_top_k or top_k
                        print("top-k:")
                        for cand in d.candidates[:kshow]:
                            print(f"  - {cand.server}:{cand.name}:{cand.version} score={cand.score:.4f}")
                continue

            expected_tool_id = ex.get("expected_tool_id")
            expected_tool_name = ex.get("expected_tool_name")

            expected_key: tuple[Optional[str], str] | None = None
            expected_str = None
            if isinstance(expected_tool_id, str) and expected_tool_id:
                parts = expected_tool_id.split(":")
                if len(parts) >= 2:
                    expected_key = (parts[0], parts[1])
                    expected_str = expected_tool_id
            elif isinstance(expected_tool_name, str) and expected_tool_name:
                expected_key = (server_filter, expected_tool_name)
                expected_str = f"{server_filter or '*'}:{expected_tool_name}:*"

            srv_bucket = (
                str(server_filter)
                if server_filter
                else (str(expected_key[0]) if expected_key and expected_key[0] else "unknown")
            )
            per_server.setdefault(
                srv_bucket, {"total": 0, "top1": 0, "topk": 0, "abstained": 0}
            )
            per_server[srv_bucket]["total"] += 1
            per_server[srv_bucket]["abstained"] += 1 if d.best_tool is None else 0

            if expected_key is None:
                continue

            top1_ok = d.best_tool is not None and _tool_key_from_candidate(d.best_tool) == expected_key
            if top1_ok:
                correct_top1 += 1
                per_server[srv_bucket]["top1"] += 1
            elif debug:
                # Determine failure reason: abstained vs wrong top1; and whether expected was in candidates.
                expected_in_topk = any(
                    _tool_key_from_candidate(c) == expected_key for c in d.candidates[: max(1, top_k)]
                )
                reason = "abstained" if d.best_tool is None else "wrong_top1"
                if reason == "wrong_top1" and not expected_in_topk:
                    reason = "retrieval_miss"
                print("")
                print(f"FAIL ({reason})")
                print(f"query: {q}")
                print(f"expected: {expected_str or expected_key}")
                if d.best_tool is None:
                    print("predicted: abstain")
                else:
                    print(f"predicted: {d.best_tool.server}:{d.best_tool.name}:{d.best_tool.version} score={d.best_tool.score:.4f}")
                print(f"router_abstained: {d.abstained} confidence={d.confidence:.4f} margin={d.margin:.4f}")
                print(f"expected_in_topk: {expected_in_topk}")
                kshow = debug_top_k or top_k
                print("top-k:")
                for cand in d.candidates[:kshow]:
                    marker = "  "
                    if _tool_key_from_candidate(cand) == expected_key:
                        marker = "✓ "
                    print(f"{marker}- {cand.server}:{cand.name}:{cand.version} score={cand.score:.4f}")

            for cand in d.candidates[: max(1, top_k)]:
                if _tool_key_from_candidate(cand) == expected_key:
                    hit_topk += 1
                    per_server[srv_bucket]["topk"] += 1
                    break

        top1 = correct_top1 / max(1, total)
        topk_hit = hit_topk / max(1, total)
        abstain_rate = abstained / max(1, total)

        abstain_precision = None
        abstain_recall = None
        if abstain_labeled_total > 0:
            abstain_precision = abstain_true_pos / max(1, abstain_pred_true)
            abstain_recall = abstain_true_pos / max(1, abstain_expected_true)

        return {
            "mode": "real",
            "descriptor_root": str(descriptor_root),
            "labels_path": str(labels_path),
            "embedder": embedder,
            "model_name": model_name,
            "total_tools_indexed": total_tools,
            "total_queries": total,
            "overall_top1_accuracy": round(top1, 4),
            "overall_topk_hit_rate": round(topk_hit, 4),
            "abstention_rate": round(abstain_rate, 4),
            "abstain_precision": None if abstain_precision is None else round(abstain_precision, 4),
            "abstain_recall": None if abstain_recall is None else round(abstain_recall, 4),
            "per_server": per_server,
            "per_size_summary": [
                {
                    "tool_count": total_tools,
                    "queries": total,
                    "top1_accuracy": round(top1, 4),
                    "topk_hit_rate": round(topk_hit, 4),
                }
            ],
        }


def _print_real_report(result: dict[str, Any]) -> None:
    print("")
    print("=== MCP Tool Router Benchmark (real descriptor catalog) ===")
    print(f"Descriptor root: {result['descriptor_root']}")
    print(f"Labels:          {result['labels_path']}")
    emb = result["embedder"] + (f" ({result['model_name']})" if result.get("model_name") else "")
    print(f"Embedder:        {emb}")
    print("")
    print(f"Total tools indexed: {result['total_tools_indexed']}")
    print(f"Total queries:       {result['total_queries']}")
    print("")
    print(f"Overall top-1 accuracy: {result['overall_top1_accuracy']}")
    print(f"Overall top-k hit rate: {result['overall_topk_hit_rate']}")
    print(f"Abstention rate:        {result['abstention_rate']}")
    if result.get("abstain_precision") is not None:
        print(f"Abstain precision:      {result['abstain_precision']}")
        print(f"Abstain recall:         {result['abstain_recall']}")
    else:
        print("Abstain precision/recall: N/A (no expected_abstain labels)")
    print("")
    print("Breakdown by server:")
    for server in sorted(result["per_server"].keys()):
        row = result["per_server"][server]
        tot = max(1, int(row["total"]))
        top1 = float(row["top1"]) / tot
        topk = float(row["topk"]) / tot
        abst = float(row["abstained"]) / tot
        print(f"- {server}: total={row['total']} top1={top1:.3f} topk={topk:.3f} abstain={abst:.3f}")
    print("")


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", choices=["hash", "st"], default="hash")
    ap.add_argument("--model-name", default=os.environ.get("MCP_TOOL_ROUTER_MODEL"))
    ap.add_argument("--sizes", default="10,100,500,1000")
    ap.add_argument("--servers", type=int, default=4)
    ap.add_argument("--queries", type=int, default=80)
    ap.add_argument("--descriptor-root", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--min-confidence", type=float, default=0.22)
    ap.add_argument("--min-margin", type=float, default=0.04)
    ap.add_argument(
        "--sweep-thresholds",
        action="store_true",
        help="Run a grid over min_confidence/min_margin (real mode only).",
    )
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug-top-k", type=int, default=None)
    ap.add_argument("--server", default=None, help="Benchmark only one server (real mode).")
    ap.add_argument("--per-server", action="store_true", help="Run real-mode benchmark separately per server.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.descriptor_root:
        if not args.labels:
            raise SystemExit("--labels is required when --descriptor-root is provided")
        root = Path(args.descriptor_root)

        if args.sweep_thresholds:
            conf_values = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22]
            margin_values = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]
            results: list[dict[str, Any]] = []

            for c in conf_values:
                for m in margin_values:
                    r = run_real_catalog(
                        descriptor_root=root,
                        labels_path=Path(args.labels),
                        embedder=args.embedder,
                        model_name=args.model_name,
                        top_k=int(args.top_k),
                        min_confidence=float(c),
                        min_margin=float(m),
                        debug=False,
                        debug_top_k=None,
                        server=args.server or "filesystem",
                    )
                    r["min_confidence"] = c
                    r["min_margin"] = m
                    results.append(r)

            def sort_key(r: dict[str, Any]):
                return (
                    -float(r["overall_top1_accuracy"]),
                    -float(r.get("abstain_recall") or 0.0),
                    float(r["abstention_rate"]),
                )

            results.sort(key=sort_key)

            print("")
            print("min_conf  min_margin  top1     topk     abstain  abstain_prec  abstain_rec")
            print("--------  ----------  -------  -------  -------  ------------  -----------")
            for r in results:
                print(
                    f"{r['min_confidence']:8.2f}  {r['min_margin']:10.2f}  "
                    f"{r['overall_top1_accuracy']:7.3f}  {r['overall_topk_hit_rate']:7.3f}  "
                    f"{r['abstention_rate']:7.3f}  "
                    f"{(r['abstain_precision'] if r['abstain_precision'] is not None else 0.0):12.3f}  "
                    f"{(r['abstain_recall'] if r['abstain_recall'] is not None else 0.0):11.3f}"
                )

            best_top1 = results[0]
            conservative = max(
                results,
                key=lambda r: (
                    float(r.get("abstain_recall") or 0.0),
                    -float(r["abstention_rate"]),
                    float(r["overall_top1_accuracy"]),
                ),
            )
            balanced_candidates = [r for r in results if (r.get("abstain_recall") or 0.0) >= 0.5]
            balanced = (
                max(
                    balanced_candidates,
                    key=lambda r: (
                        float(r["overall_top1_accuracy"]),
                        -float(r["abstention_rate"]),
                    ),
                )
                if balanced_candidates
                else best_top1
            )

            print("")
            print("Recommended threshold pairs:")
            print(
                f"- Aggressive router (max accuracy): "
                f"min_confidence={best_top1['min_confidence']:.2f}, "
                f"min_margin={best_top1['min_margin']:.2f} "
                f"(top1={best_top1['overall_top1_accuracy']:.3f}, "
                f"abstain_recall={best_top1.get('abstain_recall') or 0.0:.3f}, "
                f"abstention_rate={best_top1['abstention_rate']:.3f})"
            )
            print(
                f"- Conservative router (prefer abstaining): "
                f"min_confidence={conservative['min_confidence']:.2f}, "
                f"min_margin={conservative['min_margin']:.2f} "
                f"(top1={conservative['overall_top1_accuracy']:.3f}, "
                f"abstain_recall={conservative.get('abstain_recall') or 0.0:.3f}, "
                f"abstention_rate={conservative['abstention_rate']:.3f})"
            )
            print(
                f"- Balanced router: "
                f"min_confidence={balanced['min_confidence']:.2f}, "
                f"min_margin={balanced['min_margin']:.2f} "
                f"(top1={balanced['overall_top1_accuracy']:.3f}, "
                f"abstain_recall={balanced.get('abstain_recall') or 0.0:.3f}, "
                f"abstention_rate={balanced['abstention_rate']:.3f})"
            )

            print("")
            print(
                "Interpretation:\n"
                "- Higher top-1 means the router more often picks the expected tool when one is labeled.\n"
                "- Higher abstain_recall means it correctly abstains more often when labels say expected_abstain=true.\n"
                "- Higher abstention_rate means it gives up more often overall (more fallbacks/clarifying questions)."
            )
            return 0

        if args.per_server:
            # Determine servers from descriptor tree.
            servers = sorted([p.name for p in root.iterdir() if p.is_dir() and (p / "tools").exists()])
            all_results: dict[str, Any] = {"mode": "real_per_server", "servers": []}
            for srv in servers:
                result = run_real_catalog(
                    descriptor_root=root,
                    labels_path=Path(args.labels),
                    embedder=args.embedder,
                    model_name=args.model_name,
                    top_k=int(args.top_k),
                    min_confidence=float(args.min_confidence),
                    min_margin=float(args.min_margin),
                    debug=bool(args.debug),
                    debug_top_k=args.debug_top_k,
                    server=srv,
                )
                print("")
                print(f"##### SERVER: {srv} #####")
                _print_real_report(result)
                all_results["servers"].append(result)
            print(json.dumps(all_results, indent=2))
            return 0

        result = run_real_catalog(
            descriptor_root=root,
            labels_path=Path(args.labels),
            embedder=args.embedder,
            model_name=args.model_name,
            top_k=int(args.top_k),
            min_confidence=float(args.min_confidence),
            min_margin=float(args.min_margin),
            debug=bool(args.debug),
            debug_top_k=args.debug_top_k,
            server=args.server,
        )
        _print_real_report(result)
        print(json.dumps(result, indent=2))
        return 0

    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    results = []
    for sz in sizes:
        results.append(
            run_case(
                BenchCase(size=sz, servers=args.servers, queries=args.queries),
                embedder=args.embedder,
                model_name=args.model_name,
            )
        )

    print(json.dumps({"results": results}, indent=2))
    # exit non-zero if any failed
    return 0 if all(r["passed"] for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())

