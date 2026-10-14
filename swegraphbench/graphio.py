"""Load competition graphs / embeddings and compute audit statistics."""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np


LIB_PREFIX = {  # which top-level dotted prefix is the *library* (vs tests/docs/scripts)
    "fastapi": "fastapi",
    "rich": "rich",
    "requests": "requests",
    "httpx": "httpx",
}


def load_graph(path: Path) -> dict:
    path = Path(path)
    if path.suffix == ".gz":
        import gzip
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


def save_graph(g: dict, path: Path) -> None:
    path = Path(path)
    if path.suffix == ".gz":
        import gzip
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(g, f)
    else:
        path.write_text(json.dumps(g), encoding="utf-8")


def load_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    z = np.load(path)
    keys = list(z.files)
    mat = np.stack([z[k] for k in keys]).astype(np.float32)
    return keys, mat


def graph_key(path: Path) -> tuple[str, str]:
    """fastapi_<sha>.json -> ('fastapi', '<sha>')"""
    stem = Path(path).name
    for suf in (".json.gz", ".json"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    repo, sha = stem.rsplit("_", 1)
    return repo, sha


def audit_graph(g: dict, repo_short: str) -> dict:
    nodes = g["nodes"]
    edges = g["edges"]
    ids = [n["id"] for n in nodes]
    idset = set(ids)
    pairs = {(e["source"], e["target"]) for e in edges}
    recip = sum(1 for s, t in pairs if (t, s) in pairs)
    self_loops = sum(1 for s, t in pairs if s == t)
    deg = Counter()
    for s, t in pairs:
        deg[s] += 1
        deg[t] += 1
    isolated = sum(1 for i in ids if i not in deg)
    lib = LIB_PREFIX.get(repo_short, repo_short)
    lib_nodes = [i for i in ids if i.startswith(lib + ".")]
    test_nodes = [i for i in ids if i.startswith("tests.") or i.startswith("test_") or ".test_" in i]
    lib_edges = sum(1 for s, t in pairs if s.startswith(lib + ".") and t.startswith(lib + "."))
    tl = [len(n["text"]) for n in nodes]
    edge_types = Counter(e.get("type") for e in edges)
    has_file_attr = any(k in nodes[0] for k in ("file", "path", "filepath", "lineno"))
    return {
        "n_nodes": len(ids),
        "n_unique_ids": len(idset),
        "n_edges": len(edges),
        "n_unique_pairs": len(pairs),
        "reciprocal_pair_frac": recip / len(pairs) if pairs else 0.0,
        "self_loop_pairs": self_loops,
        "isolated_frac": isolated / len(ids) if ids else 0.0,
        "lib_node_frac": len(lib_nodes) / len(ids) if ids else 0.0,
        "test_node_frac": len(test_nodes) / len(ids) if ids else 0.0,
        "n_lib_nodes": len(lib_nodes),
        "lib_lib_edges": lib_edges,
        "edge_types": dict(edge_types),
        "node_attrs": sorted(nodes[0].keys()) if nodes else [],
        "has_location_attr": has_file_attr,
        "text_len_median": statistics.median(tl) if tl else 0,
        "max_degree_node": deg.most_common(1)[0] if deg else None,
    }
