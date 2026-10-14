"""Audit every released graph + embedding file and measure gold-symbol coverage.

Outputs:
  results/graph_audit.jsonl      one row per graph file
  results/coverage.jsonl         one row per task (graph availability + gold coverage)
  results/audit_summary.json     aggregate numbers used in the paper
"""
import json, sys, collections, statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from swegraphbench.graphio import load_graph, audit_graph, graph_key  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "results"
REPO_SHORT = {"fastapi/fastapi": "fastapi", "Textualize/rich": "rich", "psf/requests": "requests", "encode/httpx": "httpx"}

tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in (DATA / "tasks.jsonl").open(encoding="utf-8") if l.strip()}
gold = {json.loads(l)["instance_id"]: json.loads(l) for l in (OUT / "gold.jsonl").open(encoding="utf-8")}

graph_files = {graph_key(p): p for p in (DATA / "graphs").glob("*.json")}
emb_files = {graph_key(p): p for p in (DATA / "embeddings").glob("*.npz")}
print("graph files:", len(graph_files), "embedding files:", len(emb_files))

# ---------- 1. per-graph audit ----------
audits = {}
with (OUT / "graph_audit.jsonl").open("w", encoding="utf-8") as f:
    for (repo, sha), p in sorted(graph_files.items()):
        g = load_graph(p)
        a = audit_graph(g, repo)
        a.update({"repo": repo, "sha": sha, "file": p.name})
        # embedding/node key agreement
        if (repo, sha) in emb_files:
            z = np.load(emb_files[(repo, sha)])
            ek = set(z.files)
            nk = {n["id"] for n in g["nodes"]}
            a["emb_keys"] = len(ek)
            a["emb_dim"] = int(z[z.files[0]].shape[0])
            a["emb_minus_nodes"] = len(ek - nk)
            a["nodes_minus_emb"] = len(nk - ek)
        audits[(repo, sha)] = a
        f.write(json.dumps(a) + "\n")

def agg(key):
    vals = [a[key] for a in audits.values() if key in a]
    return {"min": min(vals), "median": statistics.median(vals), "max": max(vals)} if vals else None

# ---------- 2. per-task coverage ----------
def candidate_names(sym: str) -> list[str]:
    """The competition graph drops the `docs_src.` prefix; try both forms."""
    out = [sym]
    if sym.startswith("docs_src."):
        out.append(sym[len("docs_src."):])
    return out

cov_rows = []
c = collections.Counter()
per_repo = collections.defaultdict(collections.Counter)
for iid, t in tasks.items():
    repo = REPO_SHORT[t["repo"]]
    key = (repo, t["base_commit"])
    gsyms = set(gold[iid]["symbols"])
    row = {"instance_id": iid, "repo": repo, "has_graph": key in graph_files, "has_emb": key in emb_files,
           "n_gold": len(gsyms), "n_gold_in_graph": 0, "n_gold_in_emb": 0, "gold_in_graph_with_edges": 0,
           "missing_gold": []}
    c["tasks"] += 1; per_repo[repo]["tasks"] += 1
    if row["has_graph"]: c["tasks_with_graph"] += 1; per_repo[repo]["tasks_with_graph"] += 1
    if row["has_emb"]: c["tasks_with_emb"] += 1; per_repo[repo]["tasks_with_emb"] += 1
    if gsyms:
        c["tasks_with_gold"] += 1; per_repo[repo]["tasks_with_gold"] += 1
    if key in graph_files and gsyms:
        g = load_graph(graph_files[key])
        ids = {n["id"] for n in g["nodes"]}
        deg = collections.Counter()
        for e in g["edges"]:
            deg[e["source"]] += 1; deg[e["target"]] += 1
        hit = set(); hit_edges = set(); missing = []
        for s in gsyms:
            m = next((cn for cn in candidate_names(s) if cn in ids), None)
            if m:
                hit.add(s)
                if deg[m] > 0: hit_edges.add(s)
            else:
                missing.append(s)
        row.update({"n_gold_in_graph": len(hit), "gold_in_graph_with_edges": len(hit_edges), "missing_gold": missing[:20]})
        c["tasks_with_graph_and_gold"] += 1; per_repo[repo]["tasks_with_graph_and_gold"] += 1
        if len(hit) == len(gsyms): c["tasks_all_gold_covered"] += 1; per_repo[repo]["tasks_all_gold_covered"] += 1
        if hit: c["tasks_any_gold_covered"] += 1; per_repo[repo]["tasks_any_gold_covered"] += 1
        c["gold_syms"] += len(gsyms); c["gold_syms_in_graph"] += len(hit); c["gold_syms_with_edges"] += len(hit_edges)
        per_repo[repo]["gold_syms"] += len(gsyms); per_repo[repo]["gold_syms_in_graph"] += len(hit); per_repo[repo]["gold_syms_with_edges"] += len(hit_edges)
    if key in emb_files and gsyms:
        ek = set(np.load(emb_files[key]).files)
        row["n_gold_in_emb"] = sum(1 for s in gsyms if any(cn in ek for cn in candidate_names(s)))
        c["gold_syms_emb_checked"] += len(gsyms); c["gold_syms_in_emb"] += row["n_gold_in_emb"]
        per_repo[repo]["gold_syms_emb_checked"] += len(gsyms); per_repo[repo]["gold_syms_in_emb"] += row["n_gold_in_emb"]
    cov_rows.append(row)

with (OUT / "coverage.jsonl").open("w", encoding="utf-8") as f:
    for r in cov_rows:
        f.write(json.dumps(r) + "\n")

summary = {
    "n_graph_files": len(graph_files), "n_emb_files": len(emb_files),
    "graph_files_by_repo": dict(collections.Counter(r for r, _ in graph_files)),
    "emb_files_by_repo": dict(collections.Counter(r for r, _ in emb_files)),
    "graph_stats": {k: agg(k) for k in ["n_nodes", "n_edges", "reciprocal_pair_frac", "isolated_frac", "lib_node_frac", "test_node_frac", "self_loop_pairs", "emb_minus_nodes", "nodes_minus_emb", "emb_dim"]},
    "edge_types_union": sorted({t for a in audits.values() for t in a["edge_types"]}),
    "node_attrs": sorted({k for a in audits.values() for k in a["node_attrs"]}),
    "any_location_attr": any(a["has_location_attr"] for a in audits.values()),
    "coverage": dict(c),
    "coverage_per_repo": {k: dict(v) for k, v in per_repo.items()},
}
summary["coverage"]["gold_symbol_coverage_frac"] = c["gold_syms_in_graph"] / c["gold_syms"] if c["gold_syms"] else None
summary["coverage"]["gold_symbol_with_edges_frac"] = c["gold_syms_with_edges"] / c["gold_syms"] if c["gold_syms"] else None
summary["coverage"]["gold_symbol_emb_coverage_frac"] = c["gold_syms_in_emb"] / c["gold_syms_emb_checked"] if c["gold_syms_emb_checked"] else None
(OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
# show some missing examples
miss = [(r["instance_id"], r["missing_gold"][:3]) for r in cov_rows if r["missing_gold"]]
print("missing examples:", miss[:12])
