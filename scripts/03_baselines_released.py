"""Run LLM-free localization baselines on the RELEASED graphs/embeddings.

Evaluated on tasks that have a graph and >=1 gold symbol (65 tasks). Gold is matched
against the released node ids (docs_src. prefix dropped), so a gold symbol that is not
a node is counted as a miss (this is the 'coverage ceiling' row).

Outputs results/baselines_released.json and results/baselines_released_per_task.jsonl
"""
import json, sys, time, math, statistics, random
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from swegraphbench.graphio import load_graph, load_embeddings, graph_key  # noqa
from swegraphbench.retrieval import GraphIndex, recall_at, hit_at, mrr  # noqa

DATA, OUT = ROOT / "data", ROOT / "results"
REPO_SHORT = {"fastapi/fastapi": "fastapi", "Textualize/rich": "rich", "psf/requests": "requests", "encode/httpx": "httpx"}
KS = (1, 5, 10, 20, 50)
METHODS = ["bm25_name", "bm25_text", "symbol_seed", "emb_expand", "ppr", "hybrid"]

tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in (DATA / "tasks.jsonl").open(encoding="utf-8") if l.strip()}
gold = {json.loads(l)["instance_id"]: json.loads(l) for l in (OUT / "gold.jsonl").open(encoding="utf-8")}
graph_files = {graph_key(p): p for p in (DATA / "graphs").glob("*.json")}
emb_files = {graph_key(p): p for p in (DATA / "embeddings").glob("*.npz")}


def norm_gold(syms, ids):
    out = set()
    for s in syms:
        for cand in (s, s[len("docs_src."):] if s.startswith("docs_src.") else None):
            if cand and cand in ids:
                out.add(cand); break
    return out


rows = []
t0 = time.time()
for iid, t in tasks.items():
    key = (REPO_SHORT[t["repo"]], t["base_commit"])
    gs = gold[iid]["symbols"]
    if key not in graph_files or not gs:
        continue
    g = load_graph(graph_files[key])
    ek, em = load_embeddings(emb_files[key])
    idx = GraphIndex(g, ek, em)
    ids = set(idx.nodes)
    gold_all = set(gs)
    gold_in = norm_gold(gs, ids)
    row = {"instance_id": iid, "repo": key[0], "n_gold": len(gold_all), "n_gold_in_graph": len(gold_in), "methods": {}}
    for m in METHODS:
        ranked = getattr(idx, m)(t["problem_statement"], k=max(KS))
        # score against ALL gold (strict) and against in-graph gold (ceiling-adjusted)
        row["methods"][m] = {
            "strict": {f"recall@{k}": len(set(ranked[:k]) & gold_in) / len(gold_all) for k in KS}
                      | {f"hit@{k}": hit_at(ranked, gold_in, k) for k in KS} | {"mrr": mrr(ranked, gold_in)},
            "n_ranked": len(ranked),
        }
        if gold_in:
            row["methods"][m]["in_graph"] = {f"recall@{k}": recall_at(ranked, gold_in, k) for k in KS} | {"mrr": mrr(ranked, gold_in)}
    rows.append(row)
    print(f"{iid:16s} gold={len(gold_all):3d} in_graph={len(gold_in):3d} " + " ".join(f"{m}:h5={row['methods'][m]['strict']['hit@5']:.0f}" for m in METHODS), flush=True)

print("elapsed", round(time.time() - t0, 1), "s")
with (OUT / "baselines_released_per_task.jsonl").open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")


def boot_ci(vals, n=2000, seed=0):
    rng = random.Random(seed); vals = list(vals); m = []
    for _ in range(n):
        s = [vals[rng.randrange(len(vals))] for _ in vals]; m.append(sum(s) / len(s))
    m.sort(); return m[int(0.025 * n)], m[int(0.975 * n)]


summary = {"n_tasks": len(rows), "n_tasks_with_in_graph_gold": sum(1 for r in rows if r["n_gold_in_graph"]), "methods": {}}
for m in METHODS:
    s = {}
    for metric in [f"recall@{k}" for k in KS] + [f"hit@{k}" for k in KS] + ["mrr"]:
        vals = [r["methods"][m]["strict"][metric] for r in rows]
        lo, hi = boot_ci(vals)
        s[metric] = {"mean": sum(vals) / len(vals), "ci95": [lo, hi]}
    vals_in = [r["methods"][m]["in_graph"]["recall@5"] for r in rows if "in_graph" in r["methods"][m]]
    s["in_graph_recall@5"] = sum(vals_in) / len(vals_in)
    vals_in = [r["methods"][m]["in_graph"]["mrr"] for r in rows if "in_graph" in r["methods"][m]]
    s["in_graph_mrr"] = sum(vals_in) / len(vals_in)
    summary["methods"][m] = s
# coverage ceiling: best possible strict recall
summary["ceiling_strict_recall"] = sum(r["n_gold_in_graph"] / r["n_gold"] for r in rows) / len(rows)
summary["ceiling_strict_hit"] = sum(1 for r in rows if r["n_gold_in_graph"]) / len(rows)
(OUT / "baselines_released.json").write_text(json.dumps(summary, indent=2))
print(f"\n{'method':12s} " + " ".join(f"{k:>8s}" for k in ["R@1", "R@5", "R@10", "R@20", "H@1", "H@5", "H@10", "MRR"]))
for m in METHODS:
    s = summary["methods"][m]
    print(f"{m:12s} " + " ".join(f"{s[k]['mean']:8.3f}" for k in ["recall@1", "recall@5", "recall@10", "recall@20", "hit@1", "hit@5", "hit@10", "mrr"]))
print(f"ceiling      strict recall {summary['ceiling_strict_recall']:.3f}  strict hit {summary['ceiling_strict_hit']:.3f}")
