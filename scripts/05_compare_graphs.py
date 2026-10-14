"""Compare localization on RELEASED vs REBUILT graphs, same tasks, same methods.

Conditions (all LLM-free):
  released            released graph + released embeddings (65 tasks that have a graph)
  released-notests    same, test nodes removed from candidate set
  rebuilt             rebuilt graph, undirected PPR, no embeddings (emb_expand degrades to symbol_seed)
  rebuilt-notests     rebuilt graph, tests removed
  rebuilt-notests-dir rebuilt graph, tests removed, DIRECTED PPR (callee direction)
  rebuilt-relemb      rebuilt graph + released embeddings for the nodes that have one

Also evaluates rebuilt-notests on ALL 121 tasks with gold (incl. the 56 rich/requests tasks
that have no released graph) -> what the resource unlocks.

Outputs results/compare_graphs.json (+ per-task jsonl)
"""
import json, sys, random
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from swegraphbench.graphio import load_graph, load_embeddings, graph_key  # noqa
from swegraphbench.retrieval import GraphIndex, hit_at, mrr  # noqa

DATA, OUT = ROOT / "data", ROOT / "results"
REB = OUT / "rebuilt_graphs"
REPO_SHORT = {"fastapi/fastapi": "fastapi", "Textualize/rich": "rich", "psf/requests": "requests", "encode/httpx": "httpx"}
KS = (1, 5, 10, 20)
METHODS = ["bm25_text", "symbol_seed", "emb_expand", "ppr", "hybrid"]

tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in (DATA / "tasks.jsonl").open(encoding="utf-8") if l.strip()}
gold = {json.loads(l)["instance_id"]: json.loads(l) for l in (OUT / "gold.jsonl").open(encoding="utf-8")}
rel_graphs = {graph_key(p): p for p in (DATA / "graphs").glob("*.json")}
emb_files = {graph_key(p): p for p in (DATA / "embeddings").glob("*.npz")}
reb_graphs = {graph_key(p): p for p in REB.glob("*.json")}


def evaluate(ranked, gold_all, ids):
    gin = {s for s in gold_all if s in ids}
    out = {f"recall@{k}": len(set(ranked[:k]) & gin) / len(gold_all) for k in KS}
    out |= {f"hit@{k}": hit_at(ranked, gin, k) for k in KS}
    out["mrr"] = mrr(ranked, gin)
    out["coverage"] = len(gin) / len(gold_all)
    return out


def run_condition(name, task_ids, graph_for, use_emb, exclude_tests, directed):
    rows = []
    for iid in task_ids:
        t = tasks[iid]; key = (REPO_SHORT[t["repo"]], t["base_commit"])
        gp = graph_for(key)
        if gp is None:
            continue
        g = load_graph(gp)
        ek, em = (load_embeddings(emb_files[key]) if use_emb and key in emb_files else (None, None))
        idx = GraphIndex(g, ek, em, exclude_tests=exclude_tests, directed=directed)
        ids = set(idx.nodes)
        row = {"instance_id": iid, "repo": key[0], "n_gold": len(gold[iid]["symbols"]), "methods": {}}
        for m in METHODS:
            ranked = getattr(idx, m)(t["problem_statement"], k=max(KS))
            row["methods"][m] = evaluate(ranked, set(gold[iid]["symbols"]), ids)
        rows.append(row)
    print(f"[{name}] {len(rows)} tasks", flush=True)
    return rows


def boot_ci(vals, n=2000, seed=0):
    rng = random.Random(seed); m = []
    for _ in range(n):
        s = [vals[rng.randrange(len(vals))] for _ in vals]; m.append(sum(s) / len(s))
    m.sort(); return m[int(0.025 * n)], m[int(0.975 * n)]


def summarize(rows):
    s = {"n_tasks": len(rows), "coverage": sum(r["methods"][METHODS[0]]["coverage"] for r in rows) / len(rows), "methods": {}}
    for m in METHODS:
        s["methods"][m] = {}
        for metric in [f"recall@{k}" for k in KS] + [f"hit@{k}" for k in KS] + ["mrr"]:
            vals = [r["methods"][m][metric] for r in rows]
            lo, hi = boot_ci(vals)
            s["methods"][m][metric] = {"mean": sum(vals) / len(vals), "ci95": [lo, hi]}
    return s


with_gold = [i for i in tasks if gold[i]["symbols"]]
with_rel = [i for i in with_gold if (REPO_SHORT[tasks[i]["repo"]], tasks[i]["base_commit"]) in rel_graphs]
no_rel = [i for i in with_gold if i not in set(with_rel)]
print("tasks with gold:", len(with_gold), "| with released graph:", len(with_rel), "| without:", len(no_rel))

conditions = {
    "released":            dict(task_ids=with_rel, graph_for=lambda k: rel_graphs.get(k), use_emb=True,  exclude_tests=False, directed=False),
    "released-notests":    dict(task_ids=with_rel, graph_for=lambda k: rel_graphs.get(k), use_emb=True,  exclude_tests=True,  directed=False),
    "rebuilt":             dict(task_ids=with_rel, graph_for=lambda k: reb_graphs.get(k), use_emb=False, exclude_tests=False, directed=False),
    "rebuilt-notests":     dict(task_ids=with_rel, graph_for=lambda k: reb_graphs.get(k), use_emb=False, exclude_tests=True,  directed=False),
    "rebuilt-notests-dir": dict(task_ids=with_rel, graph_for=lambda k: reb_graphs.get(k), use_emb=False, exclude_tests=True,  directed=True),
    "rebuilt-relemb-notests": dict(task_ids=with_rel, graph_for=lambda k: reb_graphs.get(k), use_emb=True, exclude_tests=True, directed=False),
    "rebuilt-notests-ALL": dict(task_ids=with_gold, graph_for=lambda k: reb_graphs.get(k), use_emb=False, exclude_tests=True, directed=False),
    "rebuilt-notests-NOREL": dict(task_ids=no_rel, graph_for=lambda k: reb_graphs.get(k), use_emb=False, exclude_tests=True, directed=False),
}
summary, per_task = {}, {}
for name, cfg in conditions.items():
    rows = run_condition(name, **cfg)
    if rows:
        summary[name] = summarize(rows); per_task[name] = rows
(OUT / "compare_graphs.json").write_text(json.dumps(summary, indent=2))
with (OUT / "compare_graphs_per_task.jsonl").open("w", encoding="utf-8") as f:
    for name, rows in per_task.items():
        for r in rows:
            f.write(json.dumps({"condition": name, **r}) + "\n")

print(f"\n{'condition':26s} {'n':>4s} {'cov':>6s} | " + " | ".join(f"{m[:11]:>23s}" for m in METHODS))
print(" " * 40 + " | ".join(f"{'H@5':>7s}{'R@10':>8s}{'MRR':>8s}" for _ in METHODS))
for name, s in summary.items():
    print(f"{name:26s} {s['n_tasks']:4d} {s['coverage']:6.3f} | " + " | ".join(
        f"{s['methods'][m]['hit@5']['mean']:7.3f}{s['methods'][m]['recall@10']['mean']:8.3f}{s['methods'][m]['mrr']['mean']:8.3f}" for m in METHODS))
