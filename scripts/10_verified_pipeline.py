"""Generalization study: the same gold construction, graph builder and LLM-free baselines on SWE-bench Verified.

Stages (run in order; each is idempotent and resumable):
  gold      -> results_verified/gold.jsonl
  build     -> results_verified/graphs/<repo>_<sha>.json.gz      (--shard k/M to parallelise)
  bench     -> results_verified/bench_per_task.jsonl               (--shard k/M to parallelise)
  summary   -> results_verified/summary.json  (overall, per repo, per difficulty; bootstrap CIs)

Usage: python scripts/10_verified_pipeline.py <stage> [--shard k/M]
"""
import json, sys, subprocess, tempfile, time, tarfile, io, random, collections, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from swegraphbench.gold import gold_for_task  # noqa
from swegraphbench.build_graph import build  # noqa
from swegraphbench.graphio import load_graph, save_graph, graph_key  # noqa
from swegraphbench.retrieval import GraphIndex, hit_at, mrr  # noqa

TASKS = ROOT / "data_verified" / "tasks_verified.jsonl"
OUT = ROOT / "results_verified"
GRAPHS = OUT / "graphs"
OUT.mkdir(exist_ok=True); GRAPHS.mkdir(exist_ok=True)
KS = (1, 5, 10, 20)
METHODS = ["bm25_text", "symbol_seed", "ppr", "hybrid"]

tasks = [json.loads(l) for l in TASKS.open(encoding="utf-8") if l.strip()]
repo_dir = lambda repo: ROOT / "repos" / repo.split("/")[1]
short = lambda repo: repo.split("/")[1]


def shard(items):
    for a in sys.argv[2:]:
        if a.startswith("--shard"):
            k, m = map(int, sys.argv[sys.argv.index(a) + 1].split("/")) if a == "--shard" else map(int, a.split("=")[1].split("/"))
            return [x for i, x in enumerate(items) if i % m == k]
    return items


def stage_gold():
    rows = []
    for t in tasks:
        g = gold_for_task(t, repo_dir(t["repo"]))
        rows.append(g.to_json())
    with (OUT / "gold.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n = sum(1 for r in rows if r["symbols"])
    print(f"gold: {len(rows)} tasks, {n} with >=1 symbol")


def stage_build():
    pairs = sorted({(t["repo"], t["base_commit"]) for t in tasks})
    pairs = shard(pairs)
    stats_f = (OUT / "build_stats.jsonl").open("a", encoding="utf-8")
    for repo, sha in pairs:
        out = GRAPHS / f"{short(repo)}_{sha}.json.gz"
        if out.exists():
            continue
        t0 = time.time()
        tar_bytes = subprocess.run(["git", "-C", str(repo_dir(repo)), "archive", "--format=tar", sha], capture_output=True, check=True).stdout
        with tempfile.TemporaryDirectory(prefix="swegb_") as td:
            with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tf:
                tf.extractall(td, filter="data")
            t1 = time.time()
            g = build(Path(td), f"{short(repo)}_{sha}")
        save_graph(g, out)
        s = {"repo": repo, "sha": sha, "n_nodes": len(g["nodes"]), "n_edges": len(g["edges"]),
             "n_calls": sum(e["type"] == "calls" for e in g["edges"]), "extract_s": round(t1 - t0, 1), "build_s": round(time.time() - t1, 1)}
        stats_f.write(json.dumps(s) + "\n"); stats_f.flush()
        print(json.dumps(s), flush=True)


def stage_bench():
    gold = {json.loads(l)["instance_id"]: json.loads(l) for l in (OUT / "gold.jsonl").open(encoding="utf-8")}
    done = set()
    per = OUT / "bench_per_task.jsonl"
    if per.exists():
        done = {json.loads(l)["instance_id"] for l in per.open(encoding="utf-8")}
    todo = shard([t for t in tasks if gold[t["instance_id"]]["symbols"] and t["instance_id"] not in done])
    with per.open("a", encoding="utf-8") as f:
        for t in todo:
            gp = GRAPHS / f"{short(t['repo'])}_{t['base_commit']}.json.gz"
            if not gp.exists():
                continue
            t0 = time.time()
            idx = GraphIndex(load_graph(gp), exclude_tests=True)
            ids = set(idx.nodes)
            gall = set(gold[t["instance_id"]]["symbols"]); gin = {s for s in gall if s in ids}
            row = {"instance_id": t["instance_id"], "repo": t["repo"], "difficulty": t.get("difficulty"), "n_gold": len(gall),
                   "coverage": len(gin) / len(gall), "n_nodes": len(ids), "methods": {}}
            for m in METHODS:
                ranked = getattr(idx, m)(t["problem_statement"], k=max(KS))
                row["methods"][m] = {f"recall@{k}": len(set(ranked[:k]) & gin) / len(gall) for k in KS} | \
                                    {f"hit@{k}": hit_at(ranked, gin, k) for k in KS} | {"mrr": mrr(ranked, gin)}
            row["seconds"] = round(time.time() - t0, 1)
            f.write(json.dumps(row) + "\n"); f.flush()
            print(t["instance_id"], f"nodes={len(ids)} cov={row['coverage']:.2f} hybrid h5={row['methods']['hybrid']['hit@5']:.0f} {row['seconds']}s", flush=True)


def boot_ci(vals, n=2000, seed=0):
    rng = random.Random(seed); m = []
    for _ in range(n):
        s = [vals[rng.randrange(len(vals))] for _ in vals]; m.append(sum(s) / len(s))
    m.sort(); return m[int(0.025 * n)], m[int(0.975 * n)]


def summarize(rows):
    s = {"n_tasks": len(rows), "coverage": sum(r["coverage"] for r in rows) / len(rows), "methods": {}}
    for m in METHODS:
        s["methods"][m] = {}
        for metric in [f"hit@{k}" for k in KS] + [f"recall@{k}" for k in KS] + ["mrr"]:
            vals = [r["methods"][m][metric] for r in rows]
            s["methods"][m][metric] = {"mean": sum(vals) / len(vals), "ci95": boot_ci(vals)} if len(rows) >= 5 else {"mean": sum(vals) / len(vals)}
    return s


def stage_summary():
    rows = [json.loads(l) for l in (OUT / "bench_per_task.jsonl").open(encoding="utf-8")]
    gold = [json.loads(l) for l in (OUT / "gold.jsonl").open(encoding="utf-8")]
    builds = [json.loads(l) for l in (OUT / "build_stats.jsonl").open(encoding="utf-8")] if (OUT / "build_stats.jsonl").exists() else []
    out = {"overall": summarize(rows), "per_repo": {}, "per_difficulty": {},
           "gold": {"tasks": len(gold), "with_symbols": sum(1 for g in gold if g["symbols"]),
                    "only_new_files": sum(1 for g in gold if g["new_files"] and not g["symbols"])},
           "build": {"n_graphs": len(builds), "build_s_median": statistics.median(b["build_s"] for b in builds) if builds else None,
                     "build_s_max": max((b["build_s"] for b in builds), default=None),
                     "n_nodes_median": statistics.median(b["n_nodes"] for b in builds) if builds else None,
                     "n_nodes_max": max((b["n_nodes"] for b in builds), default=None)}}
    for repo in sorted({r["repo"] for r in rows}):
        out["per_repo"][repo] = summarize([r for r in rows if r["repo"] == repo])
    for d in sorted({r["difficulty"] for r in rows if r["difficulty"]}):
        out["per_difficulty"][d] = summarize([r for r in rows if r["difficulty"] == d])
    (OUT / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"{'group':28s} {'n':>4s} {'cov':>6s} | " + " | ".join(f"{m[:10]:>20s}" for m in METHODS))
    print(" " * 42 + " | ".join(f"{'H@5':>6s}{'R@10':>7s}{'MRR':>7s}" for _ in METHODS))
    def line(name, s):
        print(f"{name:28s} {s['n_tasks']:4d} {s['coverage']:6.3f} | " + " | ".join(
            f"{s['methods'][m]['hit@5']['mean']:6.3f}{s['methods'][m]['recall@10']['mean']:7.3f}{s['methods'][m]['mrr']['mean']:7.3f}" for m in METHODS))
    line("ALL Verified", out["overall"])
    for k, v in out["per_repo"].items(): line(k, v)
    for k, v in out["per_difficulty"].items(): line(k, v)


if __name__ == "__main__":
    {"gold": stage_gold, "build": stage_build, "bench": stage_bench, "summary": stage_summary}[sys.argv[1]]()
