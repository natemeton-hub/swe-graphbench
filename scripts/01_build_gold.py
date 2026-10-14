"""Build gold symbol sets for every task -> results/gold.jsonl + results/gold_summary.json"""
import json, sys, subprocess, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from swegraphbench.gold import gold_for_task  # noqa: E402

DATA = ROOT / "data"
REPOS = ROOT / "repos"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

REPO_DIR = {"fastapi/fastapi": "fastapi", "Textualize/rich": "rich", "psf/requests": "requests", "encode/httpx": "httpx"}

tasks = [json.loads(l) for l in (DATA / "tasks.jsonl").open(encoding="utf-8") if l.strip()]

# 1. verify commits exist
missing = []
for t in tasks:
    rd = REPOS / REPO_DIR[t["repo"]]
    r = subprocess.run(["git", "-C", str(rd), "cat-file", "-e", f'{t["base_commit"]}^{{commit}}'], capture_output=True)
    if r.returncode != 0:
        missing.append(t["instance_id"])
print("tasks:", len(tasks), "| base commits missing locally:", len(missing), missing[:10])

# 2. gold
rows = []
agg = collections.Counter()
per_repo = collections.defaultdict(collections.Counter)
for t in tasks:
    g = gold_for_task(t, REPOS / REPO_DIR[t["repo"]])
    rows.append(g.to_json())
    r = t["repo"]
    agg["tasks"] += 1; per_repo[r]["tasks"] += 1
    if g.symbols: agg["with_symbols"] += 1; per_repo[r]["with_symbols"] += 1
    if g.new_files: agg["with_new_files"] += 1; per_repo[r]["with_new_files"] += 1
    if g.new_files and not g.symbols: agg["only_new_files"] += 1; per_repo[r]["only_new_files"] += 1
    if g.non_python_files and not g.python_files and not g.new_files: agg["only_non_python"] += 1; per_repo[r]["only_non_python"] += 1
    if g.module_level_files: agg["with_module_level"] += 1
    agg["n_symbols_total"] += len(g.symbols)
    agg["n_py_files_total"] += len(g.python_files)

with (OUT / "gold.jsonl").open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

n_sym = [len(r["symbols"]) for r in rows if r["symbols"]]
n_files = [len(r["python_files"]) for r in rows if r["python_files"]]
summary = {
    "aggregate": dict(agg),
    "per_repo": {k: dict(v) for k, v in per_repo.items()},
    "symbols_per_task_with_symbols": {"min": min(n_sym), "median": sorted(n_sym)[len(n_sym)//2], "max": max(n_sym)},
    "py_files_per_task": {"min": min(n_files), "median": sorted(n_files)[len(n_files)//2], "max": max(n_files)},
    "missing_commits": missing,
}
(OUT / "gold_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print("example:", json.dumps(rows[1], indent=1)[:1200])
