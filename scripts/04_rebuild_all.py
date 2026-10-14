"""Rebuild a code graph for every (repo, base_commit) in tasks.jsonl -> results/rebuilt_graphs/<repo>_<sha>.json

Uses `git archive` to materialise each commit in a temp dir (no checkout of the clone), then the
pure-stdlib builder. Also records build time and basic stats per graph.
"""
import json, sys, subprocess, tempfile, time, shutil, tarfile, io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from swegraphbench.build_graph import build  # noqa

DATA, OUT = ROOT / "data", ROOT / "results" / "rebuilt_graphs"
OUT.mkdir(parents=True, exist_ok=True)
REPO_DIR = {"fastapi/fastapi": "fastapi", "Textualize/rich": "rich", "psf/requests": "requests", "encode/httpx": "httpx"}

tasks = [json.loads(l) for l in (DATA / "tasks.jsonl").open(encoding="utf-8") if l.strip()]
pairs = sorted({(REPO_DIR[t["repo"]], t["base_commit"]) for t in tasks})
if "--reverse" in sys.argv:  # allows a second worker to meet the first in the middle
    pairs = pairs[::-1]
print("commits to build:", len(pairs))
stats = []
for repo, sha in pairs:
    out = OUT / f"{repo}_{sha}.json"
    if out.exists():
        continue
    t0 = time.time()
    tar_bytes = subprocess.run(["git", "-C", str(ROOT / "repos" / repo), "archive", "--format=tar", sha], capture_output=True, check=True).stdout
    with tempfile.TemporaryDirectory(prefix="swegb_") as td:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tf:
            tf.extractall(td, filter="data")
        t1 = time.time()
        g = build(Path(td), f"{repo}_{sha}")
    t2 = time.time()
    out.write_text(json.dumps(g), encoding="utf-8")
    s = {"repo": repo, "sha": sha, "n_nodes": len(g["nodes"]), "n_edges": len(g["edges"]),
         "n_calls": sum(e["type"] == "calls" for e in g["edges"]), "extract_s": round(t1 - t0, 2), "build_s": round(t2 - t1, 2),
         "resolution": g["graph"]["call_resolution_stats"]}
    stats.append(s)
    print(json.dumps(s), flush=True)
with (ROOT / "results" / "rebuild_stats.jsonl").open("a", encoding="utf-8") as f:
    for s in stats:
        f.write(json.dumps(s) + "\n")
print("done")
