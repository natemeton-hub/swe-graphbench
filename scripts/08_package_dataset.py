"""Assemble the Kaggle dataset folder (results/kaggle_dataset) for `kaggle datasets create/version`.

Contents:
  swegraphbench/            the package (builder, gold, retrieval, graphio)
  scripts/                  the pipeline scripts
  tests/                    builder unit test
  gold.jsonl                gold symbols for 129 tasks
  rebuilt_graphs/*.json     127 corrected graphs
  results/*.json|jsonl      audit, baseline and comparison outputs
  README.md, LICENSE, requirements.txt, dataset-metadata.json
"""
import json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "results" / "kaggle_dataset"
USER = sys.argv[1] if len(sys.argv) > 1 else "nateshingleton"
SLUG = "swe-graphbench"

if DST.exists():
    shutil.rmtree(DST)
DST.mkdir(parents=True)
for d in ["swegraphbench", "scripts", "tests"]:
    shutil.copytree(ROOT / d, DST / d, ignore=shutil.ignore_patterns("__pycache__"))
for f in ["README.md", "LICENSE", "requirements.txt"]:
    shutil.copy(ROOT / f, DST / f)
shutil.copy(ROOT / "results" / "gold.jsonl", DST / "gold.jsonl")
shutil.copytree(ROOT / "results" / "rebuilt_graphs", DST / "rebuilt_graphs")
(DST / "results").mkdir()
for p in (ROOT / "results").glob("*.json*"):
    shutil.copy(p, DST / "results" / p.name)
meta = {
    "title": "SWE-GraphBench: corrected code graphs + localization benchmark (Gemma 4 Developer Agent)",
    "id": f"{USER}/{SLUG}",
    "licenses": [{"name": "Apache 2.0"}],
    "keywords": ["software engineering", "code graph", "fault localization", "gemma", "benchmark"],
    "subtitle": "Pure-stdlib graph builder, graphs for all 127 task commits, gold symbols for 129 tasks, audit + baseline results",
}
(DST / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
total = sum(p.stat().st_size for p in DST.rglob("*") if p.is_file())
print("packaged", DST, f"{total/1e6:.1f} MB", "files:", sum(1 for p in DST.rglob('*') if p.is_file()))
