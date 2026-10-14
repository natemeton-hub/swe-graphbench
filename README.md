# SWE-GraphBench

**Auditing and rebuilding code graphs for local software-engineering agents.**
Built on the public release of the Kaggle *Gemma 4 Developer Agent* competition
(`tasks.jsonl`, `graphs/*.json`, `embeddings/*.npz`).

The resource has four parts:

| Part | What it is | Where |
| --- | --- | --- |
| Audit | Deterministic statistics of the released graphs and embeddings (direction, isolation, composition, missing async / nested defs, coverage of gold-patch symbols) | `scripts/02_audit_coverage.py`, `scripts/06_node_mapping_audit.py`, `results/*.json` |
| Builder | Pure-stdlib `ast` code-graph builder emitting the competition schema plus `kind`, `file`, `lineno`, `is_async`, `parent`, `docstring` and `contains` / `imports` / `inherits` / directed `calls` edges with a recorded resolution strategy | `swegraphbench/build_graph.py` (no third-party deps) |
| Corrected graphs | One graph per (repo, base_commit) for all 127 commits in `tasks.jsonl`, including the 59 commits that have no released graph | `scripts/04_rebuild_all.py` -> Kaggle dataset |
| Benchmark | Gold symbol sets for 121 tasks (innermost def/class touched by the reference patch) and LLM-free localization baselines (BM25, symbol-seed, embedding expansion, personalized PageRank, RRF hybrid) with Recall@k / Hit@k / MRR and bootstrap CIs | `swegraphbench/gold.py`, `swegraphbench/retrieval.py`, `scripts/03_*.py`, `scripts/05_*.py` |

## Quick start

```bash
pip install -r requirements.txt
# 1. competition data (requires joining the competition on Kaggle)
kaggle competitions download -c gemma-4-developer-agent -p data && cd data && unzip -q *.zip && cd ..
# 2. upstream repositories at the task commits
mkdir repos && for r in fastapi/fastapi Textualize/rich psf/requests encode/httpx; do git clone --no-checkout https://github.com/$r repos/${r#*/}; done
# 3. pipeline
python scripts/01_build_gold.py          # results/gold.jsonl
python scripts/02_audit_coverage.py      # results/audit_summary.json, graph_audit.jsonl, coverage.jsonl
python scripts/03_baselines_released.py  # results/baselines_released.json
python scripts/04_rebuild_all.py         # results/rebuilt_graphs/*.json  (~25 s per commit, CPU)
python scripts/05_compare_graphs.py      # results/compare_graphs.json
python scripts/06_node_mapping_audit.py  # results/node_mapping_summary.json
python tests/test_build_graph.py         # builder unit test
```

Build a graph for any Python repository (no dependencies beyond the standard library):

```bash
python -m swegraphbench.build_graph /path/to/repo out.json --name myrepo
```

## Graph schema

Compatible with the competition's NetworkX node-link JSON (`directed`, `multigraph`, `graph`, `nodes`, `edges`);
the harness tools `get_code_neighbors` / `get_code_subgraph` can load it unchanged. Extra node attributes:
`kind` (module | class | function), `file`, `lineno`, `end_lineno`, `is_async`, `parent`, `docstring`.
Extra edge types: `contains`, `imports`, `inherits`. `calls` edges are directed caller -> callee and carry
`resolution` in {`self`, `scope`, `enclosing`, `suffix2`, `suffix1`} so consumers can filter by confidence.

Node ids follow the released convention: dotted path from the repository root with a leading `src/` or
`docs_src/` directory dropped, full nesting kept (`pkg.module.Class.method`, `pkg.module.outer.inner`).

## Gold construction

For every task, each hunk of the reference `patch` is mapped to the old-file lines it touches
(pure insertions attribute to the preceding context line); the innermost enclosing `def`/`class` at the
base commit becomes a gold symbol. Added files, non-Python files and module-level edits are recorded
separately rather than silently dropped (`results/gold.jsonl`).

## License

Apache-2.0. The competition data itself is governed by the competition rules and is not redistributed here.
