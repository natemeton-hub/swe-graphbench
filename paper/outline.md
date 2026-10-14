# SWE-GraphBench — working outline (numbers to be filled from results/)

**Title:** SWE-GraphBench: Auditing and Rebuilding Code Graphs for Local Software-Engineering Agents
**Subtitle:** A dependency-free graph builder, corrected graphs for all 129 public tasks, and a symbol-level localization benchmark for the Gemma 4 Developer Agent release
**Target award:** Best New Resource (also competitive for Overall Best Paper)
**Budget:** 3,000 words. Kaggle Writeup + public notebook + GitHub repo + Kaggle dataset of rebuilt graphs.

## Contributions (one sentence each; every one must map to a result file)
1. **Audit** of the released code-graph/embedding artifacts (results/audit_summary.json, node_mapping_summary.json):
   - graphs exist for only 68/129 tasks (fastapi+httpx); rich/requests tasks get no code-intelligence tools at all
   - 100% of call edges are reciprocal in every graph -> caller/callee direction is lost
   - median 68% of nodes isolated; median 9% of nodes are library code, 64% are tests
   - 0 async defs (fastapi core is async; N async defs missing), nested functions hoisted -> id collisions
   - embeddings: 256-d, unnormalised, deterministic per text (14.6% exact duplicates = duplicate texts), model undocumented, query must be a symbol name
   - only node attrs: id/name/text; no file path or line numbers
2. **Builder**: pure-stdlib `ast` graph builder (~300 lines, no deps), competition-compatible schema + extras
   (kind, file, lineno, is_async, parent, docstring; contains/imports/inherits/calls edges with recorded resolution strategy).
   Runs in ~20 s on fastapi (1,120 files) -> can run inside the air-gapped sandbox as a skill script, so it applies to the *hidden* repos.
3. **Corrected graphs** for all 127 (repo, commit) pairs incl. the 59 rich/requests commits with no released graph (Kaggle dataset).
4. **Benchmark**: 121 tasks with gold symbols derived from the reference patch (symbol = innermost def/class containing an edited old-file line),
   LLM-free baselines (BM25 name/text, symbol-seed, embedding expansion, PPR, RRF hybrid), Recall@k / Hit@k / MRR with bootstrap CIs,
   coverage ceiling reported alongside every number.
5. **Findings**: released vs rebuilt graph on identical tasks/methods; effect of test-node filtering; effect of edge direction; what the embeddings add over symbol-seeding (~nothing so far).

## Section plan (word targets)
- Abstract (150)
- 1 Introduction (350): local agents, 32k context, tool budget; graph tools are the only repo-level navigation; what is actually in the release; our resource.
- 2 Related work (400): SWE-bench family; localization (Agentless, LocAgent/LocBench, SweRank, CoSIL, RepoGraph, GREPO, ARISE); concurrent entrants (aghasalim, satiricalguru, G-HRR) — cite honestly, state what we add (rebuilt graphs, hoisting/direction audit, MRR, frozen release, in-sandbox builder).
- 3 The released artifacts: an audit (550): Table 1 audit stats; Figure 1 (node composition + isolated fraction); hoisting example.
- 4 SWE-GraphBench (650): 4.1 gold construction; 4.2 builder design + resolution strategies + validation vs released (confirmed pairs); 4.3 metrics & protocol.
- 5 Experiments (600): Table 2 released vs rebuilt; Table 3 rich/requests unlocked; ablations (tests filter, direction, embeddings).
- 6 Limitations & ethics (150): static resolution misses dynamic dispatch; gold = patch-touched symbols (necessary not sufficient); public tasks only; no LLM-in-loop numbers (optional Gemma 4 E2B rerank on Kaggle if time).
- 7 Conclusion (100)
- References

## Open items
- [ ] rebuild all 127 graphs (running)
- [ ] 05_compare_graphs, 06_node_mapping_audit
- [ ] figures (dataviz skill)
- [ ] optional: rebuilt-node embeddings with an open model on Kaggle GPU; optional Gemma 4 E2B reranker
- [ ] Kaggle dataset upload (rebuilt graphs + gold.jsonl + results), public notebook, GitHub repo, LICENSE (Apache-2.0)
- [ ] user to confirm author name / affiliation for the writeup
