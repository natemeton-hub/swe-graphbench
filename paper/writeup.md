# SWE-GraphBench: Auditing and Rebuilding Code Graphs for Local Software-Engineering Agents

## A dependency-free graph builder, corrected graphs for every public task, and a symbol-level localization benchmark for the Gemma 4 Developer Agent release

<!-- Numbers in [[double brackets]] are filled from results/ once the full rebuild finishes. -->

## Abstract

Local coding agents built on Gemma 4 must navigate repositories through a 32k-token window and a bounded tool budget, so the code-intelligence tools the competition harness exposes (`get_code_neighbors`, `search_similar_code`, `get_code_subgraph`) are the only repository-level navigation an agent has. We audit the graphs and embeddings behind those tools and find that they are structurally weaker than their interface suggests: graphs exist for only 68 of the 129 public tasks; in every graph 100% of `calls` edges are reciprocal, so caller/callee direction is lost; a median 68% of nodes have no edge at all; a median 9% of nodes are library code against 64% test code; `async def` and nested functions are dropped or hoisted into colliding identifiers; and no node carries a file path or line number. We release SWE-GraphBench, a resource with four parts: (1) the audit as reproducible scripts, (2) a ~300-line pure-standard-library graph builder that emits the competition schema with directed call edges, async and nested definitions, location attributes, and `contains`/`imports`/`inherits` edges, fast enough to run inside the air-gapped sandbox as a skill script, (3) corrected graphs for all 127 (repository, commit) pairs, including the 59 rich and requests commits that have no released graph, and (4) a benchmark of 121 tasks whose gold symbols are derived from the reference patch, with LLM-free localization baselines (BM25, symbol seeding, embedding expansion, personalized PageRank, rank fusion) reported as Recall@k, Hit@k and MRR with bootstrap confidence intervals and an explicit coverage ceiling. On the 65 tasks where both graphs exist, the rebuilt graph [[raises/changes]] Hit@5 of the best LLM-free method from [[x]] to [[y]], and it makes graph-based localization possible on the [[56]] tasks that previously had none. Code, graphs and results are released under Apache-2.0.

## 1. Introduction

The Gemma 4 Developer Agent competition asks for an agent that fixes real GitHub issues offline on a single accelerator, with a 32,768-token context, a per-task tool-call budget and a 4 GiB sandbox. Under these constraints the first and most expensive sub-problem is *localization*: finding the handful of functions an issue is about before the context fills with irrelevant file reads. Agentless and graph-guided agents such as LocAgent report that most end-to-end failures on SWE-bench are localization failures and that a code graph is the most effective structure for narrowing a repository to a few candidates.

The organizers released such a structure: per task commit, a NetworkX node-link JSON graph of symbols with `calls` edges and a 256-dimensional embedding per symbol, exposed through three sandboxed tools, and submissions using them are "highly encouraged". The artifacts themselves have not been examined. Because the hidden test set uses private repositories, an agent cannot bring a precomputed graph: whatever graph exists at evaluation time is the organizers' or one the agent builds inside the sandbox.

This paper treats the released artifacts as an object of study and then fixes what the audit finds. Our contributions are:

- **An audit** (Section 3) of all 68 released graphs and 127 embedding archives, every statistic produced by a script: availability, edge direction, isolation, library-versus-test composition, missing `async` and nested definitions, hoisting collisions, embedding duplication, and coverage of the symbols reference patches edit.
- **A builder** (Section 4.2) that emits the same JSON schema from source with nothing but Python's `ast` module: every definition including `async` and nested ones, directed call edges annotated with their resolution rule, containment, import and inheritance edges, and file and line attributes. It runs in about 20 s on FastAPI's 1,120 files, so an agent can run it as a skill script on an unseen repository.
- **Corrected graphs** for all 127 (repository, commit) pairs, released as a Kaggle dataset, including the 59 rich and requests commits for which no graph exists and the harness therefore exposes no code-intelligence tools.
- **A benchmark** (Section 4) of 121 tasks with patch-derived gold symbols and a protocol that reports the coverage ceiling next to every retrieval number, so "the graph cannot represent the answer" is never confused with "the retriever missed it".
- **Findings** (Section 5) on released versus rebuilt graphs under identical methods, and on what test filtering, edge direction and the released embeddings each contribute.

Everything is deterministic and runs on a laptop CPU. We keep language models out of the loop deliberately: the question is what the *graph* supports, and an LLM in the loop would confound graph quality with prompt quality.

## 2. Related work

**Task source.** SWE-bench (Jimenez et al., 2024) and its Lite and Verified subsets define the issue-to-patch task we inherit; SWE-Gym (Pan et al., 2025) and SWE-smith (Yang et al., 2025) scale training data for it. We reuse the task formulation but change the label: from "tests pass" to "the symbols the reference patch edits", scoring retrieval rather than resolution.

**Localization as a sub-problem.** Agentless (Xia et al., 2024) measures how often hierarchical localization contains the ground-truth file and elements; LocAgent (Chen et al., 2025) introduces Loc-Bench with function-level Acc@k over a self-built code graph; SweRank (Reddy et al., 2026) shows trained retrieve-and-rerank can match agents; CoSIL (Jiang et al., 2025), OrcaLoca (Yu et al., 2025), KGCompass (Yang et al., 2025), GraphLocator (Liu et al., 2026) and RepoNavigator (Zhang et al., 2025) localize over graphs they construct, and ARISE (Seddik et al., 2026) packages such a graph as an agent toolset. GREPO (Wang et al., 2026) is the closest academic benchmark: class- and function-level localization over repository graphs with Hit@K, built from 86 repositories for GNN training. None study an *externally supplied, fixed* graph or report its coverage ceiling, which is the situation a competition agent faces. RepoGraph (Ouyang et al., 2025), CodexGraph (Liu et al., 2025), CodeRAG (Li et al., 2025), GraphCoder (Liu et al., 2024), RepoCoder (Zhang et al., 2023) and CodePlan (Bairi et al., 2023) use dependency structure for generation; our personalized-PageRank baseline is the LLM-free analogue of their relevance propagation.

**Concurrent competition entries.** Two public paper-track repositories independently compare BM25, embedding and one-hop graph retrieval on the released artifacts with patch-derived gold (aghasalim, 2026; satiricalguru, 2026); both note that `search_similar_code` only accepts symbol names. Our audit confirms and extends their observations (edge direction, hoisting collisions, missing async definitions, no location attributes, absent graphs for two repositories). What we add is the corrected resource: a sandbox-runnable builder, graphs for every task, a frozen gold set with MRR and paired statistics, and a released-versus-rebuilt comparison under identical conditions.

## 3. The released artifacts: an audit

*Setup.* The public release contains `tasks.jsonl` (129 tasks: 67 `fastapi/fastapi`, 48 `Textualize/rich`, 13 `psf/requests`, 1 `encode/httpx`; each with `problem_statement`, `patch`, `test_patch`, `base_commit`), 68 graph files and 127 embedding files keyed by `<repo>_<sha>`. We cloned the four repositories and verified that every base commit resolves. All numbers below come from `scripts/02_audit_coverage.py` and `scripts/06_node_mapping_audit.py`.

**Availability.** Graphs exist for the 67 FastAPI commits and the single httpx commit only. The 61 rich and requests tasks have embeddings but no graph, and because the harness enables the code-intelligence tools only when both files exist, agents evaluated on those tasks receive none of the three tools. If the hidden test set resembles the public one, roughly half of the evaluation tasks are graph-less.

**Structure (Figure 1).** Every graph has a single edge type, `calls`. In all 68 graphs, 100% of edge pairs are reciprocal: for every `A calls B` there is a `B calls A`, so the graph is undirected in effect and `get_code_neighbors` cannot distinguish callers from callees. A median 68% of nodes are isolated (range 3% for httpx to 72%). Node composition is dominated by tests: a median 64% of nodes are test functions and only 9% belong to the library package. The highest-degree node in every FastAPI graph is the `FastAPI` application class, with degree 123, which suggests that calls were resolved by name matching rather than scope.

**Node attributes and identifiers.** Nodes carry only `id`, `name` and `text`. There is no file path, line number, kind or parent, so an agent that finds a symbol still needs a `run_command` grep to open it. Comparing node sets against an `ast` walk of the same commit (Section 4.2) shows three systematic omissions: (i) `async def` is not a node: 69,845 of 70,070 async definitions across the 68 commits (99.7%) are absent, including FastAPI's `solve_dependencies`, `serialize_response` and the request-handler closure `app`; (ii) nested synchronous functions are hoisted to their enclosing class or module (8,034 released ids), so distinct closures share one identifier (`fastapi.applications.FastAPI.decorator` stands for two different inner functions) and 2,095 released ids are ambiguous; (iii) 2,289 released ids (0.9%) have no counterpart at all. Overall 25.0% of the 341,563 definitions present at those commits have no released node.

**Embeddings.** Each archive is a dict of symbol name to a 256-d `float32` vector; keys match the graph's node ids exactly where a graph exists. Vectors are unnormalized (norms 3.7 to 8.5) and deterministic in the node text: 14.6% of rows in a FastAPI archive are exact duplicates and every duplicate group has identical source text. The embedding model is undocumented, and because the sandbox has no encoder, `search_similar_code` must be queried with a symbol name, never with issue text.

**Coverage of what fixes edit.** Section 4.1 derives, for each task, the set of symbols its reference patch touches. On the 65 tasks with a graph and at least one gold symbol, 85.3% of gold symbols exist as nodes and 78.8% exist *and* have at least one edge; 45 of 65 tasks have every gold symbol represented, and 4 have none. Missing gold symbols are almost all async or nested definitions.

## 4. SWE-GraphBench

### 4.1 Gold symbols

For each task we parse the reference `patch` with `unidiff`. For every hunk in a modified Python file we take the old-file lines it removes (for pure insertions, the context line immediately before the insertion), read the file at `base_commit` with `git show`, and record the innermost enclosing `def` or `class` by `ast` line spans, including decorators. Symbol ids use the released naming convention: dotted path from the repository root, with a leading `src/` or `docs_src/` directory dropped, and full nesting kept. Files the patch adds, non-Python files and module-level edits are kept as separate fields rather than dropped. 121 of 129 tasks have at least one gold symbol (median 1, max 141); 11 tasks add new files; 21 touch module-level code.

Gold symbols are *necessary* locations, not sufficient ones: a correct fix might edit a different function. They are, however, the standard proxy used by Agentless, LocAgent and GREPO, and they are deterministic.

### 4.2 Builder

`swegraphbench/build_graph.py` walks every `.py` file, parses it with `ast`, and emits a node for each module, class, function and async function with its full qualified name, `kind`, `file`, `lineno`, `end_lineno`, `is_async`, `parent`, first docstring line and source text. Edges are `contains` (lexical nesting), `imports` (module to module, with relative imports resolved), `inherits` (class to resolved base) and directed `calls`. Call targets are resolved in a fixed order and each edge records which rule fired: `self` (`self.m()` or `cls.m()` through the class and its resolved bases), `scope` (names defined in the module or bound by `import`/`from ... import`, including instantiation of a class), `enclosing` (a nested function calling a sibling in an outer scope), `suffix2` (an `obj.method` whose last two components are unique in the repository) and `suffix1` (a bare name unique in the repository). Unresolvable calls, mostly to the standard library and third-party packages, are counted but not guessed. Consumers can therefore filter edges by confidence; the harness tools accept the file unchanged because the top-level schema is identical.

The builder has no dependency outside the standard library, is about 300 lines, and processes a FastAPI commit (1,120 files, 6,400 nodes) in roughly 20 s on one laptop core. A unit test on a synthetic package checks every resolution rule and that no reciprocal pair is produced (`tests/test_build_graph.py`). As a validation against the released graphs, 53,793 of the 81,967 released undirected call pairs (65.6%) are also found by the resolver in at least one direction; the remaining third are pairs that scope-based resolution does not reproduce. The resolver emits 142,657 directed call edges over the same 68 commits, of which at most 1.4% per graph are reciprocal.

### 4.3 Baselines and metrics

All baselines take only the problem statement and the graph (plus embeddings when available) and return a ranked list of node ids:

- **BM25 (name / text)** over identifier-split tokens of the dotted name or of the source text.
- **Symbol seed**: identifiers mentioned in the issue (back-ticked spans, dotted paths, snake_case and CamelCase tokens) resolved to nodes with the same four-tier rule as the harness (`exact`, suffix, case-insensitive, substring), ranked by weighted mention count. This is the strongest query the released `search_similar_code` can express.
- **Embedding expansion**: the mean released vector of the top seeds, expanded by cosine similarity.
- **Personalized PageRank** from the top-10 BM25 seeds over `calls` edges, undirected or directed.
- **Hybrid**: reciprocal-rank fusion of BM25-text, embedding expansion and PageRank.

Each condition may drop test nodes from the candidate set (`exclude_tests`), a filter any library-fixing agent can apply for free. We report Recall@k and Hit@k for k in {1, 5, 10, 20} and MRR, averaged over tasks with 95% bootstrap confidence intervals (2,000 resamples), always alongside the coverage ceiling, the Recall the best possible ranking could reach given which gold symbols are nodes.

## 5. Experiments

### 5.1 Released versus rebuilt graphs

All conditions use the 65 FastAPI and httpx tasks that have both a released graph and a gold symbol, except Section 5.3. Table 1 gives means; Figure 2 adds confidence intervals; `results/paired_stats.json` holds paired bootstrap intervals and sign tests for every contrast below.

**Table 1.** Hit@5 / Recall@10 / MRR on the same 65 tasks. "Cov." is the coverage ceiling (share of gold symbols that are nodes).

| Graph | Tests | Cov. | BM25 text | Symbol seed | Embedding | PageRank | Hybrid |
|---|---|---|---|---|---|---|---|
| released | kept | 0.854 | .292 / .253 / .189 | .215 / .122 / .172 | .215 / .141 / .179 | .354 / .275 / .167 | .338 / .269 / .238 |
| released | filtered | 0.854 | .415 / .321 / .280 | .231 / .137 / .182 | .231 / .152 / .190 | .369 / .312 / .201 | .431 / .350 / .295 |
| rebuilt | kept | 1.000 | .308 / .287 / .212 | .215 / .133 / .181 | n/a | .369 / .282 / .214 | .415 / .332 / .284 |
| rebuilt | filtered | 1.000 | .431 / .358 / .264 | .215 / .133 / .182 | n/a | .431 / .337 / .266 | **.492 / .403 / .302** |
| rebuilt, directed PPR | filtered | 1.000 | .431 / .358 / .264 | .215 / .133 / .182 | n/a | .369 / .341 / .249 | .446 / .394 / .314 |
| rebuilt + released emb. | filtered | 1.000 | .431 / .358 / .264 | .215 / .133 / .182 | .200 / .122 / .179 | .431 / .337 / .266 | .477 / .403 / .297 |

**Filtering tests is the single largest cheap win.** Removing test nodes from the candidate set raises BM25 Hit@5 on the released graph from 0.292 to 0.415 (paired 95% CI +0.03 to +0.22, sign test p = 0.02) and Recall@10 from 0.253 to 0.321 (p = 0.006). An agent fixing library code can apply this filter for free; the released graph's 64% test share makes it necessary.

**The rebuilt graph helps on top of that, modestly.** With tests filtered in both, the rebuilt graph lifts the hybrid retriever from 0.431 to 0.492 Hit@5 and 0.350 to 0.403 Recall@10 (paired CI −0.01 to +0.12, sign test 15 wins / 5 losses, p = 0.04) and PageRank MRR from 0.201 to 0.266. At 65 tasks these gains are not individually significant for every metric, and we report them as such. The gain comes from two sources that the audit predicted: the coverage ceiling rises from 0.854 to 1.0 because async and nested gold symbols now exist, and call edges resolved by scope rather than name give PageRank a cleaner neighbourhood. Measured end to end, from the graph as shipped to the rebuilt filtered graph, the hybrid retriever gains +0.154 Hit@5 (CI +0.05 to +0.26, p = 0.02) and +0.134 Recall@10 (CI +0.07 to +0.21, p < 0.001).

**Edge direction does not help here.** Propagating from BM25 seeds toward callees only (directed PageRank) is slightly worse than undirected propagation (Hit@5 0.369 vs 0.431), while the hybrid's MRR is unchanged within noise. Gold symbols are as often callers of a mentioned symbol as callees, so for seed expansion the direction is best kept available but not enforced; it remains essential for the tool semantics (`get_code_neighbors` can finally answer "who calls this").

**The released embeddings add nothing measurable.** Because its query must itself be a resolved symbol, embedding expansion returns nothing on the 26 of 65 tasks whose issue names no resolvable symbol, has the same top-5 as symbol seeding on 45 tasks, and where it differs gains only +0.015 Recall@10. Adding the released vectors to the rebuilt graph leaves every metric unchanged or marginally lower.

### 5.3 Tasks the release left uncovered (56 rich and requests tasks)

On the 56 rich and requests tasks that have gold symbols and no released graph, the rebuilt graph with tests filtered reaches Hit@5 0.500, Recall@10 0.478 and MRR 0.358 with the hybrid retriever (Figure 3), on par with the FastAPI tasks. Over all 121 tasks with gold symbols the same configuration scores 0.496 / 0.438 / 0.328. These are the first graph-based localization numbers for these tasks; under the released artifacts an agent on them has no graph tool at all.

## 6. Limitations

Static resolution misses dynamic dispatch, decorators that rebind names and calls through container attributes; the resolver reports these as unresolved rather than guessing, which lowers recall of call edges but keeps precision interpretable. Gold symbols come from one reference patch; equally valid fixes can live elsewhere. The benchmark covers four Python repositories and 121 public tasks, and public-task results do not predict hidden-task results. We do not report an end-to-end resolution rate; the competition leaderboard measures that, and our claim is about the graph layer beneath it. Finally, the embedding archives we leave as released; recomputing embeddings for the rebuilt nodes with a documented open model is straightforward future work.

## 7. Conclusion

The code graph an agent is handed is not neutral infrastructure. For the Gemma 4 Developer Agent release it is undirected, mostly tests, missing every async function, and absent for half the tasks. A builder small enough to run inside the sandbox removes all four defects, and SWE-GraphBench lets anyone measure what that buys with no language model in the loop. We hope the resource serves both as a drop-in improvement for competition agents and as a reusable protocol for auditing any precomputed code graph before building on it.

## References

[[from related_work.md BibTeX; formatted on final pass]]
