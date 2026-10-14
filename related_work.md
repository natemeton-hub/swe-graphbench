# Related Work Survey: Symbol-Level Fault-Localization Benchmark over Competition Code Graphs

Survey date: 2026-10-13. Scope: prior work (2023-2026) relevant to a resource paper for the Kaggle "Gemma 4 Developer Agent" Paper Track proposing a symbol-level fault-localization benchmark + toolkit built on the competition's released code graphs (NetworkX MultiDiGraph JSON, dotted-path symbol nodes) and precomputed 256-d node embeddings (.npz), evaluating BM25 / embedding kNN / personalized-PageRank / hybrid retrieval with Recall@k and MRR, plus a graph-coverage analysis.

Verification status: every entry below marked VERIFIED was confirmed by fetching an arXiv abstract page, a project page, or a GitHub README, or by a search result that quoted the page. Items marked PARTIALLY VERIFIED have one field (usually venue) that could not be confirmed from the fetched page. UNVERIFIED items are explicitly labelled.

---

## 0. Novelty assessment (read this first)

**Does any prior work already do exactly what we propose?** No single peer-reviewed paper builds a symbol-level localization benchmark over *the competition's precomputed code graphs and embeddings*. However, the overlap is substantial in three places and must be acknowledged explicitly:

1. **Concurrent competition entries (highest overlap).** Two public Paper-Track repos already (a) derive gold symbols from the reference patch, (b) evaluate the competition's `search_similar_code` embeddings and `get_code_neighbors` graph against BM25/grep baselines, and (c) audit the released graph data:
   - **aghasalim/gemma4-code-graph-localization** (MIT, 2026): gold = functions/classes edited by the patch, named with the graph's dotted convention; methods = embed, graph (seeds + 1-hop call neighbours), grep, BM25, BM25 with tests filtered; metric = hit@5 with bootstrap CIs over 129 training tasks; data audit (no `async def` nodes in the graph, 21.9% duplicate embedding vectors, every edge is `calls`, embedding tool only accepts symbol names so 57/129 issues get nothing). Includes a PAPER.md draft and a `check_paper.py` reproducibility script. **This is the closest existing work.**
   - **satiricalguru/Gemma-graph-nav** (Apache-2.0, 2026): gold = functions changed by the reference fix; metrics = Recall@1, Recall@5, Acc@5 over 122 official tasks (fastapi, rich, requests, httpx); methods include BM25, **anchored personalized PageRank + BM25 ("APN")**, LLM graph navigation with Gemma 4 E2B/E4B/12B/31B, gated delegation, Agentless-style two-stage, static 1-hop; includes a graph audit of how many gold symbols exist as graph nodes; reports 65/129 issues mention no identifier matching a graph node.
   - Neither reports MRR; neither packages the evaluation as a reusable, documented toolkit with a frozen benchmark release; neither systematically sweeps PPR damping / hybrid fusion weights or reports coverage as a first-class result. These are the gaps we should claim, and we must cite both as concurrent work.

2. **Closest academic benchmark: GREPO (arXiv 2602.13921, Feb 2026).** A class-and-function-level bug-localization benchmark over repository graphs (Directory/File/Class/Function nodes; Contain/Call/Inherit edges resolved with Jedi), 86 Python repos, 47,294 issue-to-PR tasks, Hit@K (K=1,5,10,20) = "mean query recall", designed for GNNs. Differences: GREPO builds its own graphs from commit histories (not SWE-bench, not the competition's released artifacts), targets GNN training at scale, does not report MRR or BM25, and does not analyse coverage of gold symbols in a third-party graph.

3. **Function-level issue-localization evaluations on SWE-bench Lite** (LocAgent/Loc-Bench, SweRank/SweLoc, CoSIL, OrcaLoca, KGCompass, GraphLocator, RepoNavigator, ARISE). All evaluate file/module/function localization with Acc@k or Recall@k, several using code graphs they construct themselves. None uses externally supplied, fixed graphs + embeddings as the object of study, and none frames the contribution as a benchmark/toolkit for *retrieval strategies without an LLM in the loop*.

**Recommended novelty framing:** "the first frozen, symbol-level localization benchmark and open toolkit defined over the competition's released graph + embedding artifacts, with LLM-free retrieval baselines (BM25, kNN, PPR, hybrids), Recall@k and MRR, and a graph-coverage audit quantifying what fraction of gold-patch symbols are reachable at all." Cite aghasalim and satiricalguru as concurrent independent efforts and position ours as the consolidating, reusable resource.

---

## 1. SWE-bench family (task source)

### 1.1 SWE-bench — VERIFIED
- Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., Narasimhan, K. 2024. ICLR 2024. arXiv:2310.06770. https://arxiv.org/abs/2310.06770
- 2,294 real GitHub issue/PR tasks from 12 Python repos; model must edit the codebase so hidden tests pass. Our benchmark reuses the SWE-bench task formulation (problem statement -> gold patch) but re-targets the label to the set of graph symbols the patch touches, and scores retrieval rather than resolution.

### 1.2 SWE-bench Lite — VERIFIED
- Jimenez, C. E., Yang, J., Geng, J. 2024. https://www.swebench.com/lite.html
- 300 test + 23 dev instances (11 repos) filtered to single-file, <=3-hunk patches with >=40-word statements. Many localization papers (Agentless, LocAgent, SweRank, CoSIL) report function-level numbers on Lite; our resource provides the analogous split over the competition's repos/graphs.

### 1.3 SWE-bench Verified — VERIFIED
- OpenAI (with SWE-bench authors). 13 Aug 2024. https://openai.com/index/introducing-swe-bench-verified/
- 500 human-validated instances with difficulty labels (196 "<15 min", 45 ">1 hour"). Relevant as the de-facto resolution benchmark; our gold-symbol extraction procedure is applicable to Verified as well, which we can note as future work.

### 1.4 SWE-Gym — VERIFIED
- Pan, J., Wang, X., Neubig, G., Jaitly, N., Ji, H., Suhr, A., Zhang, Y. 2025. ICML 2025. arXiv:2412.21139. https://arxiv.org/abs/2412.21139
- 2,438 executable Python tasks for training agents and verifiers; +19 pp on SWE-bench Verified/Lite. A training environment, not a localization benchmark; we differ by isolating the localization sub-problem and scoring it without execution.

### 1.5 SWE-smith — VERIFIED
- Yang, J. et al. 2025. NeurIPS 2025 Datasets & Benchmarks (spotlight). arXiv:2504.21798. https://arxiv.org/abs/2504.21798
- Synthesizes 50k test-breaking task instances from 128 repos and trains SWE-agent-LM-32B (40.2% on Verified). Scale-oriented data generation for end-to-end agents; our resource is small, frozen and diagnostic, and could in principle be extended with SWE-smith-style synthetic bugs whose gold symbols are known by construction.

## 2. Issue localization methods and benchmarks

### 2.1 Agentless — VERIFIED
- Xia, C. S., Deng, Y., Dunn, S., Zhang, L. 2024. arXiv:2407.01489 (no venue listed on arXiv page). https://arxiv.org/abs/2407.01489
- Three-phase pipeline (localize, repair, validate). Hierarchical localization: repo tree -> top-N files (LLM ranking fused with text-embedding-3-small chunk retrieval), file skeletons -> related classes/functions, full code -> edit locations. Reports ground-truth-file containment 81.67% and related-element containment 58.33% on SWE-bench Lite (Table 2). Our resource formalizes exactly this intermediate target (which symbols must be found) as a standalone, LLM-free benchmark, and adds graph propagation as a baseline Agentless lacks.

### 2.2 LocAgent + Loc-Bench — VERIFIED
- Chen, Z., Tang, X., Deng, G., Wu, F., Wu, J., Jiang, Z., Prasanna, V., Cohan, A., Wang, X. 2025. ACL 2025 (Long). arXiv:2503.09089. https://arxiv.org/abs/2503.09089
- Parses a repo into a heterogeneous graph (files/classes/functions; import/invoke/inherit edges) and lets an LLM agent traverse it; introduces Loc-Bench (560 issues incl. feature/security/performance) with file/module/function-level Acc@k; 92.7% file-level accuracy with fine-tuned Qwen2.5-Coder-32B. Closest *methodological* ancestor: same graph abstraction, but LocAgent builds its own graph and requires an agent; we evaluate fixed, externally-provided graphs with non-agentic retrievers and report MRR plus coverage.

### 2.3 SweRank / SweLoc — VERIFIED
- Reddy, R. G., Suresh, T., Doo, J., Liu, Y., Nguyen, X. P., Zhou, Y., Yavuz, S., Xiong, C., Ji, H., Joty, S. 2025/2026. ICLR 2026 (camera-ready noted on arXiv v2). arXiv:2505.07849. https://arxiv.org/abs/2505.07849
- Retrieve-and-rerank (SweRankEmbed + SweRankLLM) for file/class/function localization, trained on SweLoc (issues paired with code changes mined from GitHub); SOTA on SWE-bench Lite and LocBench (reported ~82% Acc@10 function-level). Shows embedding retrieval can beat agents; our benchmark supplies a complementary setting where the embedding is fixed (competition .npz) and we ask how far graph propagation and lexical hybrids can push it.

### 2.4 CoSIL — VERIFIED
- Jiang, Z. et al. 2025. arXiv:2503.22424 (later retitled "Issue Localization via LLM-Driven Iterative Code Graph Searching"). https://arxiv.org/abs/2503.22424
- Training-free, two-phase search over a module call graph then a function call graph built on-the-fly by the LLM; Top-1 43% (Lite) / 44.6% (Verified) with Qwen2.5-Coder-32B. Differs: graph constructed during search; ours is precomputed and the propagation (PPR) is algorithmic.

### 2.5 OrcaLoca — VERIFIED
- Yu, Z., Zhang, H., Zhao, Y., Huang, H., Yao, M., Ding, K., Zhao, J. 2025. ICML 2025 (PMLR 267). arXiv:2502.00350. https://arxiv.org/abs/2502.00350
- Agent framework for issue localization with priority scheduling, relevance-scored action decomposition and distance-aware context pruning. Agentic; our baselines are non-agentic and reproducible on CPU.

### 2.6 KGCompass — VERIFIED
- Yang, B., Ren, J., Jin, S., Liu, Y., Liu, F., Le, B., Tian, H. 2025. arXiv:2503.21710. https://arxiv.org/abs/2503.21710
- Repository-aware knowledge graph linking issues, PRs and code entities (files/classes/functions); narrows to ~20 candidate functions then uses graph paths as repair context; 58.3% on Lite at ~$0.20. Uses graph structure for *candidate narrowing*, as we do, but with an LLM in the loop and a richer issue-PR graph; we isolate the code-graph-only signal.

### 2.7 GraphLocator — VERIFIED
- Liu, W., Peng, C., et al. 2025/2026. FSE 2026 (DOI 10.1145/3797079). arXiv:2512.22469. https://arxiv.org/abs/2512.22469
- Causal issue graph (sub-issues tied to code entities) expanded over a repository graph; +19.49% function-level recall across three Python/Java datasets. Confirms function-level recall as the community metric; we add MRR and a coverage denominator.

### 2.8 RepoNavigator ("One Tool Is Enough") — VERIFIED
- Zhang, Z., Duan, Y., et al. 2025 (v6 May 2026). arXiv:2512.20957. https://arxiv.org/abs/2512.20957
- RL-trained agent whose single tool is "jump to definition of a called symbol"; 7B/14B/32B models beat larger baselines on repository-level issue localization. Demonstrates that call-edge following is the key primitive; our PPR-over-call-edges baseline is the LLM-free analogue.

### 2.9 GREPO — VERIFIED (closest academic benchmark; see Section 0)
- Wang, J., Chen, L., Wang, X., Kang, S., Yang, H., Zheng, D., Zhang, M. 2026. arXiv:2602.13921. https://arxiv.org/abs/2602.13921 ; code: github.com/qingpingmo/GREPO
- Class/function-level bug-localization benchmark for GNNs: 86 Python repos, 47,294 tasks, graph nodes Directory/File/Class/Function with Contain/Call/Inherit edges, Hit@K (1/5/10/20) = mean query recall; GNNs beat LocAgent/Agentless/CF-RAG baselines. We differ in using the competition's fixed graphs + embeddings, SWE-bench-style issues, LLM-free IR baselines incl. BM25 and PPR, MRR, and explicit coverage analysis.

### 2.10 ARISE — VERIFIED
- Seddik, S., Seddik, F., Esmaeili, A., Sadatbenis, M., Fard, F. 2026. arXiv:2605.03117. https://arxiv.org/abs/2605.03117
- Multi-granularity program graph (file -> statement, def-use edges) exposed via a 3-tier tool API; mounted on SWE-agent raises Lite resolve 17.3% -> 22.0% and Function R@1 0.43 -> 0.60. A toolset for agents, evaluated through an agent; ours is a benchmark for the graph itself.

### 2.11 Retrieval-Oriented Code Representations in Agentic Bug Localization — VERIFIED
- Caumartin, G., Chen, T.-H., Costa, D. E. 2026. arXiv:2607.11046. https://arxiv.org/abs/2607.11046
- Compares five file representations (paths, raw code, three LLM summaries) under lexical, semantic and LLM retrieval on Long Code Arena and SWE-bench Verified; role-aware summaries +40% Hit@5 over paths; 94% Hit@6 with Agentless. File-level and representation-focused; we are symbol-level and graph-focused.

### 2.12 CoRNStack / CodeRankEmbed / Nomic Embed Code — VERIFIED
- Suresh, T., Reddy, R. G., Xu, Y., Nussbaum, Z., Mulyar, A., Duderstadt, B., Ji, H. 2025. ICLR 2025. arXiv:2412.01007. https://arxiv.org/abs/2412.01007 ; Nomic blog: https://home.nomic.ai/blog/posts/introducing-state-of-the-art-nomic-embed-code
- Consistency-filtered contrastive code corpus; trained retriever + reranker improve function localization for GitHub issues. Nomic Embed Code (7B) is the production follow-up. Represents the "better embedding" axis; our benchmark instead fixes the embedding and asks what graph structure adds.

## 3. Repository graphs for code generation / completion

### 3.1 RepoGraph — VERIFIED
- Ouyang, S., et al. 2025. ICLR 2025. arXiv:2410.14684. https://arxiv.org/abs/2410.14684
- Line-level repository graph plug-in; improves Agentless (GPT-4o) file/function/line localization by ~5.6/3.0/2.4 points. Most direct precedent for "graph helps localization" measured at symbol granularity; ours makes that measurement the artifact.

### 3.2 CodexGraph — VERIFIED
- Liu, X., et al. 2024/2025. NAACL 2025 (Long). arXiv:2408.03910. https://arxiv.org/abs/2408.03910
- LLM agent issues graph-database queries over a code-structure graph; evaluated on CrossCodeEval, SWE-bench, EvoCodeBench. Query-language interface vs. our fixed retrieval algorithms.

### 3.3 CodeRAG (v2 retitled GraphCodeAgent) — VERIFIED
- Li, J., Shi, X., Zhang, K., Li, G., Jin, Z., et al. 2025. arXiv:2504.10046 (v1 "CodeRAG: Supportive Code Retrieval on Bigraph for Real-World Code Generation"; v2 Nov 2025 "GraphCodeAgent: Dual Graph-Guided LLM Agent..."). https://arxiv.org/abs/2504.10046
- Requirement graph + dependency-semantic code graph (import/contain/inherit/call/similarity edges) with agentic multi-hop retrieval; +40.9 Pass@1 on DevEval. Note: a second, different 2025 paper also titled "CodeRAG" exists (arXiv 2509.16112, repository-level completion) - UNVERIFIED beyond the search hit; cite 2504.10046 unless the completion one is intended.

### 3.4 GraphCoder — VERIFIED
- Liu, W., Yu, A., Zan, D., Shen, B., Zhang, W., Zhao, H., Jin, Z., Wang, Q. 2024. arXiv:2406.07003. https://arxiv.org/abs/2406.07003
- Code context graph (control-flow + data/control dependence) with coarse-to-fine retrieval for repository-level completion. Statement-level graph for completion; ours is symbol-level for localization.

### 3.5 RepoCoder — VERIFIED
- Zhang, F., Chen, B., Zhang, Y., Keung, J., Liu, J., Zan, D., Mao, Y., Lou, J.-G., Chen, W. 2023. EMNLP 2023. arXiv:2303.12570. https://arxiv.org/abs/2303.12570
- Iterative retrieval-generation for repository-level completion with the RepoEval benchmark. Establishes similarity-based repo retrieval; no graph, no issue-to-symbol labels.

### 3.6 CodePlan — VERIFIED
- Bairi, R., Sonwane, A., Kanade, A., D C, V., Iyer, A., Parthasarathy, S., Rajamani, S., Ashok, B., Shet, S. 2023. arXiv:2309.12499. https://arxiv.org/abs/2309.12499
- Repository-level editing as planning over an incremental dependency graph with change-impact analysis. Dependency graph used for *propagating edits*; we use it for *propagating relevance* (PPR).

## 4. Agent frameworks (consumers of localization)

### 4.1 SWE-agent — VERIFIED
- Yang, J., Jimenez, C. E., Wettig, A., Lieret, K., Yao, S., Narasimhan, K., Press, O. 2024. NeurIPS 2024. arXiv:2405.15793. https://arxiv.org/abs/2405.15793
- Agent-computer interface (search/view/edit commands) for issue resolution. Our toolkit can serve as an offline pre-localization stage or a tool backend for such agents.

### 4.2 OpenHands — VERIFIED
- Wang, X., Li, B., Song, Y., Xu, F. F., Tang, X., et al. 2025. ICLR 2025. arXiv:2407.16741. https://arxiv.org/abs/2407.16741
- Open platform (sandboxed execution, multi-agent, 15 benchmarks incl. SWE-bench). Same relation as 4.1.

### 4.3 mini-SWE-agent — VERIFIED
- SWE-agent team (Lieret, Yang, Press et al.). 2025. https://github.com/SWE-agent/mini-swe-agent ; PyPI mini-swe-agent
- ~100-line bash-only agent, >74% on SWE-bench Verified; has superseded SWE-agent for development. Shows strong models need little tooling; our benchmark targets the opposite regime (small local Gemma 4, CPU budgets) where cheap pre-localization matters.

## 5. LLM-based fault localization (classical FL lineage)

### 5.1 AutoFL — VERIFIED
- Kang, S., An, G., Yoo, S. 2024. FSE 2024. arXiv:2308.05487. https://arxiv.org/abs/2308.05487
- LLM navigates a repo via function calls to localize and explain faults; +233% method-level acc@1 on 798 Java/Python bugs (Defects4J, BugsInPy). Test-driven, method-level; our labels come from patches on issue text, no tests required at evaluation time.

### 5.2 LLMAO — PARTIALLY VERIFIED (arXiv confirmed; ICSE 2024 venue not shown on arXiv page)
- Yang, A. Z. H., Martins, R., Le Goues, C., Hellendoorn, V. J. 2023/2024. ICSE 2024. arXiv:2310.01726. https://arxiv.org/abs/2310.01726
- Bidirectional adapters on code LLMs for test-free line-level FL (Defects4J). Line-level, supervised; ours is symbol-level, unsupervised retrieval.

### 5.3 FuseFL — VERIFIED
- Widyasari, R., Ang, J. W., Nguyen, T. G., Sharma, N., Lo, D. 2024. SANER 2024. arXiv:2403.10507. https://arxiv.org/abs/2403.10507
- Fuses SBFL scores, test outcomes and code descriptions for explainable FL on student programs (Refactory). Shows value of fusing heterogeneous signals, which motivates our hybrid (lexical + embedding + graph) baselines.

## 6. Code embedding models (the embedding axis)

### 6.1 CodeSage — VERIFIED
- Zhang, D., Ahmad, W., et al. 2024. ICLR 2024. arXiv:2402.01935. https://arxiv.org/abs/2402.01935
- Two-stage (MLM + structure, then contrastive) encoder pretraining; v2 adds Matryoshka dims. Baseline family often compared against; the competition's 256-d vectors are of unspecified provenance, which our coverage/duplicate audit partly characterizes.

### 6.2 CodeT5+ — PARTIALLY VERIFIED (arXiv confirmed; EMNLP 2023 venue not on arXiv page)
- Wang, Y., Le, H., Gotmare, A. D., Bui, N. D. Q., Li, J., Hoi, S. C. H. 2023. arXiv:2305.07922. https://arxiv.org/abs/2305.07922
- Encoder-decoder code LLM family with contrastive / text-code matching objectives; the 110M embedding variant is widely used for text-to-code retrieval.

### 6.3 voyage-code-3 — VERIFIED
- Voyage AI. 4 Dec 2024. https://blog.voyageai.com/2024/12/04/voyage-code-3/
- Proprietary code embedding (2048/1024/512/256 dims, int8/binary); +13.8% vs OpenAI-v3-large, +16.8% vs CodeSage-large on 32 code-retrieval datasets. Note 256-d option matches the competition vector size.

### 6.4 Jina Code Embeddings — VERIFIED
- Kryvosheieva, D., Sturua, S., Gunther, M., Martens, S., Xiao, H. 2025. NeurIPS 2025 Deep Learning for Code workshop. arXiv:2508.21290. https://arxiv.org/abs/2508.21290
- 0.5B/1.5B autoregressive-backbone embedders with last-token pooling; ~79% avg on 25 code-retrieval benchmarks, matching voyage-code-3. Candidate drop-in replacement to test against the competition's fixed embeddings.

### 6.5 Nomic Embed Code — VERIFIED (blog)
- Nomic AI. 2025. https://home.nomic.ai/blog/posts/introducing-state-of-the-art-nomic-embed-code
- 7B open-weights retriever trained on CoRNStack; claims to beat voyage-code-3 on CodeSearchNet. See 2.12.

## 7. Gemma 4 and the competition

### 7.1 Gemma 4 Technical Report — VERIFIED
- Gemma Team (Google DeepMind). 2026. arXiv:2607.02770. https://arxiv.org/abs/2607.02770
- Open-weight natively multimodal models from 2.3B to 31B (dense and MoE, incl. 26B-A4B), thinking mode, long context. The competition mandates gemma-4-31b-it-qat-w4a16-ct (per multiple entrant repos). The abstract does not make coding-agent claims; cite for the model only.

### 7.2 Kaggle competition pages — PARTIALLY VERIFIED (pages exist; body is JS-rendered and could not be scraped)
- Google. "The Gemma 4 Developer Agent Competition." https://www.kaggle.com/competitions/gemma-4-developer-agent
- Google. "The Gemma 4 Developer Agent Paper Track." https://www.kaggle.com/competitions/gemma-4-developer-agent-paper
- From search snippets and entrant READMEs (secondary, verify on Kaggle): main track $65k, paper track $35k ($15k best paper / $10k new resource / $10k new application, per a third-party issue tracker), paper = Kaggle Writeup <= 3,000 words of original unpublished research; paper deadline 12 Nov 2026, final 2 Dec 2026; ~120 hidden issues from private Python repos; per-task budget 60 min / 100 tool calls; metric = resolution rate; released artifacts include `graphs/*.json` and `embeddings/*.npz` (256 files, 256-d float32 per symbol) with tools `get_code_neighbors(node, edge_type, max_neighbors)` and `search_similar_code(query, k)`; hosts listed as Elan Markowitz, Bryan Perozzi et al. (Google DeepMind). Known data quirks reported by entrants: no `async def` nodes in the multigraph; all edges typed `calls`; ~50% zero-byte files from a hard-link ZIP issue (HindsightWise).

### 7.3 Competition entries (concurrent work) — all VERIFIED via GitHub README
- **aghasalim/gemma4-code-graph-localization** (MIT, 2026). https://github.com/aghasalim/gemma4-code-graph-localization . LLM-free study: hit@5 of embed 0.29, graph 0.30, grep 0.41, bm25 0.48, bm25_src 0.64 (95% bootstrap CIs, 129 tasks); audits: no async defs, 21.9% duplicate vectors, all-`calls` edges, 57/129 issues with no symbol-name match. **Closest overlap; cite as concurrent.**
- **satiricalguru/Gemma-graph-nav** (Apache-2.0, 2026). https://github.com/satiricalguru/Gemma-graph-nav . "Should a small LLM navigate your code graph, or should the graph navigate for it?" R@1/R@5/Acc@5 over gold functions from reference fixes, 122 tasks; BM25, anchored personalized PageRank + BM25 (APN), Gemma 4 E2B/E4B/12B/31B navigation, gated delegation, Agentless-style; graph-coverage audit; 65/129 issues name no graph symbol. **Overlaps on PPR + Recall@k; cite as concurrent.**
- **Tarekswd/g-hrr-gemma4** (2026, Tarek Ahmadieh). https://github.com/Tarekswd/g-hrr-gemma4 . G-HRR: hybrid BM25 + dense (RRF) + bounded-BFS AST-graph expansion + hierarchical pruning; 100 SWE-bench instances, 44% resolve vs 18% direct; focal-function hit rate semantic 68% / graph 80% / union 94%; non-monotonic depth effect (1-hop best). Agent-centric; supports our hybrid hypothesis.
- **nazarcoder123 "GraphSWE-Gemma"** (2026). https://github.com/nazarcoder123/Google---The-Gemma-4-Developer-Agent-Paper-Track-Kaggle- . ADK agent navigating the NetworkX multigraph + 256-d embeddings; ego caller/callee subgraphs around defect sites ("93.1% token compression"); AST-parse patch verification; evaluation on mock tasks only, no Recall@k. Not a benchmark.
- **yosh95/g4-developer-agent-paper** (Apache-2.0, 2026). https://github.com/yosh95/g4-developer-agent-paper . "Every Tool Call Is a Token You Cannot Buy Back": cost accounting for Gemma 4 31B INT4 on a 4-core/3.7 GB CPU host over an offline SWE-bench-shaped set (58 hidden / 55 traced / 18 paired); reports gold-file top-5 for 14/18 and top-1 for 9/18 tasks as a side metric. No code graphs. Cite for the CPU-budget motivation.
- **HindsightWise/gemma-4-developer-agent** (2026). https://github.com/HindsightWise/gemma-4-developer-agent . Evaluation harness + dual-container grader; documents the missing-`async def` trap and zero-byte graph files; no localization metrics.

### 7.4 Not found / UNVERIFIED
- No 2026 arXiv paper specifically on "Gemma 4 coding agents" or on graph tools inside Google ADK was found; the only ADK references are entrant repos. If the paper needs an ADK citation, use the ADK docs/GitHub (not fetched here - UNVERIFIED).
- No dedicated 2025 survey of LLM-based fault localization was found; the nearest are "Large Language Models for Fault Localization: An Empirical Study" (arXiv 2510.20521, Oct 2025, statement-level on HumanEval-Java/Defects4J - seen only in a search snippet, PARTIALLY VERIFIED) and the broader LLM4SE survey (arXiv 2312.15223).

---

## 8. Suggested paragraph structure for the paper's Related Work (approx. 450 words)

1. *Task source* (SWE-bench, Lite, Verified; SWE-Gym/SWE-smith for scale).
2. *Localization as a sub-problem* (Agentless hierarchical localization; LocAgent/Loc-Bench; SweRank/SweLoc; CoSIL, OrcaLoca, KGCompass, GraphLocator, RepoNavigator) - all report file/function Acc@k or Recall@k; most build their own graph.
3. *Graph representations of repositories* (RepoGraph, CodexGraph, CodeRAG/GraphCodeAgent, GraphCoder, RepoCoder, CodePlan, ARISE).
4. *Benchmarks for graph-based localization* (GREPO) - closest academic artifact; contrast on graph provenance, task source, metrics and coverage audit.
5. *Embeddings* (CodeSage, CodeT5+, voyage-code-3, Jina, CoRNStack/Nomic).
6. *Classical LLM-FL lineage* (AutoFL, LLMAO, FuseFL) - test-dependent or line-level.
7. *Concurrent competition work* (aghasalim, satiricalguru, G-HRR, GraphSWE-Gemma, yosh95) - state clearly what each measured and what our consolidated resource adds (frozen release, MRR, PPR/hybrid sweeps, coverage as a headline number, toolkit API).

---

## 9. BibTeX (verified entries only)

```bibtex
@inproceedings{jimenez2024swebench,
  title     = {{SWE-bench}: Can Language Models Resolve Real-World {GitHub} Issues?},
  author    = {Jimenez, Carlos E. and Yang, John and Wettig, Alexander and Yao, Shunyu and Pei, Kexin and Press, Ofir and Narasimhan, Karthik},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024},
  eprint    = {2310.06770},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2310.06770}
}

@misc{jimenez2024swebenchlite,
  title  = {{SWE-bench Lite}: A Curated Subset of {SWE-bench} for Faster, More Cost-Effective Evaluation},
  author = {Jimenez, Carlos E. and Yang, John and Geng, Jiayi},
  year   = {2024},
  howpublished = {\url{https://www.swebench.com/lite.html}}
}

@misc{openai2024swebenchverified,
  title  = {Introducing {SWE-bench Verified}},
  author = {{OpenAI}},
  year   = {2024},
  month  = aug,
  howpublished = {\url{https://openai.com/index/introducing-swe-bench-verified/}}
}

@inproceedings{pan2025swegym,
  title     = {Training Software Engineering Agents and Verifiers with {SWE-Gym}},
  author    = {Pan, Jiayi and Wang, Xingyao and Neubig, Graham and Jaitly, Navdeep and Ji, Heng and Suhr, Alane and Zhang, Yizhe},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2025},
  eprint    = {2412.21139},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2412.21139}
}

@inproceedings{yang2025swesmith,
  title     = {{SWE-smith}: Scaling Data for Software Engineering Agents},
  author    = {Yang, John and others},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track},
  year      = {2025},
  eprint    = {2504.21798},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2504.21798}
}

@article{xia2024agentless,
  title   = {Agentless: Demystifying {LLM}-based Software Engineering Agents},
  author  = {Xia, Chunqiu Steven and Deng, Yinlin and Dunn, Soren and Zhang, Lingming},
  journal = {arXiv preprint arXiv:2407.01489},
  year    = {2024},
  url     = {https://arxiv.org/abs/2407.01489}
}

@inproceedings{chen2025locagent,
  title     = {{LocAgent}: Graph-Guided {LLM} Agents for Code Localization},
  author    = {Chen, Zhaoling and Tang, Xiangru and Deng, Gangda and Wu, Fang and Wu, Jialong and Jiang, Zhiwei and Prasanna, Viktor and Cohan, Arman and Wang, Xingyao},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2025},
  eprint    = {2503.09089},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2503.09089}
}

@inproceedings{reddy2026swerank,
  title     = {{SweRank}: Software Issue Localization with Code Ranking},
  author    = {Reddy, Revanth Gangi and Suresh, Tarun and Doo, JaeHyeok and Liu, Ye and Nguyen, Xuan Phi and Zhou, Yingbo and Yavuz, Semih and Xiong, Caiming and Ji, Heng and Joty, Shafiq},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  eprint    = {2505.07849},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2505.07849}
}

@article{jiang2025cosil,
  title   = {{CoSIL}: Software Issue Localization via {LLM}-Driven Code Repository Graph Searching},
  author  = {Jiang, Zhonghao and others},
  journal = {arXiv preprint arXiv:2503.22424},
  year    = {2025},
  url     = {https://arxiv.org/abs/2503.22424}
}

@inproceedings{yu2025orcaloca,
  title     = {{OrcaLoca}: An {LLM} Agent Framework for Software Issue Localization},
  author    = {Yu, Zhongming and Zhang, Hejia and Zhao, Yujie and Huang, Hanxian and Yao, Matrix and Ding, Ke and Zhao, Jishen},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML), PMLR 267},
  pages     = {73416--73436},
  year      = {2025},
  eprint    = {2502.00350},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2502.00350}
}

@article{yang2025kgcompass,
  title   = {Enhancing Repository-Level Software Repair via Repository-Aware Knowledge Graphs},
  author  = {Yang, Boyang and Ren, Jiadong and Jin, Shunfu and Liu, Yang and Liu, Feng and Le, Bach and Tian, Haoye},
  journal = {arXiv preprint arXiv:2503.21710},
  year    = {2025},
  url     = {https://arxiv.org/abs/2503.21710}
}

@inproceedings{liu2026graphlocator,
  title     = {{GraphLocator}: Graph-guided Causal Reasoning for Issue Localization},
  author    = {Liu, Wei and Peng, Chao and others},
  booktitle = {Proceedings of the ACM International Conference on the Foundations of Software Engineering (FSE)},
  year      = {2026},
  doi       = {10.1145/3797079},
  eprint    = {2512.22469},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2512.22469}
}

@article{zhang2025reponavigator,
  title   = {One Tool Is Enough: Reinforcement Learning for Repository-Level {LLM} Agents},
  author  = {Zhang, Zhaoxi and Duan, Yitong and Zhang, Yanzhi and Xu, Yiming and Wang, Zhixiang and Liang, Kun and Li, Weikang and Liang, Jiahui and Xia, Deguo and Huang, Jizhou and He, Jiyan and Wu, Yunfang},
  journal = {arXiv preprint arXiv:2512.20957},
  year    = {2025},
  url     = {https://arxiv.org/abs/2512.20957}
}

@article{wang2026grepo,
  title   = {{GREPO}: A Benchmark for Graph Neural Networks on Repository-Level Bug Localization},
  author  = {Wang, Juntong and Chen, Libin and Wang, Xiyuan and Kang, Shijia and Yang, Haotong and Zheng, Da and Zhang, Muhan},
  journal = {arXiv preprint arXiv:2602.13921},
  year    = {2026},
  url     = {https://arxiv.org/abs/2602.13921}
}

@article{seddik2026arise,
  title   = {{ARISE}: A Repository-level Graph Representation and Toolset for Agentic Program Repair and Fault Localization},
  author  = {Seddik, Shahd and Seddik, Fahd and Esmaeili, Amirrezza and Sadatbenis, Mahdieh and Fard, Fatemeh},
  journal = {arXiv preprint arXiv:2605.03117},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.03117}
}

@article{caumartin2026retrievaloriented,
  title   = {Retrieval-Oriented Code Representations in Agentic Bug Localization},
  author  = {Caumartin, Genevieve and Chen, Tse-Hsun and Costa, Diego Elias},
  journal = {arXiv preprint arXiv:2607.11046},
  year    = {2026},
  url     = {https://arxiv.org/abs/2607.11046}
}

@inproceedings{suresh2025cornstack,
  title     = {{CoRNStack}: High-Quality Contrastive Data for Better Code Retrieval and Reranking},
  author    = {Suresh, Tarun and Reddy, Revanth Gangi and Xu, Yifei and Nussbaum, Zach and Mulyar, Andriy and Duderstadt, Brandon and Ji, Heng},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  eprint    = {2412.01007},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2412.01007}
}

@inproceedings{ouyang2025repograph,
  title     = {{RepoGraph}: Enhancing {AI} Software Engineering with Repository-level Code Graph},
  author    = {Ouyang, Siru and others},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  eprint    = {2410.14684},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2410.14684}
}

@inproceedings{liu2025codexgraph,
  title     = {{CodexGraph}: Bridging Large Language Models and Code Repositories via Code Graph Databases},
  author    = {Liu, Xiangyan and others},
  booktitle = {Proceedings of the 2025 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), Long Papers},
  year      = {2025},
  eprint    = {2408.03910},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2408.03910}
}

@article{li2025coderag,
  title   = {{CodeRAG}: Supportive Code Retrieval on Bigraph for Real-World Code Generation},
  author  = {Li, Jia and Shi, Xianjie and Zhang, Kechi and Li, Ge and Jin, Zhi and Li, Lei and Zhang, Huangzhao and Li, Jia and Liu, Fang and Zhang, Yuwei and Tao, Zhengwei and Dong, Yihong and Zhu, Yuqi and Tao, Chongyang},
  journal = {arXiv preprint arXiv:2504.10046},
  note    = {v2 retitled ``GraphCodeAgent: Dual Graph-Guided LLM Agent for Retrieval-Augmented Repo-Level Code Generation''},
  year    = {2025},
  url     = {https://arxiv.org/abs/2504.10046}
}

@article{liu2024graphcoder,
  title   = {{GraphCoder}: Enhancing Repository-Level Code Completion via Code Context Graph-based Retrieval and Language Model},
  author  = {Liu, Wei and Yu, Ailun and Zan, Daoguang and Shen, Bo and Zhang, Wei and Zhao, Haiyan and Jin, Zhi and Wang, Qianxiang},
  journal = {arXiv preprint arXiv:2406.07003},
  year    = {2024},
  url     = {https://arxiv.org/abs/2406.07003}
}

@inproceedings{zhang2023repocoder,
  title     = {{RepoCoder}: Repository-Level Code Completion Through Iterative Retrieval and Generation},
  author    = {Zhang, Fengji and Chen, Bei and Zhang, Yue and Keung, Jacky and Liu, Jin and Zan, Daoguang and Mao, Yi and Lou, Jian-Guang and Chen, Weizhu},
  booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages     = {2471--2484},
  year      = {2023},
  eprint    = {2303.12570},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2303.12570}
}

@article{bairi2023codeplan,
  title   = {{CodePlan}: Repository-level Coding using {LLMs} and Planning},
  author  = {Bairi, Ramakrishna and Sonwane, Atharv and Kanade, Aditya and D C, Vageesh and Iyer, Arun and Parthasarathy, Suresh and Rajamani, Sriram and Ashok, B. and Shet, Shashank},
  journal = {arXiv preprint arXiv:2309.12499},
  year    = {2023},
  url     = {https://arxiv.org/abs/2309.12499}
}

@inproceedings{yang2024sweagent,
  title     = {{SWE-agent}: Agent-Computer Interfaces Enable Automated Software Engineering},
  author    = {Yang, John and Jimenez, Carlos E. and Wettig, Alexander and Lieret, Kilian and Yao, Shunyu and Narasimhan, Karthik and Press, Ofir},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024},
  eprint    = {2405.15793},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2405.15793}
}

@inproceedings{wang2025openhands,
  title     = {{OpenHands}: An Open Platform for {AI} Software Developers as Generalist Agents},
  author    = {Wang, Xingyao and Li, Boxuan and Song, Yufan and Xu, Frank F. and Tang, Xiangru and others},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  eprint    = {2407.16741},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2407.16741}
}

@misc{sweagent2025miniswe,
  title  = {mini-{SWE}-agent: The 100-line {AI} agent that solves {GitHub} issues},
  author = {{SWE-agent team}},
  year   = {2025},
  howpublished = {\url{https://github.com/SWE-agent/mini-swe-agent}}
}

@inproceedings{kang2024autofl,
  title     = {A Quantitative and Qualitative Evaluation of {LLM}-Based Explainable Fault Localization},
  author    = {Kang, Sungmin and An, Gabin and Yoo, Shin},
  booktitle = {Proceedings of the ACM International Conference on the Foundations of Software Engineering (FSE)},
  year      = {2024},
  eprint    = {2308.05487},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2308.05487}
}

@inproceedings{yang2024llmao,
  title     = {Large Language Models for Test-Free Fault Localization},
  author    = {Yang, Aidan Z. H. and Martins, Ruben and Le Goues, Claire and Hellendoorn, Vincent J.},
  booktitle = {Proceedings of the 46th IEEE/ACM International Conference on Software Engineering (ICSE)},
  year      = {2024},
  eprint    = {2310.01726},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2310.01726}
}

@inproceedings{widyasari2024fusefl,
  title     = {Demystifying Faulty Code with {LLM}: Step-by-Step Reasoning for Explainable Fault Localization},
  author    = {Widyasari, Ratnadira and Ang, Jia Wei and Nguyen, Truong Giang and Sharma, Neil and Lo, David},
  booktitle = {IEEE International Conference on Software Analysis, Evolution and Reengineering (SANER)},
  year      = {2024},
  eprint    = {2403.10507},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2403.10507}
}

@inproceedings{zhang2024codesage,
  title     = {Code Representation Learning at Scale},
  author    = {Zhang, Dejiao and Ahmad, Wasi and others},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024},
  eprint    = {2402.01935},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2402.01935}
}

@inproceedings{wang2023codet5plus,
  title     = {{CodeT5+}: Open Code Large Language Models for Code Understanding and Generation},
  author    = {Wang, Yue and Le, Hung and Gotmare, Akhilesh Deepak and Bui, Nghi D. Q. and Li, Junnan and Hoi, Steven C. H.},
  booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2023},
  eprint    = {2305.07922},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2305.07922}
}

@misc{voyage2024code3,
  title  = {voyage-code-3: More Accurate Code Retrieval with Lower Dimensional, Quantized Embeddings},
  author = {{Voyage AI}},
  year   = {2024},
  month  = dec,
  howpublished = {\url{https://blog.voyageai.com/2024/12/04/voyage-code-3/}}
}

@inproceedings{kryvosheieva2025jinacode,
  title     = {Efficient Code Embeddings from Code Generation Models},
  author    = {Kryvosheieva, Daria and Sturua, Saba and G{\"u}nther, Michael and Martens, Scott and Xiao, Han},
  booktitle = {NeurIPS 2025 Workshop on Deep Learning for Code},
  year      = {2025},
  eprint    = {2508.21290},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2508.21290}
}

@misc{nomic2025embedcode,
  title  = {Nomic Embed Code: A State-of-the-Art Code Retriever},
  author = {{Nomic AI}},
  year   = {2025},
  howpublished = {\url{https://home.nomic.ai/blog/posts/introducing-state-of-the-art-nomic-embed-code}}
}

@article{gemmateam2026gemma4,
  title   = {Gemma 4 Technical Report},
  author  = {{Gemma Team}},
  journal = {arXiv preprint arXiv:2607.02770},
  year    = {2026},
  url     = {https://arxiv.org/abs/2607.02770}
}

@misc{google2026gemma4agentcomp,
  title  = {Google -- The {Gemma 4} Developer Agent Competition},
  author = {{Google}},
  year   = {2026},
  howpublished = {Kaggle, \url{https://www.kaggle.com/competitions/gemma-4-developer-agent}}
}

@misc{google2026gemma4papertrack,
  title  = {Google -- The {Gemma 4} Developer Agent Paper Track},
  author = {{Google}},
  year   = {2026},
  howpublished = {Kaggle, \url{https://www.kaggle.com/competitions/gemma-4-developer-agent-paper}}
}

@misc{aghasalim2026codegraphloc,
  title  = {gemma4-code-graph-localization: How Often the {Gemma 4} Competition's Code Graph and Embeddings Find the Code a Fix Edits},
  author = {{aghasalim}},
  year   = {2026},
  howpublished = {GitHub, \url{https://github.com/aghasalim/gemma4-code-graph-localization}}
}

@misc{satiricalguru2026graphnav,
  title  = {Gemma-graph-nav: Should a Small {LLM} Navigate Your Code Graph, or Should the Graph Navigate for It?},
  author = {{satiricalguru}},
  year   = {2026},
  howpublished = {GitHub, \url{https://github.com/satiricalguru/Gemma-graph-nav}}
}

@misc{ahmadieh2026ghrr,
  title  = {Graph-Guided Hierarchical Repository Reasoning ({G-HRR}) for Local Software Engineering Agents},
  author = {Ahmadieh, Tarek},
  year   = {2026},
  howpublished = {GitHub, \url{https://github.com/Tarekswd/g-hrr-gemma4}}
}

@misc{nazarcoder2026graphswe,
  title  = {{GraphSWE-Gemma}: Hierarchical Code-Graph Reasoning and Test-Driven Self-Correction for Local Autonomous Software Engineering Agents},
  author = {{nazarcoder123}},
  year   = {2026},
  howpublished = {GitHub, \url{https://github.com/nazarcoder123/Google---The-Gemma-4-Developer-Agent-Paper-Track-Kaggle-}}
}

@misc{yosh2026toolcall,
  title  = {Every Tool Call Is a Token You Cannot Buy Back},
  author = {{yosh95}},
  year   = {2026},
  howpublished = {GitHub, \url{https://github.com/yosh95/g4-developer-agent-paper}}
}
```
