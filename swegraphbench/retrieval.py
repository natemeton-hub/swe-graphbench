"""Symbol-level localization baselines over a competition graph + embedding archive.

All methods return a ranked list of node ids for a problem statement. None of them
call a language model; they are the cheap, offline strategies an agent on consumer
hardware could run inside its tool budget.

  bm25_text     BM25 over node source text (identifier-split tokens)
  bm25_name     BM25 over dotted node names only
  symbol_seed   identifiers mentioned in the issue that resolve to node names (what
                `search_similar_code` effectively supports), ranked by mention count
  emb_expand    symbol_seed, then expand by cosine similarity in the released embeddings
  ppr           personalized PageRank on the call graph from the lexical (bm25_text) seeds
  hybrid        reciprocal-rank fusion of bm25_text + emb_expand + ppr
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np
from rank_bm25 import BM25Okapi

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "for", "on", "with", "this", "that", "be",
         "as", "are", "by", "if", "not", "at", "from", "but", "we", "i", "you", "can", "when", "should", "would",
         "def", "class", "self", "return", "none", "true", "false", "import", "pass", "python", "py"}


def tokenize(text: str) -> list[str]:
    toks = []
    for ident in _IDENT.findall(text):
        parts = [p for p in re.split(r"_+", ident) if p]
        for p in parts:
            for sub in _CAMEL.split(p):
                s = sub.lower()
                if len(s) > 1 and s not in _STOP:
                    toks.append(s)
    return toks


def issue_identifiers(problem: str) -> Counter:
    """Candidate code identifiers in an issue: backticked spans, dotted paths, snake/Camel names."""
    c: Counter = Counter()
    for m in re.findall(r"`([^`\n]{1,80})`", problem):
        for ident in _IDENT.findall(m):
            c[ident] += 2
    for m in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b", problem):
        c[m] += 2
        c[m.split(".")[-1]] += 1
    for ident in _IDENT.findall(problem):
        if "_" in ident or re.search(r"[a-z][A-Z]", ident):
            c[ident] += 1
    return c


def is_test_node(node_id: str) -> bool:
    parts = node_id.split(".")
    return parts[0] in ("tests", "test") or any(p.startswith("test_") or p.endswith("_test") for p in parts)


class GraphIndex:
    def __init__(self, graph: dict, emb_keys: list[str] | None = None, emb: np.ndarray | None = None,
                 exclude_tests: bool = False, directed: bool = False):
        """`directed=True` keeps edge direction for PPR (only meaningful for rebuilt graphs).
        `exclude_tests=True` drops test-suite nodes from the candidate set (an agent fixing library
        code can apply the same filter for free)."""
        nodes_raw = [n for n in graph["nodes"] if n.get("kind", "function") != "module"]
        if exclude_tests:
            nodes_raw = [n for n in nodes_raw if not is_test_node(n["id"])]
        self.nodes = [n["id"] for n in nodes_raw]
        self.text = {n["id"]: n["text"] for n in nodes_raw}
        keep = set(self.nodes)
        self.directed = directed
        self.pos = {n: i for i, n in enumerate(self.nodes)}
        self.adj: dict[str, set[str]] = defaultdict(set)
        for e in graph["edges"]:
            if e.get("type", "calls") != "calls":
                continue
            s, t = e["source"], e["target"]
            if s != t and s in keep and t in keep:
                if directed == "reverse":      # propagate from callee to its callers
                    self.adj[t].add(s)
                elif directed:                 # caller -> callee
                    self.adj[s].add(t)
                else:                          # undirected
                    self.adj[s].add(t)
                    self.adj[t].add(s)
        self._bm25_text = BM25Okapi([tokenize(self.text[n]) + tokenize(n) for n in self.nodes])
        self._bm25_name = BM25Okapi([tokenize(n) for n in self.nodes])
        # suffix index for short-name resolution (mirrors harness resolve_node_name tiers)
        self.suffix: dict[str, list[str]] = defaultdict(list)
        for n in self.nodes:
            parts = n.split(".")
            for i in range(len(parts)):
                self.suffix[".".join(parts[i:])].append(n)
        self.lower = defaultdict(list)
        for n in self.nodes:
            self.lower[n.lower()].append(n)
        if emb is not None and emb_keys is not None:
            self.emb_pos = {k: i for i, k in enumerate(emb_keys)}
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            self.emb = emb / np.clip(norms, 1e-8, None)
        else:
            self.emb_pos, self.emb = None, None

    # ---- ranking helpers ----
    def _rank_scores(self, scores: np.ndarray, k: int) -> list[str]:
        idx = np.argsort(-scores)[:k]
        return [self.nodes[i] for i in idx if scores[i] > 0]

    def bm25_text(self, problem: str, k: int = 50) -> list[str]:
        cache = getattr(self, "_bm25_cache", None)
        if cache is None:
            cache = self._bm25_cache = {}
        if problem not in cache:
            cache[problem] = self._bm25_text.get_scores(tokenize(problem))
        return self._rank_scores(cache[problem], k)

    def bm25_name(self, problem: str, k: int = 50) -> list[str]:
        return self._rank_scores(self._bm25_name.get_scores(tokenize(problem)), k)

    def resolve(self, name: str) -> list[str]:
        if name in self.pos:
            return [name]
        if name in self.suffix:
            return self.suffix[name]
        return self.lower.get(name.lower(), [])

    def symbol_seed(self, problem: str, k: int = 50) -> list[str]:
        sc: Counter = Counter()
        for ident, w in issue_identifiers(problem).items():
            for n in self.resolve(ident):
                sc[n] += w
        return [n for n, _ in sc.most_common(k)]

    def emb_expand(self, problem: str, k: int = 50, seeds: int = 5) -> list[str]:
        if self.emb is None:
            return self.symbol_seed(problem, k)
        seed = [s for s in self.symbol_seed(problem, seeds) if s in self.emb_pos]
        if not seed:
            return []
        q = self.emb[[self.emb_pos[s] for s in seed]].mean(axis=0)
        sims = self.emb @ q
        order = np.argsort(-sims)
        inv = {i: kk for kk, i in self.emb_pos.items()}
        out = list(seed)
        for i in order:
            n = inv[int(i)]
            if n not in out:
                out.append(n)
            if len(out) >= k:
                break
        return out

    def _transition(self):
        """Column-stochastic sparse transition matrix T[m, n] = 1/outdeg(n) for edge n->m (cached)."""
        if getattr(self, "_T", None) is None:
            from scipy.sparse import csr_matrix
            rows, cols, vals = [], [], []
            for n, nb in self.adj.items():
                if nb:
                    share = 1.0 / len(nb)
                    for m in nb:
                        rows.append(self.pos[m]); cols.append(self.pos[n]); vals.append(share)
            N = len(self.nodes)
            self._T = csr_matrix((vals, (rows, cols)), shape=(N, N))
        return self._T

    def ppr(self, problem: str, k: int = 50, seeds: int = 10, alpha: float = 0.15, iters: int = 30) -> list[str]:
        seed = self.bm25_text(problem, seeds)
        if not seed:
            return []
        p0 = np.zeros(len(self.nodes))
        for s in seed:
            p0[self.pos[s]] = 1.0
        p0 /= p0.sum()
        p = p0.copy()
        T = self._transition()
        for _ in range(iters):
            nxt = T @ p
            lost = 1.0 - nxt.sum()  # mass at dangling (out-degree 0) nodes returns to the seeds
            p = (1 - alpha) * nxt + (alpha + (1 - alpha) * lost) * p0
        return self._rank_scores(p, k)

    def hybrid(self, problem: str, k: int = 50) -> list[str]:
        lists = [self.bm25_text(problem, k), self.emb_expand(problem, k), self.ppr(problem, k)]
        return rrf(lists, k)


def rrf(lists: list[list[str]], k: int, c: int = 60) -> list[str]:
    sc: dict[str, float] = defaultdict(float)
    for lst in lists:
        for r, n in enumerate(lst):
            sc[n] += 1.0 / (c + r + 1)
    return [n for n, _ in sorted(sc.items(), key=lambda x: -x[1])[:k]]


# ---- metrics ----
def recall_at(ranked: list[str], gold: set[str], k: int) -> float:
    return len(set(ranked[:k]) & gold) / len(gold) if gold else math.nan


def hit_at(ranked: list[str], gold: set[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & gold else 0.0


def mrr(ranked: list[str], gold: set[str]) -> float:
    for i, n in enumerate(ranked):
        if n in gold:
            return 1.0 / (i + 1)
    return 0.0
