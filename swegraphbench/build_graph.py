"""Pure-stdlib code-graph builder producing the competition's JSON schema (plus extras).

Design goals
  * zero third-party dependencies (runs inside the air-gapped sandbox as a skill script)
  * every def/class is a node, including `async def` and nested functions
  * directed `calls` edges with a recorded resolution strategy
  * `contains`, `inherits`, `imports` edges
  * file / line attributes on every node
  * node ids follow the released convention (`src/` and `docs_src/` prefixes dropped)

Usage:  python -m swegraphbench.build_graph <repo_root> <out.json> [--name NAME]
"""
from __future__ import annotations

import ast
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

STRIP_PREFIXES = ("src", "docs_src")
SKIP_DIRS = {".git", "__pycache__", "build", "dist", ".venv", "venv", "node_modules", "site-packages", ".tox", ".eggs"}
MAX_TEXT = 20000


def module_name(rel: str) -> str:
    parts = list(Path(rel).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    if len(parts) > 1 and parts[0] in STRIP_PREFIXES:
        parts = parts[1:]
    return ".".join(parts)


def iter_py_files(root: Path):
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d not in SKIP_DIRS and not d.startswith("."))
        for fn in sorted(fns):
            if fn.endswith(".py"):
                yield Path(dp) / fn


class _Collector(ast.NodeVisitor):
    """First pass: definitions, import tables, class bases, raw call sites per scope."""

    def __init__(self, mod: str, rel: str, src: str, pkg: str):
        self.mod, self.rel, self.src, self.pkg = mod, rel, src, pkg
        self.lines = src.splitlines()
        self.nodes: dict[str, dict] = {}
        self.contains: list[tuple[str, str]] = []
        self.imports: dict[str, str] = {}      # local alias -> fully qualified target
        self.module_imports: set[str] = set()  # imported module ids
        self.bases: dict[str, list[ast.expr]] = {}
        self.calls: list[tuple[str, ast.expr, str | None]] = []  # (scope_id, callee expr, enclosing class)
        self.stack: list[tuple[str, str]] = []  # (kind, qualname)

    # ---- helpers ----
    def _qual(self, name: str) -> str:
        return ".".join([q for _, q in self.stack] + [name])

    def _id(self, qual: str) -> str:
        return f"{self.mod}.{qual}" if qual else self.mod

    def _cur_id(self) -> str:
        return self._id(".".join(q for _, q in self.stack))

    def _cur_class(self) -> str | None:
        for kind, q in reversed(self.stack):
            if kind == "class":
                return self._id(".".join(qq for kk, qq in self.stack[: self.stack.index((kind, q)) + 1]))
        return None

    def _segment(self, node: ast.AST) -> str:
        start = node.lineno
        if getattr(node, "decorator_list", None):
            start = min(start, min(d.lineno for d in node.decorator_list))
        return "\n".join(self.lines[start - 1 : node.end_lineno])[:MAX_TEXT]

    def _resolve_relative(self, module: str | None, level: int) -> str:
        if level == 0:
            return module or ""
        base = self.pkg.split(".")
        base = base[: len(base) - (level - 1)] if level - 1 <= len(base) else []
        return ".".join([p for p in base if p] + ([module] if module else []))

    # ---- visitors ----
    def visit_Import(self, node: ast.Import):
        for a in node.names:
            self.imports[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
            self.module_imports.add(a.name)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        target = self._resolve_relative(node.module, node.level)
        self.module_imports.add(target)
        for a in node.names:
            if a.name == "*":
                continue
            self.imports[a.asname or a.name] = f"{target}.{a.name}" if target else a.name

    def _def(self, node, kind: str):
        qual = self._qual(node.name)
        nid = self._id(qual)
        parent = self._cur_id()
        self.nodes[nid] = {
            "id": nid, "name": nid, "kind": kind, "file": self.rel,
            "lineno": node.lineno, "end_lineno": node.end_lineno, "text": self._segment(node),
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            "parent": parent,
            "docstring": (ast.get_docstring(node) or "").split("\n")[0][:200],
        }
        self.contains.append((parent, nid))
        if kind == "class":
            self.bases[nid] = list(node.bases)
        self.stack.append(("class" if kind == "class" else "function", node.name))
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node):
        self._def(node, "function")

    def visit_AsyncFunctionDef(self, node):
        self._def(node, "function")

    def visit_ClassDef(self, node):
        self._def(node, "class")

    def visit_Call(self, node: ast.Call):
        self.calls.append((self._cur_id(), node.func, self._cur_class()))
        self.generic_visit(node)


def _attr_chain(e: ast.expr) -> list[str] | None:
    parts = []
    while isinstance(e, ast.Attribute):
        parts.append(e.attr)
        e = e.value
    if isinstance(e, ast.Name):
        parts.append(e.id)
        return parts[::-1]
    if isinstance(e, ast.Call):  # Foo().bar()  -> treat receiver as call result of Foo
        inner = _attr_chain(e.func)
        return inner + parts[::-1] if inner else None
    return None


def build(root: Path, name: str, include_tests: bool = True) -> dict:
    root = Path(root)
    cols: dict[str, _Collector] = {}
    for f in iter_py_files(root):
        rel = f.relative_to(root).as_posix()
        if not include_tests and ("/tests/" in f"/{rel}" or Path(rel).name.startswith("test_")):
            continue
        try:
            src = f.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src)
        except (SyntaxError, ValueError):
            continue
        mod = module_name(rel)
        pkg = mod if Path(rel).name == "__init__.py" else mod.rsplit(".", 1)[0] if "." in mod else ""
        c = _Collector(mod, rel, src, pkg)
        c.nodes[mod] = {"id": mod, "name": mod, "kind": "module", "file": rel, "lineno": 1,
                        "end_lineno": len(c.lines), "text": "", "is_async": False, "parent": None,
                        "docstring": (ast.get_docstring(tree) or "").split("\n")[0][:200]}
        c.visit(tree)
        cols[mod] = c

    all_nodes: dict[str, dict] = {}
    for c in cols.values():
        all_nodes.update(c.nodes)
    # suffix index: last component -> ids  (for unique-suffix fallback)
    by_last: dict[str, list[str]] = defaultdict(list)
    by_tail2: dict[str, list[str]] = defaultdict(list)
    for nid, n in all_nodes.items():
        if n["kind"] != "module":
            parts = nid.split(".")
            by_last[parts[-1]].append(nid)
            if len(parts) >= 2:
                by_tail2[".".join(parts[-2:])].append(nid)

    edges: list[dict] = []
    seen: set[tuple] = set()

    def add(src: str, dst: str, etype: str, **attrs):
        if src == dst and etype == "calls":
            attrs["self_loop"] = True
        key = (src, dst, etype)
        if key in seen or src not in all_nodes or dst not in all_nodes:
            return
        seen.add(key)
        edges.append({"type": etype, "source": src, "target": dst, "key": 0, **attrs})

    # contains / imports
    for c in cols.values():
        for p, ch in c.contains:
            add(p, ch, "contains")
        for m in c.module_imports:
            if m in all_nodes:
                add(c.mod, m, "imports")

    # inheritance (resolve bases through the import table / module scope)
    def resolve_name_in_module(c: _Collector, chain: list[str]) -> str | None:
        head = chain[0]
        cand = None
        if head in c.imports:
            cand = c.imports[head]
        elif f"{c.mod}.{head}" in all_nodes:
            cand = f"{c.mod}.{head}"
        elif head in cols:  # bare module name
            cand = head
        if cand is None:
            return None
        full = ".".join([cand] + chain[1:])
        if full in all_nodes:
            return full
        # imported module alias: a.b imported, then a.b.c.d
        if cand in all_nodes and all_nodes[cand]["kind"] == "module":
            return full if full in all_nodes else None
        return None

    class_bases: dict[str, list[str]] = defaultdict(list)
    for c in cols.values():
        for cls, bases in c.bases.items():
            for b in bases:
                chain = _attr_chain(b)
                if not chain:
                    continue
                tgt = resolve_name_in_module(c, chain)
                if tgt:
                    add(cls, tgt, "inherits")
                    class_bases[cls].append(tgt)

    def class_member(cls: str, attr: str, depth: int = 0) -> str | None:
        if f"{cls}.{attr}" in all_nodes:
            return f"{cls}.{attr}"
        if depth < 6:
            for b in class_bases.get(cls, []):
                r = class_member(b, attr, depth + 1)
                if r:
                    return r
        return None

    # calls
    stats = defaultdict(int)
    for c in cols.values():
        for scope, func, cur_cls in c.calls:
            chain = _attr_chain(func)
            if not chain:
                stats["unresolvable_expr"] += 1
                continue
            tgt, how = None, None
            if chain[0] in ("self", "cls") and cur_cls and len(chain) >= 2:
                tgt = class_member(cur_cls, chain[1]); how = "self"
                if tgt and len(chain) > 2:
                    tgt = None  # self.x.y(): receiver unknown
            if tgt is None:
                r = resolve_name_in_module(c, chain)
                if r:
                    tgt, how = r, "scope"
            if tgt is None and len(chain) == 1 and chain[0] in c.nodes and False:
                pass
            if tgt is None:
                # enclosing-scope lookup for nested functions:  scope "mod.outer.inner" may call "helper" defined in mod.outer
                parts = scope.split(".")
                for i in range(len(parts), 0, -1):
                    cand = ".".join(parts[:i] + [chain[0]])
                    if cand in all_nodes and len(chain) == 1:
                        tgt, how = cand, "enclosing"; break
            if tgt is None:
                # unique-suffix fallback for method calls on unknown receivers
                if len(chain) >= 2:
                    key2 = ".".join(chain[-2:])
                    if len(by_tail2.get(key2, [])) == 1:
                        tgt, how = by_tail2[key2][0], "suffix2"
                if tgt is None and len(by_last.get(chain[-1], [])) == 1 and all_nodes[by_last[chain[-1]][0]]["kind"] != "module":
                    tgt, how = by_last[chain[-1]][0], "suffix1"
            if tgt is None:
                stats["unresolved"] += 1
                continue
            stats[how] += 1
            add(scope, tgt, "calls", resolution=how)

    nodes_out = [all_nodes[k] for k in sorted(all_nodes)]
    return {
        "directed": True, "multigraph": True,
        "graph": {"repo_name": name, "builder": "swegraphbench.build_graph", "call_resolution_stats": dict(stats)},
        "nodes": nodes_out, "edges": edges,
    }


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("out"); ap.add_argument("--name", default=None)
    ap.add_argument("--no-tests", action="store_true")
    a = ap.parse_args(argv)
    g = build(Path(a.root), a.name or Path(a.root).name, include_tests=not a.no_tests)
    Path(a.out).write_text(json.dumps(g), encoding="utf-8")
    kinds = defaultdict(int)
    for n in g["nodes"]:
        kinds[n["kind"]] += 1
    et = defaultdict(int)
    for e in g["edges"]:
        et[e["type"]] += 1
    print(json.dumps({"nodes": len(g["nodes"]), "kinds": kinds, "edges": len(g["edges"]), "edge_types": et,
                      "call_resolution": g["graph"]["call_resolution_stats"]}))


if __name__ == "__main__":
    main()
