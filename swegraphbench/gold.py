"""Map a task's reference patch to the set of symbols (graph node ids) it modifies.

Gold symbols are computed from the *base commit* version of each modified file:
for every hunk we take the old-file line span it touches and find the innermost
enclosing function/class (via `ast`). Dotted names follow the competition graph
convention (`pkg.module.Class.method`), derived from the file path.

Edge cases recorded explicitly rather than silently dropped:
  * files added by the patch (no base version) -> `new_files`
  * non-Python files (docs, yaml, md)            -> `non_python_files`
  * edits outside any def/class (module level)  -> `module_level_files`
"""
from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from unidiff import PatchSet


@dataclass
class GoldSpec:
    instance_id: str
    repo: str
    base_commit: str
    symbols: set[str] = field(default_factory=set)        # dotted symbol names
    files: set[str] = field(default_factory=set)          # all modified paths (any type)
    python_files: set[str] = field(default_factory=set)
    new_files: set[str] = field(default_factory=set)
    non_python_files: set[str] = field(default_factory=set)
    module_level_files: set[str] = field(default_factory=set)
    hunk_lines: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "symbols": sorted(self.symbols),
            "files": sorted(self.files),
            "python_files": sorted(self.python_files),
            "new_files": sorted(self.new_files),
            "non_python_files": sorted(self.non_python_files),
            "module_level_files": sorted(self.module_level_files),
            "hunk_lines": {k: v for k, v in self.hunk_lines.items()},
        }


STRIP_PREFIXES = ("src", "docs_src")  # the competition's extractor drops these leading directories


def module_name_from_path(path: str, strip_prefixes: tuple[str, ...] = STRIP_PREFIXES) -> str:
    """fastapi/routing.py -> fastapi.routing ; src/requests/models.py -> requests.models ;
    docs_src/a/b.py -> a.b ; pkg/__init__.py -> pkg  (matches the released graph convention)"""
    p = Path(path)
    parts = list(p.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    if len(parts) > 1 and parts[0] in strip_prefixes:
        parts = parts[1:]
    return ".".join(parts)


class _SymbolIndex(ast.NodeVisitor):
    """Collect (qualname, start_line, end_line) for every def/class, nested included."""

    def __init__(self) -> None:
        self.spans: list[tuple[str, int, int]] = []
        self._stack: list[str] = []

    def _visit_def(self, node: ast.AST) -> None:
        name = getattr(node, "name")
        self._stack.append(name)
        qual = ".".join(self._stack)
        start = node.lineno
        # include decorators in the span, as the competition's extractor does
        if getattr(node, "decorator_list", None):
            start = min(start, min(d.lineno for d in node.decorator_list))
        self.spans.append((qual, start, node.end_lineno or start))
        self.generic_visit(node)
        self._stack.pop()

    visit_FunctionDef = _visit_def
    visit_AsyncFunctionDef = _visit_def
    visit_ClassDef = _visit_def


def symbol_spans(source: str) -> list[tuple[str, int, int]]:
    tree = ast.parse(source)
    idx = _SymbolIndex()
    idx.visit(tree)
    return idx.spans


def innermost_symbol(spans: list[tuple[str, int, int]], line: int) -> str | None:
    best: tuple[str, int, int] | None = None
    for qual, s, e in spans:
        if s <= line <= e and (best is None or (e - s) < (best[2] - best[1])):
            best = (qual, s, e)
    return best[0] if best else None


def git_show(repo_dir: Path, commit: str, path: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), "show", f"{commit}:{path}"],
            capture_output=True, check=True,
        ).stdout
    except subprocess.CalledProcessError:
        return None
    return out.decode("utf-8", errors="replace")


def hunk_old_lines(hunk) -> tuple[int, int]:
    """Old-file line span a hunk touches. For pure insertions, use the insertion point
    (the old line *before* the addition) so we still attribute to the enclosing symbol."""
    touched = [l.source_line_no for l in hunk if l.is_removed or l.is_context]
    removed = [l.source_line_no for l in hunk if l.is_removed]
    if removed:
        return min(removed), max(removed)
    if touched:
        # pure insertion: attribute to the context line immediately preceding the first add
        first_add_idx = next((i for i, l in enumerate(hunk) if l.is_added), None)
        if first_add_idx is not None and first_add_idx > 0:
            prev = hunk[first_add_idx - 1]
            if prev.source_line_no:
                return prev.source_line_no, prev.source_line_no
        return min(touched), max(touched)
    return hunk.source_start, hunk.source_start


def gold_for_task(task: dict, repo_dir: Path) -> GoldSpec:
    g = GoldSpec(task["instance_id"], task["repo"], task["base_commit"])
    patch = PatchSet(task["patch"])
    for pf in patch:
        path = pf.path
        g.files.add(path)
        if pf.is_added_file:
            g.new_files.add(path)
            continue
        if not path.endswith(".py"):
            g.non_python_files.add(path)
            continue
        g.python_files.add(path)
        src = git_show(repo_dir, task["base_commit"], path)
        if src is None:
            g.new_files.add(path)
            continue
        try:
            spans = symbol_spans(src)
        except SyntaxError:
            g.module_level_files.add(path)
            continue
        mod = module_name_from_path(path)
        found_any = False
        for h in pf:
            s, e = hunk_old_lines(h)
            g.hunk_lines.setdefault(path, []).append((s, e))
            syms = {innermost_symbol(spans, ln) for ln in range(s, e + 1)}
            syms.discard(None)
            if syms:
                found_any = True
                g.symbols.update(f"{mod}.{q}" for q in syms)
        if not found_any:
            g.module_level_files.add(path)
    return g
