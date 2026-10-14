"""Unit test for the graph builder on a tiny synthetic package with known structure."""
import json, sys, tempfile, textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swegraphbench.build_graph import build  # noqa

FILES = {
    "src/pkg/__init__.py": "",
    "src/pkg/util.py": textwrap.dedent('''
        def helper(x):
            return x

        async def ahelper(x):
            return helper(x)
    '''),
    "src/pkg/core.py": textwrap.dedent('''
        from .util import helper, ahelper
        from pkg import util as u
        import pkg.util

        class Base:
            def run(self):
                return self.step()
            def step(self):
                return 1

        class Child(Base):
            def step(self):
                return helper(2) + self.run()
            def other(self):
                return u.helper(3) + pkg.util.helper(4)

        def outer():
            def inner():
                return helper(5)
            async def ainner():
                return await ahelper(6)
            return inner(), Child()
    '''),
    "tests/test_core.py": textwrap.dedent('''
        from pkg.core import outer
        def test_outer():
            assert outer()
    '''),
}


def test_builder():
    with tempfile.TemporaryDirectory() as td:
        for rel, src in FILES.items():
            p = Path(td) / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(src)
        g = build(Path(td), "synthetic")
    ids = {n["id"]: n for n in g["nodes"]}
    calls = {(e["source"], e["target"]): e["resolution"] for e in g["edges"] if e["type"] == "calls"}
    inherits = {(e["source"], e["target"]) for e in g["edges"] if e["type"] == "inherits"}
    contains = {(e["source"], e["target"]) for e in g["edges"] if e["type"] == "contains"}

    # src/ prefix dropped; async + nested defs present with full nesting
    for nid in ["pkg.util.helper", "pkg.util.ahelper", "pkg.core.Base.run", "pkg.core.Child.step",
                "pkg.core.outer", "pkg.core.outer.inner", "pkg.core.outer.ainner", "tests.test_core.test_outer"]:
        assert nid in ids, nid
    assert ids["pkg.util.ahelper"]["is_async"] and ids["pkg.core.outer.ainner"]["is_async"]
    assert ids["pkg.core.Child.step"]["file"] == "src/pkg/core.py" and ids["pkg.core.Child.step"]["lineno"] > 0

    # inheritance and containment
    assert ("pkg.core.Child", "pkg.core.Base") in inherits
    assert ("pkg.core.Child", "pkg.core.Child.step") in contains
    assert ("pkg.core.outer", "pkg.core.outer.inner") in contains

    # call resolution strategies
    assert calls[("pkg.core.Base.run", "pkg.core.Base.step")] == "self"          # self.step() inside Base
    assert calls[("pkg.core.Child.step", "pkg.core.Base.run")] == "self"         # inherited method via self
    assert calls[("pkg.core.Child.step", "pkg.util.helper")] == "scope"          # from-import
    assert calls[("pkg.core.Child.other", "pkg.util.helper")] in ("scope",)      # module alias + dotted import
    assert calls[("pkg.core.outer.inner", "pkg.util.helper")] == "scope"
    assert calls[("pkg.core.outer.ainner", "pkg.util.ahelper")] == "scope"
    assert calls[("pkg.core.outer", "pkg.core.outer.inner")] == "enclosing"      # nested call
    assert calls[("pkg.core.outer", "pkg.core.Child")] == "scope"                # instantiation -> class
    assert calls[("pkg.util.ahelper", "pkg.util.helper")] == "scope"
    assert calls[("tests.test_core.test_outer", "pkg.core.outer")] == "scope"
    # direction: no reciprocal pairs
    assert not any((t, s) in calls for s, t in calls)
    print("builder test passed;", len(ids), "nodes,", len(calls), "call edges")


if __name__ == "__main__":
    test_builder()
