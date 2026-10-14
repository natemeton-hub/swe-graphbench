"""For every released graph with a rebuilt counterpart: what does the released extractor drop or collapse?

Per graph:
  rebuilt_defs            non-module nodes in rebuilt graph (ground truth from `ast`)
  released_nodes          nodes in released graph
  exact_match             released ids that exist verbatim in rebuilt
  async_missing           rebuilt async defs with no released node (released drops them)
  nested_hoisted          released ids that only match a rebuilt node once the enclosing function is removed
  hoist_collisions        released ids to which >1 distinct rebuilt definitions map (ambiguous id)
  unexplained_released    released ids with no rebuilt counterpart at all
  call_direction          fraction of released call pairs that are reciprocal; fraction of rebuilt call pairs reciprocal
  released_pairs_confirmed released undirected call pairs that the resolver also finds (either direction)
"""
import json, sys, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from swegraphbench.graphio import load_graph, graph_key  # noqa

DATA, OUT = ROOT / "data", ROOT / "results"
rel = {graph_key(p): p for p in (DATA / "graphs").glob("*.json")}
reb = {graph_key(p): p for p in (OUT / "rebuilt_graphs").glob("*.json")}

rows = []
for key in sorted(rel):
    if key not in reb:
        continue
    R, B = load_graph(rel[key]), load_graph(reb[key])
    rid = {n["id"] for n in R["nodes"]}
    bn = {n["id"]: n for n in B["nodes"] if n["kind"] != "module"}
    # hoisted name: drop every *function* ancestor from the dotted path (keep classes/modules)
    def hoisted(nid):
        parts = nid.split("."); n = bn[nid]
        keep = []
        cur = n
        chain = [nid]
        while cur["parent"] in bn:
            chain.append(cur["parent"]); cur = bn[cur["parent"]]
        mod_prefix = cur["parent"]  # module id
        # rebuild id without function ancestors
        names = []
        for cid in reversed(chain):
            c = bn[cid]
            if c["kind"] == "function" and cid != nid and bn.get(c["parent"], {}).get("kind") != "class" or (c["kind"] == "function" and cid != nid):
                continue
            names.append(cid.split(".")[-1])
        return f"{mod_prefix}.{'.'.join(names)}" if mod_prefix else ".".join(names)
    hoist_map = collections.defaultdict(list)
    for nid in bn:
        hoist_map[hoisted(nid)].append(nid)
    exact = rid & set(bn)
    nested_hoisted = {r for r in rid - exact if r in hoist_map}
    collisions = {r for r in rid if len(hoist_map.get(r, [])) > 1}
    unexplained = rid - exact - nested_hoisted
    async_defs = {i for i, n in bn.items() if n["is_async"]}
    async_missing = {i for i in async_defs if i not in rid and hoisted(i) not in rid}
    rp = {(e["source"], e["target"]) for e in R["edges"]}
    bp = {(e["source"], e["target"]) for e in B["edges"] if e["type"] == "calls"}
    und_r = {frozenset(p) for p in rp if p[0] != p[1]}
    und_b = {frozenset(p) for p in bp if p[0] != p[1]}
    row = {
        "repo": key[0], "sha": key[1],
        "rebuilt_defs": len(bn), "released_nodes": len(rid), "exact_match": len(exact),
        "async_defs": len(async_defs), "async_missing": len(async_missing),
        "nested_hoisted": len(nested_hoisted), "hoist_collisions": len(collisions),
        "unexplained_released": len(unexplained), "unexplained_examples": sorted(unexplained)[:5],
        "released_recip_frac": sum(1 for s, t in rp if (t, s) in rp) / len(rp) if rp else None,
        "rebuilt_recip_frac": sum(1 for s, t in bp if (t, s) in bp) / len(bp) if bp else None,
        "released_und_pairs": len(und_r), "released_pairs_confirmed": len(und_r & und_b),
        "rebuilt_call_edges": len(bp), "rebuilt_edges_by_type": dict(collections.Counter(e["type"] for e in B["edges"])),
        "resolution": B["graph"].get("call_resolution_stats", {}),
    }
    rows.append(row)
    print(json.dumps({k: row[k] for k in ["sha", "rebuilt_defs", "released_nodes", "exact_match", "async_defs", "async_missing", "nested_hoisted", "hoist_collisions", "unexplained_released", "released_pairs_confirmed", "released_und_pairs"]}), flush=True)

with (OUT / "node_mapping_audit.jsonl").open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

def tot(k): return sum(r[k] for r in rows)
summary = {
    "n_graphs": len(rows),
    "rebuilt_defs": tot("rebuilt_defs"), "released_nodes": tot("released_nodes"), "exact_match": tot("exact_match"),
    "async_defs": tot("async_defs"), "async_missing": tot("async_missing"),
    "nested_hoisted": tot("nested_hoisted"), "hoist_collisions": tot("hoist_collisions"), "unexplained_released": tot("unexplained_released"),
    "released_und_pairs": tot("released_und_pairs"), "released_pairs_confirmed": tot("released_pairs_confirmed"),
    "rebuilt_call_edges": tot("rebuilt_call_edges"),
    "released_recip_frac_min": min(r["released_recip_frac"] for r in rows), "rebuilt_recip_frac_max": max(r["rebuilt_recip_frac"] for r in rows),
}
summary["frac_defs_missing_from_released"] = 1 - summary["exact_match"] / summary["rebuilt_defs"]
summary["frac_async_missing"] = summary["async_missing"] / summary["async_defs"] if summary["async_defs"] else None
(OUT / "node_mapping_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
