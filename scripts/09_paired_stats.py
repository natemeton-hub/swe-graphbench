"""Paired comparison between conditions on the same tasks (results/compare_graphs_per_task.jsonl).

For each (condition A, condition B, method, metric): mean difference B-A, 95% paired bootstrap CI,
and for hit@5 the exact McNemar / sign-test p-value on discordant tasks.
"""
import json, random, sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
rows = [json.loads(l) for l in (OUT / "compare_graphs_per_task.jsonl").open(encoding="utf-8")]
by = {}
for r in rows:
    by.setdefault(r["condition"], {})[r["instance_id"]] = r

PAIRS = [("released", "released-notests"), ("released-notests", "rebuilt-notests"), ("rebuilt-notests", "rebuilt-notests-dir"),
         ("rebuilt-notests", "rebuilt-notests-rev"), ("rebuilt-notests", "rebuilt-relemb-notests"), ("released", "rebuilt-notests")]
METHODS = ["bm25_text", "ppr", "hybrid"]
METRICS = ["hit@5", "recall@10", "mrr"]


def boot(diffs, n=5000, seed=0):
    rng = random.Random(seed); m = []
    for _ in range(n):
        s = [diffs[rng.randrange(len(diffs))] for _ in diffs]; m.append(sum(s) / len(s))
    m.sort(); return m[int(0.025 * n)], m[int(0.975 * n)]


def sign_test(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


out = []
print(f"{'A -> B':44s} {'method':10s} {'metric':9s} {'meanA':>6s} {'meanB':>6s} {'diff':>7s} {'CI95':>17s} {'win/loss':>9s} {'p':>7s}")
for a, b in PAIRS:
    if a not in by or b not in by:
        continue
    ids = sorted(set(by[a]) & set(by[b]))
    for m in METHODS:
        for met in METRICS:
            va = [by[a][i]["methods"][m][met] for i in ids]; vb = [by[b][i]["methods"][m][met] for i in ids]
            d = [y - x for x, y in zip(va, vb)]
            lo, hi = boot(d)
            wins = sum(1 for x in d if x > 1e-12); losses = sum(1 for x in d if x < -1e-12)
            p = sign_test(wins, losses)
            rec = {"A": a, "B": b, "method": m, "metric": met, "n": len(ids), "meanA": sum(va) / len(va), "meanB": sum(vb) / len(vb),
                   "diff": sum(d) / len(d), "ci95": [lo, hi], "wins": wins, "losses": losses, "sign_p": p}
            out.append(rec)
            print(f"{a+' -> '+b:44s} {m:10s} {met:9s} {rec['meanA']:6.3f} {rec['meanB']:6.3f} {rec['diff']:+7.3f} [{lo:+.3f},{hi:+.3f}] {wins:4d}/{losses:<4d} {p:7.3f}")
(OUT / "paired_stats.json").write_text(json.dumps(out, indent=1))
