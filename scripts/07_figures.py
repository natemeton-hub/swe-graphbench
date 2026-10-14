"""Paper figures (static PNG/SVG for the Kaggle writeup). Light mode, reference palette.

fig1_audit.png     one dot per released graph (68) for four structural metrics + median tick
fig2_compare.png   Hit@5 and MRR per method, released vs rebuilt graph (tests filtered), 65 tasks, 95% CI
fig3_unlocked.png  Hit@5 / Recall@10 / MRR on the 56 rich+requests tasks that have no released graph
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# reference palette (light)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 9,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
})


def style(ax, xgrid=True):
    if xgrid:
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.spines["left"].set_color(AXIS); ax.spines["bottom"].set_color(AXIS)


# ---------------- Figure 1: audit strip plot ----------------
def fig1():
    rows = [json.loads(l) for l in (RES / "graph_audit.jsonl").open(encoding="utf-8")]
    metrics = [("reciprocal_pair_frac", "call edges that are reciprocal"),
               ("isolated_frac", "nodes with no edge"),
               ("test_node_frac", "nodes that are test code"),
               ("lib_node_frac", "nodes that are library code")]
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    import random
    rng = random.Random(0)
    for i, (k, label) in enumerate(metrics):
        vals = [r[k] for r in rows]
        y = [i + rng.uniform(-0.18, 0.18) for _ in vals]
        ax.scatter(vals, y, s=14, color=S1, alpha=0.55, linewidths=0, zorder=3)
        med = sorted(vals)[len(vals) // 2]
        ax.plot([med, med], [i - 0.32, i + 0.32], color=INK, linewidth=2, zorder=4)
        ax.text(1.02, i, f"median {med:.0%}", va="center", ha="left", fontsize=8, color=INK2, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(metrics))); ax.set_yticklabels([m[1] for m in metrics]); ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02); ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel(f"share, one dot per released graph (n = {len(rows)})")
    style(ax)
    fig.tight_layout(); fig.savefig(FIG / "fig1_audit.png", dpi=220); fig.savefig(FIG / "fig1_audit.svg")
    print("fig1 done")


# ---------------- Figure 2: released vs rebuilt ----------------
def fig2():
    s = json.loads((RES / "compare_graphs.json").read_text())
    conds = [("released", "released graph, as shipped", S1), ("released-notests", "released graph, tests filtered", S3), ("rebuilt-notests", "rebuilt graph, tests filtered", S2)]
    methods = [("bm25_text", "BM25 (code text)"), ("symbol_seed", "symbol seed"), ("ppr", "PageRank"), ("hybrid", "hybrid (RRF)")]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    h = 0.24
    for ax, (metric, title) in zip(axes, [("hit@5", "Hit@5"), ("recall@10", "Recall@10")]):
        for j, (cname, clabel, color) in enumerate(conds):
            for i, (m, _) in enumerate(methods):
                v = s[cname]["methods"][m][metric]; mean, (lo, hi) = v["mean"], v["ci95"]
                y = i + (j - 1) * (h + 0.03)
                ax.barh(y, mean, height=h, color=color, linewidth=0, label=clabel if i == 0 else None, zorder=3)
                ax.plot([lo, hi], [y, y], color=INK, linewidth=0.7, alpha=0.6, zorder=4)
                ax.text(hi + 0.015, y, f"{mean:.2f}", va="center", fontsize=7, color=INK2)
        ax.set_yticks(range(len(methods))); ax.set_yticklabels([m[1] for m in methods]); ax.invert_yaxis()
        ax.set_xlim(0, 0.75); ax.set_title(title, loc="left", fontsize=10, color=INK)
        style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    n = s["released-notests"]["n_tasks"]
    fig.suptitle(f"Symbol localization on the same {n} FastAPI/httpx tasks. Bars: mean; ticks: 95% bootstrap CI.", x=0.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(FIG / "fig2_compare.png", dpi=220); fig.savefig(FIG / "fig2_compare.svg")
    print("fig2 done")


# ---------------- Figure 3: tasks unlocked ----------------
def fig3():
    s = json.loads((RES / "compare_graphs.json").read_text())
    c = s["rebuilt-notests-NOREL"]
    methods = [("bm25_text", "BM25 (code text)"), ("symbol_seed", "symbol seed"), ("ppr", "PageRank"), ("hybrid", "hybrid (RRF)")]
    metrics = [("hit@5", "Hit@5", S1), ("recall@10", "Recall@10", S2), ("mrr", "MRR", S3)]
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    h = 0.24
    for j, (metric, label, color) in enumerate(metrics):
        for i, (m, _) in enumerate(methods):
            v = c["methods"][m][metric]; y = i + (j - 1) * (h + 0.03)
            ax.barh(y, v["mean"], height=h, color=color, linewidth=0, label=label if i == 0 else None, zorder=3)
            ax.plot(v["ci95"], [y, y], color=INK, linewidth=0.7, alpha=0.6, zorder=4)
            ax.text(v["ci95"][1] + 0.015, y, f"{v['mean']:.2f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(methods))); ax.set_yticklabels([m[1] for m in methods]); ax.invert_yaxis()
    ax.set_xlim(0, 0.75)
    fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    ax.set_title(f"rich + requests: {c['n_tasks']} tasks with no released graph, rebuilt graph, tests filtered", loc="left", fontsize=9, color=INK2)
    style(ax)
    fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig(FIG / "fig3_unlocked.png", dpi=220); fig.savefig(FIG / "fig3_unlocked.svg")
    print("fig3 done")



# ---------------- Figure 4: SWE-bench Verified per repository ----------------
def fig4():
    s = json.loads((ROOT / "results_verified" / "summary.json").read_text())
    rows = [json.loads(l) for l in (ROOT / "results_verified" / "bench_per_task.jsonl").open(encoding="utf-8")]
    rows = list({r["instance_id"]: r for r in rows}.values())
    import collections, random
    by = collections.defaultdict(list)
    for r in rows:
        by[r["repo"]].append(r["methods"]["hybrid"]["hit@5"])
    big = {k: v for k, v in by.items() if len(v) >= 8}
    small = [x for k, v in by.items() if len(v) < 8 for x in v]
    if small:
        big["other (seaborn, flask)"] = small
    items = sorted(big.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))
    def boot(vals, n=2000, seed=0):
        rng = random.Random(seed); m = []
        for _ in range(n):
            smp = [vals[rng.randrange(len(vals))] for _ in vals]; m.append(sum(smp) / len(smp))
        m.sort(); return m[int(0.025 * n)], m[int(0.975 * n)]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    overall = s["overall"]["methods"]["hybrid"]["hit@5"]["mean"]
    ax.axvline(overall, color=AXIS, linewidth=1, linestyle="--", zorder=2)
    ax.text(overall + 0.01, -0.7, f"all 484 tasks: {overall:.2f}", fontsize=7.5, color=INK2, va="center")
    for i, (repo, vals) in enumerate(items):
        m = sum(vals) / len(vals); lo, hi = boot(vals)
        ax.barh(i, m, height=0.6, color=S1, linewidth=0, zorder=3)
        ax.plot([lo, hi], [i, i], color=INK, linewidth=0.7, alpha=0.6, zorder=4)
        ax.text(hi + 0.015, i, f"{m:.2f}  (n = {len(vals)})", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([k.split("/")[-1] if "/" in k else k for k, _ in items]); ax.invert_yaxis()
    ax.set_xlim(0, 1.0); ax.set_xlabel("hybrid Hit@5, rebuilt graph, tests filtered")
    ax.set_title("SWE-bench Verified, per repository (bars: mean, ticks: 95% bootstrap CI)", loc="left", fontsize=9, color=INK2)
    style(ax)
    fig.tight_layout(); fig.savefig(FIG / "fig4_verified.png", dpi=220); fig.savefig(FIG / "fig4_verified.svg")
    print("fig4 done")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4"]
    for w in which:
        {"1": fig1, "2": fig2, "3": fig3, "4": fig4}[w]()
