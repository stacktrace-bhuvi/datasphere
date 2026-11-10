"""Figures for the report: error decomposition, estimand gap by family, diagnostic performance.

All figures use one fixed categorical palette (assigned in fixed order, never cycled)
and a single y-axis per panel. Output is PNG + PDF, sized for a two-column paper.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FAMILY_ORDER = ["chain_mediation", "confounded_fork", "feature_collider", "erdos_renyi_dag", "layered"]
FAMILY_LABEL = {"chain_mediation": "Chain", "confounded_fork": "Fork", "feature_collider": "Collider",
                "erdos_renyi_dag": "Random DAG", "layered": "Layered (held-out)"}
EXPLAINER_ORDER = ["exact", "treeshap", "kernelshap", "lime", "permutation"]
EXPLAINER_LABEL = {"exact": "Exact SHAP", "treeshap": "TreeSHAP", "kernelshap": "KernelSHAP",
                   "lime": "LIME", "permutation": "Permutation"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def _save(fig, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.png", out_dir / f"{name}.pdf"]
    for p in paths:
        fig.savefig(p)
    plt.close(fig)
    return paths


def fig_error_decomposition(errors: pd.DataFrame, out_dir: Path, space: str = "share") -> list[Path]:
    """Mean component error per explainer (x) and model (panel), share space by default."""
    comps = [f"estimand_gap_{space}", f"model_misfit_{space}", f"explainer_error_{space}"] \
        if space == "share" else ["estimand_gap", "model_misfit", "explainer_error"]
    labels = ["Estimand gap", "Model misfit", "Explainer error"]
    models = sorted(errors["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(3.4 * len(models), 2.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        sub = errors[errors["model"] == m]
        exps = [e for e in EXPLAINER_ORDER if e in sub["explainer"].unique()]
        means = sub.groupby("explainer")[comps].mean().loc[exps]
        x = np.arange(len(exps))
        w = 0.26
        for i, (c, lab) in enumerate(zip(comps, labels)):
            ax.bar(x + (i - 1) * w, means[c], width=w * 0.92, color=PALETTE[i], label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels([EXPLAINER_LABEL[e] for e in exps], rotation=20, ha="right")
        ax.set_title(f"{m}")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("L2 distance between importance shares" if space == "share" else "normalized L2")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Where explanation error comes from (mean over SCMs and seeds)", y=1.02)
    return _save(fig, out_dir, f"fig_error_decomposition_{space}")


def fig_estimand_gap_by_family(errors: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Estimand gap (rank and share space) per graph family, one point per SCM."""
    per_scm = errors.groupby(["scm_id", "family"], as_index=False)[["estimand_gap_rank", "estimand_gap_share"]].first()
    fams = [f for f in FAMILY_ORDER if f in per_scm["family"].unique()]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    for ax, col, ttl in zip(axes, ["estimand_gap_rank", "estimand_gap_share"],
                            ["Rank error: 1 - Kendall tau-b(tau, exact SHAP of f*)",
                             "Share error: ||share(tau) - share(exact SHAP of f*)||"]):
        data = [per_scm.loc[per_scm["family"] == f, col].to_numpy() for f in fams]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False, medianprops={"color": "#0b0b0b"})
        for patch, f in zip(bp["boxes"], fams):
            patch.set_facecolor(PALETTE[FAMILY_ORDER.index(f)])
            patch.set_alpha(0.75)
        for i, d in enumerate(data, 1):
            ax.scatter(np.full(len(d), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(d)), d,
                       s=6, color="#52514e", alpha=0.6, zorder=3)
        ax.set_xticks(range(1, len(fams) + 1))
        ax.set_xticklabels([FAMILY_LABEL[f] for f in fams], rotation=20, ha="right")
        ax.set_title(ttl, fontsize=8.5)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Estimand gap is structural: it exists with a perfect model and exact SHAP", y=1.03)
    return _save(fig, out_dir, "fig_estimand_gap_by_family")


def fig_total_error_by_explainer(errors: pd.DataFrame, out_dir: Path, target: str = "total_error_rank") -> list[Path]:
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    exps = [e for e in EXPLAINER_ORDER if e in errors["explainer"].unique()]
    models = sorted(errors["model"].unique())
    x = np.arange(len(exps))
    w = 0.38
    for i, m in enumerate(models):
        sub = errors[errors["model"] == m].groupby("explainer")[target]
        means, sds = sub.mean().reindex(exps), sub.std().reindex(exps)
        ax.bar(x + (i - 0.5) * w, means, width=w * 0.92, color=PALETTE[i], label=m)
        ax.errorbar(x + (i - 0.5) * w, means, yerr=sds, fmt="none", ecolor="#52514e", elinewidth=0.8, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([EXPLAINER_LABEL[e] for e in exps], rotation=20, ha="right")
    ax.set_ylabel({"total_error_rank": "1 - Kendall tau-b vs tau",
                   "total_error_share_l2": "share L2 vs tau"}.get(target, target))
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    ax.set_title("Total explanation error by explainer (mean +- sd)")
    return _save(fig, out_dir, f"fig_total_error_by_explainer_{target}")


def fig_logfo_performance(scores: pd.DataFrame, out_dir: Path, target: str, metric: str = "spearman",
                          methods: list[str] | None = None) -> list[Path]:
    """Per held-out family metric for each method under LOGFO (grouped bars)."""
    s = scores[(scores["protocol"] == "logfo") & (scores["target"] == target) & (scores["fold"] != "pooled")
               & ~scores["fold"].astype(str).str.startswith("by_method")]
    methods = methods or list(dict.fromkeys(s["method"]))
    fams = [f for f in FAMILY_ORDER if f in s["fold"].unique()]
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    x = np.arange(len(fams))
    w = 0.8 / max(len(methods), 1)
    for i, m in enumerate(methods):
        vals = [s[(s["method"] == m) & (s["fold"] == f)][metric].mean() for f in fams]
        ax.bar(x + (i - (len(methods) - 1) / 2) * w, vals, width=w * 0.92, color=PALETTE[i % len(PALETTE)], label=m)
    ax.axhline(0, color="#0b0b0b", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_LABEL[f] for f in fams])
    ax.set_ylabel({"spearman": "Spearman(pred, actual)", "r2": "R^2", "auroc_high_error": "AUROC (high error)"}.get(metric, metric))
    ax.set_title(f"Leave-one-graph-family-out: predicting {target}")
    ax.legend(frameon=False, ncol=min(4, len(methods)), fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.22))
    ax.grid(axis="x", visible=False)
    return _save(fig, out_dir, f"fig_logfo_{metric}_{target}")


def fig_pred_vs_actual(preds: pd.DataFrame, out_dir: Path, target: str, method: str, protocol: str) -> list[Path]:
    """Predicted vs actual error, coloured by held-out family."""
    p = preds[preds["method"] == method]
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    fams = [f for f in FAMILY_ORDER if f in p["family"].unique()]
    for f in fams:
        g = p[p["family"] == f]
        ax.scatter(g["y"], g["pred"], s=8, alpha=0.55, color=PALETTE[FAMILY_ORDER.index(f)], label=FAMILY_LABEL[f],
                   edgecolors="none")
    lo, hi = np.nanmin(p[["y", "pred"]].to_numpy()), np.nanmax(p[["y", "pred"]].to_numpy())
    ax.plot([lo, hi], [lo, hi], color="#0b0b0b", linewidth=0.8, linestyle="--")
    ax.set_xlabel(f"actual {target}")
    ax.set_ylabel("predicted")
    ax.set_title(f"{method} under {protocol}", fontsize=9)
    ax.legend(frameon=False, fontsize=6.5, markerscale=1.5)
    return _save(fig, out_dir, f"fig_pred_vs_actual_{protocol}_{method}_{target}")


def fig_feature_importance(importances: pd.DataFrame, out_dir: Path, target: str, top: int = 15) -> list[Path]:
    """Horizontal bars of permutation importance of the observables inside the meta-model."""
    imp = importances.sort_values("importance_mean", ascending=True).tail(top)
    fig, ax = plt.subplots(figsize=(4.2, 0.22 * len(imp) + 1.0))
    ax.barh(imp["feature"], imp["importance_mean"], xerr=imp["importance_std"], color=PALETTE[0],
            error_kw={"ecolor": "#52514e", "elinewidth": 0.8, "capsize": 2})
    ax.set_xlabel("permutation importance (drop in R^2 on held-out family)")
    ax.set_title(f"Which observables predict {target}?", fontsize=9)
    ax.grid(axis="y", visible=False)
    return _save(fig, out_dir, f"fig_feature_importance_{target}")


def fig_observable_scatter(df: pd.DataFrame, out_dir: Path, target: str, observables: list[str]) -> list[Path]:
    """Small multiples: each observable against the target, one point per row, coloured by family."""
    n = len(observables)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.0, 2.2 * rows), squeeze=False)
    fams = [f for f in FAMILY_ORDER if f in df["family"].unique()]
    for ax, obs in zip(axes.ravel(), observables):
        for f in fams:
            g = df[df["family"] == f]
            ax.scatter(g[obs], g[target], s=5, alpha=0.45, color=PALETTE[FAMILY_ORDER.index(f)], edgecolors="none",
                       label=FAMILY_LABEL[f])
        rho = pd.Series(df[obs]).corr(df[target], method="spearman")
        ax.set_title(f"{obs}  (rho={rho:+.2f})", fontsize=8)
        ax.set_ylabel(target, fontsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes.ravel()[0].legend(frameon=False, fontsize=6, markerscale=1.5)
    fig.tight_layout()
    return _save(fig, out_dir, f"fig_observables_vs_{target}")
