"""Collect every number used in the report / deck / one-pager from a results directory.

    python deliverables/collect_numbers.py --scms-dir data/scms --results-dir results --out deliverables/numbers.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.scm.scm import SCM  # noqa: E402

FAMILIES = ["chain_mediation", "confounded_fork", "feature_collider", "erdos_renyi_dag", "layered"]


def oracle_r2(scms_dir: Path, manifest: pd.DataFrame) -> dict:
    """R^2 of the true function f* on each SCM's own draws (how learnable the outcome is)."""
    vals = {}
    for scm_id in manifest["scm_id"]:
        d = scms_dir / scm_id
        scm = SCM.from_json(d / "scm.json")
        df = pd.read_parquet(d / "data.parquet")
        fs = scm.f_star(df[scm.features].to_numpy())
        vals[scm_id] = float(1 - np.var(df["Y"].to_numpy() - fs) / np.var(df["Y"].to_numpy()))
    s = pd.Series(vals)
    return {"mean": float(s.mean()), "min": float(s.min()), "max": float(s.max()), "per_scm": vals}


def mediated_share(scms_dir: Path, errors: pd.DataFrame, manifest: pd.DataFrame) -> dict:
    """Share of true effect vs attributed importance that sits on features that are NOT parents of Y."""
    out = {}
    attr_dir = scms_dir.parent.parent / "results" / "attributions"
    for fam in FAMILIES:
        ids = manifest[manifest["family"] == fam]["scm_id"].tolist()
        rows = []
        for scm_id in ids:
            d = scms_dir / scm_id
            scm = SCM.from_json(d / "scm.json")
            gt = json.load(open(d / "ground_truth.json"))
            tau = np.array(gt["tau"])
            oracle = np.array(gt["oracle_mean_abs_shap"])
            non_parents = np.array([f not in set(scm.graph.predecessors("Y")) for f in scm.features])
            r = {"tau": tau[non_parents].sum() / tau.sum(), "oracle": oracle[non_parents].sum() / max(oracle.sum(), 1e-12),
                 "n_nonparents": int(non_parents.sum()), "d": scm.d}
            for ex in ("kernelshap", "lime", "permutation"):
                vals = []
                for model in ("xgboost", "mlp"):
                    p = attr_dir / f"{scm_id}__{model}__{ex}__seed0.json"
                    if p.exists():
                        v = np.array(json.load(open(p))["mean_abs_attributions"])
                        vals.append(v[non_parents].sum() / max(v.sum(), 1e-12))
                if vals:
                    r[ex] = float(np.mean(vals))
            rows.append(r)
        df = pd.DataFrame(rows)
        out[fam] = {k: float(df[k].mean()) for k in df.columns if k in ("tau", "oracle", "kernelshap", "lime", "permutation")}
        out[fam]["mean_nonparents"] = float(df["n_nonparents"].mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scms-dir", default="data/scms")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="deliverables/numbers.json")
    ap.add_argument("--skip-slow", action="store_true")
    a = ap.parse_args()
    scms, res = Path(a.scms_dir), Path(a.results_dir)

    manifest = pd.read_csv(scms / "manifest.csv")
    errors = pd.read_csv(res / "errors.csv").merge(manifest[["scm_id", "family", "split", "d"]], on="scm_id")
    obs = pd.read_csv(res / "observables.csv")
    models = pd.read_csv(res / "models.csv") if (res / "models.csv").exists() else None
    summary = json.load(open(res / "diagnostic_summary.json"))
    scores = pd.read_csv(res / "tables" / "diagnostic_scores.csv")
    tests = pd.read_csv(res / "tables" / "paired_tests.csv")
    cis = pd.read_csv(res / "tables" / "metric_bootstrap_ci.csv")

    N = {"benchmark": {
        "n_scms": int(manifest.shape[0]), "per_family": manifest["family"].value_counts().to_dict(),
        "d_counts": manifest["d"].value_counts().sort_index().to_dict(),
        "n_samples": 2000, "heldout_family": "layered",
        "n_attribution_rows": int(len(errors)), "n_scm_model_seed": int(errors[["scm_id", "model", "seed"]].drop_duplicates().shape[0]),
        "seeds": sorted(errors["seed"].unique().tolist()), "explainers": sorted(errors["explainer"].unique().tolist()),
        "n_observables": summary["n_observables"], "observables": summary["observables"],
    }}
    if not a.skip_slow:
        N["oracle_r2"] = oracle_r2(scms, manifest)
        N["oracle_r2"].pop("per_scm")
    if models is not None:
        N["model_fit"] = {m: {"test_r2_mean": float(g["test_r2"].mean()), "test_r2_min": float(g["test_r2"].min()),
                              "test_r2_max": float(g["test_r2"].max())} for m, g in models.groupby("model")}

    comp = ["estimand_gap_share", "model_misfit_share", "explainer_error_share", "total_error_share_l2",
            "estimand_gap_rank", "model_misfit_rank", "explainer_error_rank", "total_error_rank",
            "estimand_gap", "model_misfit", "explainer_error", "total_error_l2", "top_k_overlap"]
    N["decomposition_by_model_explainer"] = {f"{m}/{e}": g[comp].mean().round(4).to_dict()
                                             for (m, e), g in errors.groupby(["model", "explainer"])}
    N["decomposition_by_explainer"] = {e: g[comp].mean().round(4).to_dict() for e, g in errors.groupby("explainer")}
    N["error_by_family"] = {f: g[comp].mean().round(4).to_dict() for f, g in errors.groupby("family")}
    gap = errors.groupby(["scm_id", "family"], as_index=False)[["estimand_gap_rank", "estimand_gap_share", "estimand_gap"]].first()
    N["estimand_gap_by_family"] = {f: {"rank_mean": float(g["estimand_gap_rank"].mean()), "rank_sd": float(g["estimand_gap_rank"].std()),
                                       "share_mean": float(g["estimand_gap_share"].mean()), "share_sd": float(g["estimand_gap_share"].std()),
                                       "n": int(len(g))} for f, g in gap.groupby("family")}
    N["estimand_gap_overall"] = {"rank_mean": float(gap["estimand_gap_rank"].mean()), "share_mean": float(gap["estimand_gap_share"].mean()),
                                 "frac_scms_rank_gap_gt_0": float((gap["estimand_gap_rank"] > 1e-9).mean())}
    N["error_by_family_explainer"] = {f"{f}/{e}": {"rank": float(g["total_error_rank"].mean()), "share": float(g["total_error_share_l2"].mean())}
                                      for (f, e), g in errors.groupby(["family", "explainer"])}
    if not a.skip_slow:
        N["mediated_share"] = mediated_share(scms, errors, manifest)

    # correlations of single observables with the targets (pooled)
    meta = errors.drop(columns=["d"]).merge(obs, on=["scm_id", "model", "explainer", "seed"])
    obs_cols = [c for c in summary["observables"] if c in meta.columns]
    N["observable_spearman"] = {t: meta[obs_cols].corrwith(meta[t], method="spearman").round(3).sort_values(key=abs, ascending=False).to_dict()
                                for t in ("total_error_rank", "total_error_share_l2")}

    N["diagnostic"] = summary["headline"]
    pooled = scores[scores["fold"] == "pooled"]
    N["diagnostic_pooled"] = {t: {p: g.set_index("method")[["spearman", "r2", "mae", "auroc_high_error"]].round(4).to_dict("index")
                                  for p, g in pooled[pooled["target"] == t].groupby("protocol")} for t in pooled["target"].unique()}
    fam = scores[(~scores["fold"].isin(["pooled"])) & (~scores["fold"].astype(str).str.startswith("by_method"))
                 & (scores["protocol"] == "logfo")]
    N["logfo_by_family"] = {t: {m: g.set_index("fold")["spearman"].round(3).to_dict() for m, g in fam[fam["target"] == t].groupby("method")}
                            for t in fam["target"].unique()}
    N["logfo_by_family_auroc"] = {t: {m: g.set_index("fold")["auroc_high_error"].round(3).to_dict() for m, g in fam[fam["target"] == t].groupby("method")}
                                  for t in fam["target"].unique()}
    bym = scores[scores["fold"].astype(str).str.startswith("by_method") & (scores["protocol"] == "logfo")]
    N["logfo_by_method"] = {t: {m: g.assign(k=g["fold"].str.replace("by_method:", "")).set_index("k")["spearman"].round(3).to_dict()
                                for m, g in bym[bym["target"] == t].groupby("method")} for t in bym["target"].unique()}
    N["paired_tests"] = tests.round(4).to_dict("records")
    N["metric_ci"] = cis.round(4).to_dict("records")
    fi = {}
    for t in ("total_error_rank", "total_error_share_l2", "model_misfit_share", "estimand_gap_share", "explainer_error_share"):
        p = res / "tables" / f"feature_importance_{t}.csv"
        if p.exists():
            fi[t] = pd.read_csv(p).head(12).round(4).to_dict("records")
    N["feature_importance"] = fi

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    def _clean(o):
        """Make strictly valid JSON: NaN/inf -> None, numpy scalars -> python."""
        if isinstance(o, dict):
            return {str(k): _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(_clean(N), f, indent=2, allow_nan=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
