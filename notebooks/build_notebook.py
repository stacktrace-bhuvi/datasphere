"""Build notebooks/DataSphere_pipeline.ipynb with nbformat.

    python notebooks/build_notebook.py            # write the notebook
    python notebooks/build_notebook.py --execute  # also run it (needs results/ or results_small/)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parent / "DataSphere_pipeline.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
cells: list = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))      # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))        # noqa: E731

md("""
# Predicting Explanation Error Without Causal Ground Truth
## A Held-Out-SCM Study of SHAP and LIME

**Team DataSphere (Section G)** — Bhuvi S. Katti, Mythri S, Poojitha L. B. R., Diran Somanna
Advanced Data Analytics (Sem 5), PES University · code: https://github.com/stacktrace-bhuvi/datasphere

SHAP and LIME describe *the model*; people read them as statements about *the world* ("this feature drives the outcome").
With real data nobody can tell when that reading is wrong, because the causal truth is unknown.
This notebook walks through our five-stage pipeline on live examples, then reproduces the headline results of the
report from the full 300-SCM run.

| Stage | What | Code |
|---|---|---|
| 1 | Synthetic causal worlds (SCMs) from 5 graph families | `src/scm/graph.py`, `mechanisms.py`, `scm.py` |
| 2 | Causal ground truth: total effect $\\tau_j$ under $do(X_j)$, exact SHAP of the true function $f^*$ | `scm.py::true_effects`, `oracle.py`, `ground_truth.py` |
| 3 | Models (XGBoost, MLP) + explainers (exact SHAP, TreeSHAP, KernelSHAP, LIME, permutation) | `src/models`, `src/explainers` |
| 4 | Error decomposition (needs truth) and observables (no truth) | `src/metrics`, `src/observables` |
| 5 | Diagnostic meta-models vs baselines; grouped CV, leave-one-graph-family-out, paired tests | `src/diagnostic` |
""")

md("### 0. Setup\nRun from the repository root (the cell clones the repo if needed, e.g. on Colab).")
code('''
import os, sys, json, pathlib, subprocess, warnings
if not pathlib.Path("src/scm/scm.py").exists():
    if pathlib.Path("../src/scm/scm.py").exists():
        os.chdir("..")
    else:
        subprocess.run(["git", "clone", "https://github.com/stacktrace-bhuvi/datasphere.git"], check=True)
        os.chdir("datasphere")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
ROOT = pathlib.Path.cwd(); sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, networkx as nx
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 40); pd.set_option("display.precision", 3)
from IPython.display import Image, display
print("repo root:", ROOT)
''')

md("""
## Stage 1 — A causal world with known structure

An SCM is a DAG over features $X_1..X_d$ and a sink $Y$, plus one structural equation per node
($X_v = f_v(\\text{parents}) + \\varepsilon_v$). We draw the graph from one of five families and the mechanisms
at random (linear, tanh, sin, quadratic, ReLU, pairwise interaction). Below: a chain-mediation world.
""")
code('''
from src.scm.scm import SCM
scm = SCM.build("chain_mediation", d=6, seed=3, nonlinearity_prob=0.5, interaction_prob=0.2, outcome_noise_scale=0.4)
print("family:", scm.metadata["family"], "| d =", scm.d, "| non-linear nodes:", scm.metadata["n_nonlinear"],
      "| confounding:", scm.metadata["has_confounding"], "| mediation share:", scm.metadata["mediation_share"])
for node in scm.topological_order:
    m = scm.mechanisms[node]
    print(f"  {node:>2} <- {m.parents or ['(root)']}  {m.mechanism_type}  nl={m.nonlinearities}")
fig, ax = plt.subplots(figsize=(5, 3))
pos = nx.nx_agraph.graphviz_layout(scm.graph, prog="dot") if hasattr(nx, "nx_agraph") and False else nx.spring_layout(scm.graph, seed=2)
nx.draw_networkx(scm.graph, pos, ax=ax, node_color=["#eb6834" if n == "Y" else "#2a78d6" for n in scm.graph.nodes],
                 font_color="white", node_size=600, arrowsize=14)
ax.set_axis_off(); plt.show()
df = scm.sample(2000, seed=3)
df.describe().loc[["mean", "std"]]
''')

md("""
## Stage 2 — Causal ground truth

Two quantities, both computable only because we built the world:

* **Total interventional effect** $\\tau_j = \\mathbb{E}_x\\big|\\mathbb{E}[Y\\mid do(X_j = x+\\sigma_j)] - \\mathbb{E}[Y\\mid do(X_j = x-\\sigma_j)]\\big| / 2$,
  by nested Monte Carlo with common random numbers (noise averaged *before* the absolute value).
* **Exact interventional SHAP of the true function** $f^*(x) = \\mathbb{E}[Y \\mid \\text{pa}(Y)]$ on 100 fixed explain rows
  with 50 background rows — the attribution a *perfect* model and a *perfect* explainer would give.

In a chain every upstream feature truly moves $Y$, but $f^*$ only reads the parents of $Y$ — so exact SHAP gives
the mediated features **zero**. That is the *estimand gap*: SHAP is faithful to the model, not to the world.
""")
code('''
from src.scm.oracle import mean_abs_shap
tau = scm.true_effects(n_ref=50, n_noise_per_ref=5000, seed=3)
X = df[scm.features].to_numpy()
phi_star = mean_abs_shap(scm.f_star, X[:100], X[100:150])
shares = lambda v: np.abs(v) / np.abs(v).sum()
pd.DataFrame({"tau (true effect)": tau, "exact SHAP of f*": phi_star,
              "share tau": shares(tau), "share SHAP": shares(phi_star)}, index=scm.features).round(3)
''')

md("""
## Stage 3 — Fit models, run explainers

Both models are trained on the same 60/20/20 split; all explainers see the same 100 explain / 50 background rows
as the oracle. `exact` is brute-force interventional SHAP of the **fitted** model (all $2^d$ coalitions) — the $\\Phi(f)$ term.
""")
code('''
from src.models import train_model, split_xy
from src.explainers import explain, supports
n = len(df); split = np.array(["train"] * int(.6 * n) + ["val"] * int(.2 * n) + ["test"] * (n - int(.6 * n) - int(.2 * n)))
np.random.default_rng(0).shuffle(split); df["row_split"] = split
parts = split_xy(df, scm.features)
X_exp = df[df.row_split == "test"][scm.features].to_numpy()[:100]
X_bg = df[df.row_split == "train"][scm.features].to_numpy()[:50]
rows = {}
for name in ("xgboost", "mlp"):
    model = train_model(name, df, seed=0)
    print(f"{name}: test R^2 = {model.metrics['test_r2']:.3f}")
    for ex in ("exact", "treeshap", "kernelshap", "lime", "permutation"):
        if supports(model, ex):
            a = explain(model, ex, X_exp, X_bg, seed=0, X_train=parts["train"][0], X_test=parts["test"][0], y_test=parts["test"][1])
            rows[(name, ex)] = shares(a.mean_abs)
tab = pd.DataFrame(rows, index=scm.features).T
tab.insert(0, "", "")
pd.concat([pd.DataFrame([shares(tau), shares(phi_star)], index=[("truth", "tau"), ("truth", "exact SHAP f*")], columns=scm.features), tab.drop(columns="")]).round(3)
''')

md("""
## Stage 4a — Where does the error come from?

$\\tau \\to \\Phi(f^*) \\to \\Phi(f) \\to \\hat E(f)$: the three vector differences telescope exactly.
We report each component as an $L_2$ distance between **importance shares** (scale-free), as `1 - Kendall tau-b`
(rank space) and as normalized $L_2$ (raw space, as in SCHEMAS.md).
""")
code('''
from src.metrics import decompose_error
model = train_model("xgboost", df, seed=0)
exact = explain(model, "exact", X_exp, X_bg).mean_abs
for ex in ("kernelshap", "lime", "permutation"):
    emp = explain(model, ex, X_exp, X_bg, seed=0, X_train=parts["train"][0], X_test=parts["test"][0], y_test=parts["test"][1]).mean_abs
    r = decompose_error(tau, phi_star, exact, emp)
    print(f"{ex:12s} total(rank)={r['total_error_rank']:.2f}  total(share)={r['total_error_share_l2']:.3f}  "
          f"| gap={r['estimand_gap_share']:.3f}  misfit={r['model_misfit_share']:.3f}  explainer={r['explainer_error_share']:.3f}")
''')

md("""
## Stage 4b — Observables: what you *can* measure without the truth

Everything below is computed from the training data, the fitted model, held-out labels and the attributions themselves.
""")
code('''
from src.observables import data_observables, model_observables, concentration_observables, explainer_disagreement, perturb_inputs, perturbation_observables
obs = data_observables(parts["train"][0], parts["train"][1])
obs.update(model_observables(model, parts))
attrs = {ex: explain(model, ex, X_exp, X_bg, seed=0, X_train=parts["train"][0], X_test=parts["test"][0], y_test=parts["test"][1])
         for ex in ("kernelshap", "lime", "permutation")}
obs.update(concentration_observables(attrs["kernelshap"].values))
obs.update(explainer_disagreement("kernelshap", {k: a.mean_abs for k, a in attrs.items()}))
X_p = perturb_inputs(X_exp, parts["train"][0], 0.05, seed=1)
obs.update(perturbation_observables(attrs["kernelshap"].values, explain(model, "kernelshap", X_p, X_bg, seed=0).values, X_exp, X_p))
pd.Series(obs, name="KernelSHAP on XGBoost").to_frame().round(3).T
''')

md("""
## Stage 5 — Can the observables predict the error? (full 300-SCM run)

The full benchmark (`python tasks.py generate-full && python tasks.py pipeline-full`) produces one row per
(SCM, model, explainer, seed). We join observables to errors and ask a meta-model to predict the error.

Three protocols, in increasing difficulty:
1. **Grouped CV** — unseen SCMs from seen graph families (StratifiedGroupKFold over `scm_id`).
2. **Leave-one-graph-family-out (LOGFO)** — train on 4 families, test on the 5th, for every family.
3. **Designated split** — train on the four training families, test on the held-out *layered* family.

Baselines are what a practitioner could already use: max VIF, feature correlation, test error, sample size,
explainer disagreement, explainer stability, or a constant.
""")
code('''
RES = ROOT / "results" if (ROOT / "results" / "diagnostic_summary.json").exists() else ROOT / "results_small"
SCMS = ROOT / "data" / ("scms" if RES.name == "results" else "scms_small")
print("using", RES)
summary = json.load(open(RES / "diagnostic_summary.json"))
errors = pd.read_csv(RES / "errors.csv").merge(pd.read_csv(SCMS / "manifest.csv")[["scm_id", "family", "split", "d"]], on="scm_id")
observables = pd.read_csv(RES / "observables.csv")
print(f"{summary['n_scms']} SCMs, {summary['n_rows']} (scm, model, explainer, seed) rows, {summary['n_observables']} observables")
errors.groupby(["model", "explainer"])[["total_error_rank", "total_error_share_l2", "estimand_gap_share", "model_misfit_share", "explainer_error_share", "top_k_overlap"]].mean().round(3)
''')

md("**Estimand gap by graph family** — computed with the perfect model and exact SHAP, so it is pure definition mismatch.")
code('''
display(pd.read_csv(RES / "tables" / "estimand_gap_by_family.csv").round(3))
display(Image(filename=str(RES / "figures" / "fig_estimand_gap_by_family.png"), width=800))
display(Image(filename=str(RES / "figures" / "fig_error_decomposition_share.png"), width=800))
''')

md("""
**Diagnostic performance.** Pooled scores per protocol (Spearman between predicted and actual error; AUROC for flagging rows
above the training median). Two targets: the **share error** (how much importance mass is misallocated) and the **rank error**
(1 − Kendall τ-b on ≤ 10 features).
""")
code('''
scores = pd.read_csv(RES / "tables" / "diagnostic_scores.csv")
pooled = scores[(scores.fold == "pooled")].pivot_table(index=["target", "method"], columns="protocol", values=["spearman", "r2", "auroc_high_error"])
pooled.loc[["total_error_rank", "total_error_share_l2"]].round(3)
''')
code('''
display(Image(filename=str(RES / "figures" / "fig_logfo_spearman_total_error_share_l2.png"), width=850))
display(Image(filename=str(RES / "figures" / "fig_pred_vs_actual_designated_split_meta_gbm_total_error_share_l2.png"), width=420))
''')

md("""
**Which part of the error is predictable?** The same experiment with each component of the decomposition as the target.
The explainer's and the model's contributions are visible in the observables; the structural estimand gap is largely not —
and since the gap dominates, the total is only moderately predictable and the rank error hardly at all.
""")
code('''
display(Image(filename=str(RES / "figures" / "fig_component_predictability_logfo_spearman.png"), width=850))
comp = scores[(scores.fold == "pooled") & (scores.protocol == "logfo")].pivot_table(index="target", columns="method", values="spearman")
comp.loc[["explainer_error_share", "model_misfit_share", "estimand_gap_share", "total_error_share_l2", "total_error_rank"],
         ["meta_gbm", "meta_rf", "meta_ridge", "baseline_test_error", "baseline_disagreement_only", "baseline_stability_only", "baseline_max_vif"]].round(3)
''')

md("**Is the gain over the best baseline real?** Paired cluster bootstrap over SCMs (2000 resamples) and Wilcoxon signed-rank on per-SCM absolute errors (share-error target, LOGFO).")
code('''
tests = pd.read_csv(RES / "tables" / "paired_tests.csv")
cols = ["target", "protocol", "method_b", "mae_gain", "mae_gain_ci_lo", "mae_gain_ci_hi", "spearman_gain", "spearman_gain_ci_lo", "spearman_gain_ci_hi", "wilcoxon_p", "frac_scms_a_better"]
tests[(tests.target == "total_error_share_l2") & (tests.protocol == "logfo")][cols].round(3)
''')

md("**Which observables carry the signal?** Permutation importance of the meta-model trained on the four training families and scored on the held-out layered family.")
code('''
display(pd.read_csv(RES / "tables" / "feature_importance_total_error_share_l2.csv").head(12).round(4))
display(Image(filename=str(RES / "figures" / "fig_feature_importance_total_error_share_l2.png"), width=520))
display(Image(filename=str(RES / "figures" / "fig_observables_vs_total_error_share_l2.png"), width=850))
''')

md("**Stricter question:** for a *fixed* explainer and model, can the diagnostic still rank SCMs by error? (`by_method` rows of the scores table, LOGFO.)")
code('''
bm = scores[(scores.protocol == "logfo") & (scores.target == "total_error_share_l2") & scores.fold.str.startswith("by_method")]
bm = bm.assign(subset=bm.fold.str.replace("by_method:", "")).pivot_table(index="subset", columns="method", values="spearman")
bm[[c for c in ["meta_gbm", "meta_rf", "meta_ridge", "baseline_test_error", "baseline_disagreement_only", "baseline_stability_only", "baseline_max_vif"] if c in bm.columns]].round(3)
''')

md("""
**Re-running the diagnostic live.** The whole stage-5 evaluation is one function call; here is LOGFO for the GBM
meta-model and two baselines, recomputed in the notebook from the saved tables.
""")
code('''
from src.diagnostic import load_meta_table, run_methods, meta_model_factory, baseline_factory
meta = load_meta_table(RES / "errors.csv", RES / "observables.csv", SCMS / "manifest.csv")
methods = {"meta_gbm": meta_model_factory("gbm"), "baseline_test_error": baseline_factory("test_error"), "baseline_constant": baseline_factory("constant")}
sc, preds = run_methods(meta, "total_error_share_l2", methods, "logfo")
sc[~sc.fold.str.startswith("by_method")].pivot_table(index="fold", columns="method", values="spearman").round(3)
''')

md("""
## Takeaways

1. **Most explanation error is structural.** With a perfect model and exact SHAP, the attribution still ranks features
   differently from the causal effect (estimand gap), and the gap depends on the graph family. The explainer's own
   approximation error (KernelSHAP, LIME) is small by comparison; permutation importance is the noisiest.
2. **How much importance mass an explanation misallocates is predictable from observable signals** (Spearman ≈ 0.65 on
   unseen SCMs, ≈ 0.52 under leave-one-graph-family-out, AUROC ≈ 0.76), and the diagnostic transfers to graph families it
   never saw, beating every single-signal baseline (see `paired_tests.csv`).
3. **But the predictable part is the model's and the explainer's error** (ρ ≈ 0.70 and 0.91); the structural estimand gap is
   mostly invisible (ρ ≈ 0.34) and the feature *ranking* error hardly predictable (ρ ≈ 0.22). A diagnostic can warn about a
   bad model or explainer, not about SHAP asking a different question than "what if I intervene".
4. **The strongest signals** are attribution concentration (Gini, top-1 share, effective #features), feature–outcome
   association, model fit, and disagreement between explainers and between models; max VIF alone is nearly useless.
5. **Limitations.** Synthetic SCMs with $d \\le 10$, additive noise and one outcome; the diagnostic is calibrated
   to this benchmark's families and should be re-validated before use on real data (IHDP / Adult / California Housing
   checks are left as future work).
""")

nb["cells"] = cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print("wrote", OUT)
    if a.execute:
        root = OUT.parent.parent
        subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                        f"--ExecutePreprocessor.timeout={a.timeout}", str(OUT)], cwd=root, check=True)
        print("executed", OUT)


if __name__ == "__main__":
    main()
