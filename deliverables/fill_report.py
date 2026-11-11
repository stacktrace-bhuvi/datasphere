"""Generate the LaTeX tables + figure includes of the report from deliverables/numbers.json.

    python deliverables/fill_report.py --numbers deliverables/numbers.json --results-dir results --report-dir report
Writes report/tables.tex (all tables as macros) and copies the needed figures into report/figures/.
The prose (abstract, results narrative, discussion, conclusion) lives in report/*.tex and is written by hand.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

FAM = {"chain_mediation": "Chain", "confounded_fork": "Fork", "feature_collider": "Collider",
       "erdos_renyi_dag": "Random DAG", "layered": "Layered$^\\dagger$"}
EXP = {"exact": "Exact SHAP", "treeshap": "TreeSHAP", "kernelshap": "KernelSHAP", "lime": "LIME", "permutation": "Permutation"}
METHOD = {"meta_gbm": "Meta: GBM", "meta_rf": "Meta: RF", "meta_ridge": "Meta: ridge",
          "baseline_constant": "Constant", "baseline_correlation_only": "Correlation only", "baseline_max_vif": "Max VIF",
          "baseline_test_error": "Test error", "baseline_sample_size": "Sample size",
          "baseline_disagreement_only": "Disagreement only", "baseline_stability_only": "Stability only"}
HEAD = "total_error_share_l2"
FIGS = ["fig_error_decomposition_share", "fig_estimand_gap_by_family", f"fig_logfo_spearman_{HEAD}",
        f"fig_pred_vs_actual_designated_split_meta_gbm_{HEAD}", f"fig_feature_importance_{HEAD}",
        f"fig_observables_vs_{HEAD}", f"fig_total_error_by_explainer_{HEAD}", f"fig_logfo_auroc_high_error_{HEAD}",
        "fig_component_predictability_logfo_spearman", "fig_logfo_spearman_total_error_rank"]


def f(x, d=2):
    if x is None:
        return "--"
    v = float(x)
    if abs(v) < 0.5 * 10 ** (-d):
        v = 0.0
    return f"{v:.{d}f}"


def esc(s: str) -> str:
    return s.replace("_", "\\_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--numbers", default="deliverables/numbers.json")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--report-dir", default="report")
    a = ap.parse_args()
    N = json.load(open(a.numbers))
    rep = Path(a.report_dir)
    (rep / "figures").mkdir(parents=True, exist_ok=True)
    for name in FIGS:
        src = Path(a.results_dir) / "figures" / f"{name}.pdf"
        if src.exists():
            shutil.copy(src, rep / "figures" / f"{name}.pdf")

    out = []
    # ---- Table: decomposition by model x explainer (share + rank)
    dme = N["decomposition_by_model_explainer"]
    out.append(r"""\newcommand{\TableDecomposition}{%
\begin{table}[t]
\caption{The three parts of the error, averaged over SCMs and seeds. Share: $L_2$ distance between importance shares. Rank: $1-$Kendall $\tau_b$. Top-$k$: fraction of true top features found.}
\label{tab:decomp}
\centering\footnotesize
\setlength{\tabcolsep}{2pt}
\begin{tabular}{llcccccc}
\toprule
& & \multicolumn{4}{c}{Share space} & Rank & Top-$k$ \\
\cmidrule(lr){3-6}
Model & Explainer & Gap & Misfit & Expl. & Total & Total & \\
\midrule""")
    for model in ("xgboost", "mlp"):
        first = True
        for ex in ("exact", "treeshap", "kernelshap", "lime", "permutation"):
            k = f"{model}/{ex}"
            if k not in dme:
                continue
            r = dme[k]
            out.append(f"{'XGBoost' if model == 'xgboost' else 'MLP' if first else ''} & {EXP[ex]} & {f(r['estimand_gap_share'])} & {f(r['model_misfit_share'])} & {f(r['explainer_error_share'])} & {f(r['total_error_share_l2'])} & {f(r['total_error_rank'])} & {f(r['top_k_overlap'])} \\\\")
            first = False
        if model == "xgboost":
            out.append(r"\midrule")
    out.append(r"""\bottomrule
\end{tabular}
\end{table}}""")

    # ---- Table: estimand gap by family
    g = N["estimand_gap_by_family"]
    ebf = N["error_by_family_explainer"]
    out.append(r"""\newcommand{\TableGapByFamily}{%
\begin{table}[t]
\caption{Estimand gap for each graph family (perfect model, exact SHAP), mean $\pm$ standard deviation over SCMs, and total rank error of KernelSHAP and LIME on the trained models. $^\dagger$held-out family.}
\label{tab:gap}
\centering\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccc}
\toprule
Family & Gap (rank) & Gap (share) & KernelSHAP & LIME \\
\midrule""")
    for fam, lab in FAM.items():
        if fam in g:
            out.append(f"{lab} & {f(g[fam]['rank_mean'])} $\\pm$ {f(g[fam]['rank_sd'])} & {f(g[fam]['share_mean'])} $\\pm$ {f(g[fam]['share_sd'])} & {f(ebf.get(fam + '/kernelshap', {}).get('rank'))} & {f(ebf.get(fam + '/lime', {}).get('rank'))} \\\\")
    out.append(r"""\bottomrule
\end{tabular}
\end{table}}""")

    # ---- Table: diagnostic performance (pooled) per protocol
    dp = N["diagnostic_pooled"]
    for tgt, macro in (("total_error_rank", "TableDiagnosticRank"), ("total_error_share_l2", "TableDiagnosticShare")):
        if tgt not in dp:
            continue
        tl = "rank error ($1-\\tau_b$)" if tgt == "total_error_rank" else "share error"
        out.append(f"""\\newcommand{{\\{macro}}}{{%
\\begin{{table*}}[t]
\\caption{{Predicting the {tl} of an explanation from observables: Spearman $\\rho$ (predicted vs actual), $R^2$, and AUROC for detecting explanations with above-median error, for the three evaluation setups.}}
\\label{{tab:diag_{'rank' if tgt == 'total_error_rank' else 'share'}}}
\\centering\\small
\\setlength{{\\tabcolsep}}{{6pt}}
\\begin{{tabular}}{{lccccccccc}}
\\toprule
& \\multicolumn{{3}}{{c}}{{Grouped CV}} & \\multicolumn{{3}}{{c}}{{LOGFO}} & \\multicolumn{{3}}{{c}}{{Held-out layered}} \\\\
\\cmidrule(lr){{2-4}}\\cmidrule(lr){{5-7}}\\cmidrule(lr){{8-10}}
Method & $\\rho$ & $R^2$ & AUC & $\\rho$ & $R^2$ & AUC & $\\rho$ & $R^2$ & AUC \\\\
\\midrule""")
        for m, lab in METHOD.items():
            cells = []
            for proto in ("grouped_cv", "logfo", "designated_split"):
                r = dp[tgt].get(proto, {}).get(m)
                cells += [f(r["spearman"]), f(r["r2"]), f(r["auroc_high_error"])] if r else ["--"] * 3
            bold = m == "meta_gbm"
            lab_ = f"\\textbf{{{lab}}}" if bold else lab
            out.append(f"{lab_} & " + " & ".join(cells) + " \\\\")
            if m == "meta_ridge":
                out.append(r"\midrule")
        out.append(r"""\bottomrule
\end{tabular}
\end{table*}}""")

    # ---- Table: LOGFO per family, meta vs baselines
    lf = N["logfo_by_family"].get(HEAD, {})
    la = N["logfo_by_family_auroc"].get(HEAD, {})
    out.append(r"""\newcommand{\TableLogfoFamily}{%
\begin{table}[t]
\caption{LOGFO, share error: Spearman $\rho$ (AUROC in brackets) for each held-out family. Each column uses a model trained on the other four families.}
\label{tab:logfo}
\centering\scriptsize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{lccccc}
\toprule
Method & Chain & Fork & Collider & Random & Layered \\
\midrule""")
    for m, lab in METHOD.items():
        if m not in lf:
            continue
        r, ra = lf[m], la.get(m, {})
        out.append(f"{lab} & " + " & ".join(f"{f(r.get(fam))} ({f(ra.get(fam))})" for fam in FAM) + " \\\\")
        if m == "meta_ridge":
            out.append(r"\midrule")
    out.append(r"""\bottomrule
\end{tabular}
\end{table}}""")

    # ---- Table: paired tests (LOGFO and designated), meta_gbm vs each baseline
    pt = [r for r in N["paired_tests"] if r["target"] == HEAD and r["method_a"] == "meta_gbm"]
    out.append(r"""\newcommand{\TablePairedTests}{%
\begin{table}[t]
\caption{Improvement of the GBM meta-model over each baseline (share error). Bootstrap over SCMs (2000 samples): average improvement in Spearman $\rho$ and in MAE with 95\% interval; Wilcoxon test $p$-value on per-SCM errors; percentage of SCMs where the meta-model has the lower error.}
\label{tab:tests}
\centering\footnotesize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{llcccc}
\toprule
Protocol & vs.\ & $\Delta\rho$ [95\% CI] & $\Delta$MAE [95\% CI] & $p_{W}$ & better \\
\midrule""")
    for proto, plab in (("logfo", "LOGFO"), ("designated_split", "Layered")):
        rows = [r for r in pt if r["protocol"] == proto]
        first = True
        for r in rows:
            if r["method_b"].startswith("meta_"):
                continue
            pw = r.get("wilcoxon_p")
            pws = "--" if pw is None else ("$<$0.001" if pw < 0.001 else f"{pw:.3f}")
            fb = r.get("frac_scms_a_better")
            out.append(f"{plab if first else ''} & {METHOD[r['method_b']]} & {f(r['spearman_gain'])} [{f(r['spearman_gain_ci_lo'])}, {f(r['spearman_gain_ci_hi'])}] & {f(r['mae_gain'], 3)} [{f(r['mae_gain_ci_lo'], 3)}, {f(r['mae_gain_ci_hi'], 3)}] & {pws} & {'--' if fb is None else f'{100 * fb:.0f}\\%'} \\\\")
            first = False
        if proto == "logfo":
            out.append(r"\midrule")
    out.append(r"""\bottomrule
\end{tabular}
\end{table}}""")

    # ---- Table: feature importance
    fi = N["feature_importance"].get(HEAD, [])[:10]
    out.append(r"""\newcommand{\TableFeatureImportance}{%
\begin{table}[t]
\caption{Most important observables in the GBM meta-model: drop in $R^2$ on the layered family when the column is shuffled (mean $\pm$ standard deviation over 10 repeats).}
\label{tab:fi}
\centering\small
\begin{tabular}{lc}
\toprule
Observable & $\Delta R^2$ \\
\midrule""")
    for r in fi:
        out.append(f"\\texttt{{{esc(r['feature'])}}} & {f(r['importance_mean'], 3)} $\\pm$ {f(r['importance_std'], 3)} \\\\")
    out.append(r"""\bottomrule
\end{tabular}
\end{table}}""")

    # ---- Table: by-method LOGFO spearman (meta vs best baseline)
    bm = N["logfo_by_method"].get(HEAD, {})
    best = N["diagnostic"][HEAD]["logfo"]["best_baseline"]
    out.append(r"""\newcommand{\TableByMethod}{%
\begin{table}[t]
\caption{Spearman $\rho$ between predicted and actual share error within each (model, explainer) pair under LOGFO, i.e.\ ranking SCMs when the explanation method is fixed.}
\label{tab:bymethod}
\centering\small
\begin{tabular}{lcc}
\toprule
Subset & Meta: GBM & """ + METHOD.get(best, best) + r""" \\
\midrule""")
    for k in sorted(bm.get("meta_gbm", {})):
        out.append(f"{esc(k)} & {f(bm['meta_gbm'][k])} & {f(bm.get(best, {}).get(k))} \\\\")
    out.append(r"""\bottomrule
\end{tabular}
\end{table}}""")

    # ---- Table: component predictability (LOGFO pooled)
    dp = N["diagnostic_pooled"]
    comps = [("explainer_error_share", "Explainer error"), ("model_misfit_share", "Model misfit"),
             ("estimand_gap_share", "Estimand gap"), ("total_error_share_l2", "Total (share)"), ("total_error_rank", "Total (rank)")]
    meths = ["meta_gbm", "baseline_test_error", "baseline_disagreement_only", "baseline_stability_only", "baseline_max_vif"]
    out.append(r"""\newcommand{\TableComponents}{%
\begin{table}[t]
\caption{How well each part of the error can be predicted: LOGFO Spearman $\rho$ ($R^2$ in brackets) when each part of~\eqref{eq:decomp}, and the two totals, is the target.}
\label{tab:components}
\centering\footnotesize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{lccccc}
\toprule
Target & Meta: GBM & Test error & Disagreement & Stability & Max VIF \\
\midrule""")
    for tgt, lab in comps:
        cells = []
        for m in meths:
            r = dp.get(tgt, {}).get("logfo", {}).get(m)
            cells.append(f"{f(r['spearman'])} ({f(r['r2'])})" if r else "--")
        out.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    out.append(r"""\bottomrule
\end{tabular}
\end{table}}""")

    (rep / "tables.tex").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("wrote", rep / "tables.tex")


if __name__ == "__main__":
    main()
