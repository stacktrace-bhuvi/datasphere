"""Render the one-pager and the video script from deliverables/numbers.json.

    python deliverables/make_text_deliverables.py --numbers deliverables/numbers.json --out deliverables
Writes one_pager.md, one_pager.docx, video_script.md (plain, minimally formatted).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def f2(x):
    return f"{float(x):.2f}"


def pct(x, d=0):
    return f"{100 * float(x):.{d}f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--numbers", default="deliverables/numbers.json")
    ap.add_argument("--out", default="deliverables")
    a = ap.parse_args()
    N = json.load(open(a.numbers))
    B, D = N["benchmark"], N["diagnostic"]
    t = "total_error_share_l2"
    lo, de, cv = D[t]["logfo"], D[t]["designated_split"], D[t]["grouped_cv"]
    rk = D["total_error_rank"]
    comp = {k: N["diagnostic_pooled"][k]["logfo"]["meta_gbm"]["spearman"] for k in
            ("explainer_error_share", "model_misfit_share", "estimand_gap_share")}
    fitop = {k: ", ".join(r["feature"] for r in N["feature_importance"].get(k, [])[:3]) for k in comp}
    famau = N["logfo_by_family_auroc"][t]["meta_gbm"]
    vs = lo.get("vs_best_baseline", {})
    dec = N["decomposition_by_explainer"]
    gap = N["estimand_gap_by_family"]
    chain = N.get("mediated_share", {}).get("chain_mediation", {})
    best = lo["best_baseline"].replace("baseline_", "").replace("_", " ")
    fi = [r["feature"] for r in N["feature_importance"].get(t, [])[:5]]
    famsp = lo.get("meta_spearman_by_fold", {})
    r2star = N.get("oracle_r2", {}).get("mean")
    p_w = vs.get("wilcoxon_p")
    p_str = "< 0.001" if p_w is not None and p_w < 0.001 else (f"{p_w:.3f}" if p_w is not None else "n/a")

    one_pager = f"""PROJECT ONE-PAGER

Title        Predicting Explanation Error Without Causal Ground Truth:
             A Held-Out-SCM Study of SHAP and LIME
Course       Advanced Data Analytics (Sem 5), PES University, Section G
Team         DataSphere -- Bhuvi S. Katti (PES1UG24AM418), Mythri S (PES1UG24AM419),
             Poojitha L. B. R. (PES1UG24AM384), Diran Somanna (PES1UG24AM364)
Code         https://github.com/stacktrace-bhuvi/datasphere


Problem statement

    SHAP and LIME are read causally ("this feature drives the outcome") but they
    describe the fitted model, not the data-generating process. With real data
    the causal truth is unknown, so a misleading explanation cannot be detected.
    Question: can the size of an explanation's error be predicted from quantities
    that are observable without causal ground truth, and does such a diagnostic
    generalise to causal structures it has never seen?


Objectives

    1. Build a benchmark of synthetic causal worlds (SCMs) with exact causal
       ground truth and oracle Shapley values.
    2. Measure the error of SHAP, LIME and permutation importance against the
       total interventional effect, and decompose it into estimand gap, model
       misfit and explainer approximation error.
    3. Extract ground-truth-free observables (dependence, fit, stability,
       disagreement) and train a diagnostic meta-model to predict the error.
    4. Evaluate under grouped CV and leave-one-graph-family-out (LOGFO) against
       single-signal baselines, with paired statistical tests.


Dataset (generated, synthetic)

    {B['n_scms']} SCMs, 60 per graph family (chain-mediation, confounded fork,
    feature collider, Erdos-Renyi DAG, layered = held out), d = 5..10 features,
    2000 observational rows each (60/20/20 split). Mechanisms: linear, tanh, sin,
    quadratic, ReLU, pairwise interactions; Gaussian noise; mean R^2 of the true
    function {f2(r2star) if r2star else 'n/a'}. Ground truth per SCM: total effect tau_j under do(X_j)
    (nested Monte Carlo, common random numbers) and exact interventional SHAP of
    the true function on 100 fixed explain rows. Released as data/scms/ (parquet +
    json) with a manifest.


Methodology

    Stage 1-2  SCM generation and ground truth (src/scm).
    Stage 3    XGBoost and MLP per SCM, 3 seeds; explainers: exact SHAP of the
               fitted model, TreeSHAP, KernelSHAP, LIME, permutation importance,
               all on the same explain/background rows ({B['n_attribution_rows']} explanations).
    Stage 4    Error = tau -> Phi(f*) -> Phi(f) -> E_hat, reported in raw, share
               (scale-free) and rank (1 - Kendall tau-b) space; {B['n_observables']} observables
               computed without tau or the graph.
    Stage 5    GBM / RF / ridge meta-models vs 7 baselines; grouped CV, LOGFO,
               designated held-out family; paired cluster bootstrap and Wilcoxon.


Results

    Where the error comes from (share-space L2, mean over SCMs and seeds):
      estimand gap {f2(dec['exact']['estimand_gap_share'])}, model misfit {f2(dec['exact']['model_misfit_share'])},
      explainer error KernelSHAP {f2(dec['kernelshap']['explainer_error_share'])}, LIME {f2(dec['lime']['explainer_error_share'])},
      permutation {f2(dec['permutation']['explainer_error_share'])}.
      Rank error vs tau: exact SHAP {f2(dec['exact']['total_error_rank'])}, KernelSHAP {f2(dec['kernelshap']['total_error_rank'])},
      LIME {f2(dec['lime']['total_error_rank'])}, permutation {f2(dec['permutation']['total_error_rank'])}.
    Estimand gap (rank) by family: chain {f2(gap['chain_mediation']['rank_mean'])}, fork {f2(gap['confounded_fork']['rank_mean'])},
      collider {f2(gap['feature_collider']['rank_mean'])}, random DAG {f2(gap['erdos_renyi_dag']['rank_mean'])}, layered {f2(gap['layered']['rank_mean'])}.
    In chain worlds, {pct(chain.get('tau', 0))} of the true effect sits on mediated features;
      exact SHAP of the true function gives them {pct(chain.get('oracle', 0))}, LIME {pct(chain.get('lime', 0))}.
    Diagnostic (target = share error, meta-model = gradient boosting):
      LOGFO pooled Spearman {f2(lo['meta_spearman'])}, AUROC {f2(lo['meta_auroc_high_error'])}; best single-signal
      baseline ({best}) {f2(lo['best_baseline_spearman'])}. Per held-out family Spearman / AUROC:
      {' / '.join(f'{k} {f2(v)}/{f2(famau[k])}' for k, v in famsp.items())}.
      Grouped CV (unseen SCMs, seen families) Spearman {f2(cv['meta_spearman'])}, R^2 {f2(cv['meta_r2'])}.
      Held-out layered family: Spearman {f2(de['meta_spearman'])} (ties the test-error baseline {f2(de['best_baseline_spearman'])}).
      Gain over best baseline (LOGFO): delta Spearman {f2(vs.get('spearman_gain', 0))}
      [{f2(vs.get('spearman_gain_ci_lo', 0))}, {f2(vs.get('spearman_gain_ci_hi', 0))}], Wilcoxon p {p_str},
      better on {pct(vs.get('frac_scms_a_better', 0))} of SCMs.
      Rank error (1 - Kendall tau-b) is only weakly predictable: Spearman {f2(rk['logfo']['meta_spearman'])} (LOGFO).
    Which part is predictable (LOGFO Spearman per component): explainer error {f2(comp['explainer_error_share'])}
      (from {fitop['explainer_error_share']}), model misfit {f2(comp['model_misfit_share'])}
      (from {fitop['model_misfit_share']}), estimand gap {f2(comp['estimand_gap_share'])}
      (from {fitop['estimand_gap_share']}).
    Most informative observables: {', '.join(fi)}.


Conclusion

    Most explanation error is structural: even exact SHAP of a perfect model
    ranks features differently from the causal effect, and the gap depends on
    the graph family. How much importance mass an explanation misallocates is
    predictable from ground-truth-free signals and the diagnostic transfers to
    unseen graph families, beating every single-signal baseline. But the
    predictable part is the model's and the explainer's error; the dominant
    structural gap is mostly invisible to the observables. A diagnostic can warn
    about a bad model or explainer, not about SHAP answering a different
    question than "what if I intervene". Limits: synthetic SCMs with d <= 10;
    the diagnostic flags risk, it does not repair the explanation.


References

    Lundberg & Lee, NeurIPS 2017.  Ribeiro et al., KDD 2016.  Janzing et al.,
    AISTATS 2020.  Agarwal et al., NeurIPS 2022 (OpenXAI).  Krishna et al., 2022
    (disagreement problem).  Pearl, Causality, 2009.
"""

    video = f"""VIDEO SCRIPT  (target 4 minutes; ~150 words/min)
Speakers: B = Bhuvi, M = Mythri, P = Poojitha, D = Diran. Slide numbers refer to
DataSphere_Final_Presentation.pptx.


[0:00  slide 1 - title]  B
Hi, we are Team DataSphere. Our project asks a simple question: when SHAP or
LIME tells you which features matter, can you tell whether it is misleading
you, without knowing the true causes?


[0:15  slide 2 - problem]  B
People read attributions causally: "this feature drives the outcome". But SHAP
and LIME only describe what the model uses. In real data the causal truth is
unknown, so a misleading explanation looks exactly like a faithful one. Our
idea: build synthetic worlds where the truth is known, measure the error, and
learn to predict it from signals you can observe anyway.


[0:45  slides 3-4 - approach and benchmark]  B
We generated {B['n_scms']} structural causal models from five graph families: chains,
confounded forks, colliders, random DAGs and layered graphs, with five to ten
features and linear and non-linear mechanisms. For every feature we computed
the true total interventional effect under do-intervention, and the exact
Shapley values of the true function, the attribution a perfect model and a
perfect explainer would give.


[1:20  slide 5 - models and explainers]  M
On every world we trained XGBoost and an MLP with three seeds, and ran five
explainers on the same rows: brute-force exact SHAP of the fitted model,
TreeSHAP, KernelSHAP, LIME and permutation importance. That is {B['n_attribution_rows']}
explanations, each scored against the causal truth.


[1:40  slide 5-6 - decomposition and result 1]  P
We split each explanation's error into three parts. The estimand gap: truth
versus exact SHAP of the true function, what remains with a perfect model.
Model misfit: the fitted model learned a different surface. And explainer
error: KernelSHAP sampling, LIME's surrogate, or shuffling. The headline: most
of the error is structural. In share terms the estimand gap is {f2(dec['exact']['estimand_gap_share'])},
model misfit {f2(dec['exact']['model_misfit_share'])}, and KernelSHAP's own approximation error only
{f2(dec['kernelshap']['explainer_error_share'])}. Better SHAP estimators will not fix this.


[2:20  slide 7 - result 2]  P
And the gap depends on the causal structure. In chain worlds, {pct(chain.get('tau', 0))} of the
true effect sits on features that act through other features, and exact SHAP
gives them {pct(chain.get('oracle', 0))}. The gap is largest for chains and layered graphs, smallest
for random DAGs with many direct parents.


[2:45  slides 8-9 - observables and protocol]  D
So can we predict that error without the truth? We computed {B['n_observables']} observables
per explanation: feature dependence and VIF, model fit, attribution
concentration, stability across seeds and under input perturbation, and
disagreement between explainers and between models. None of them sees the
causal effect or the graph. A gradient-boosting meta-model predicts the rank
error from these, and we test it with grouped cross-validation and
leave-one-graph-family-out, against seven single-signal baselines.


[3:15  slides 10-12 - results 3, 4 and 5]  D
Under leave-one-family-out the meta-model reaches a Spearman correlation of
{f2(lo['meta_spearman'])} with the actual share error and an AUROC of {f2(lo['meta_auroc_high_error'])} for flagging
high-error explanations, against {f2(lo['best_baseline_spearman'])} for the best single-signal baseline,
and the gain is significant under a paired bootstrap over SCMs and a Wilcoxon
test. But when we predict each component separately, the picture is sharper:
explainer error is almost perfectly predictable, {f2(comp['explainer_error_share'])}, model misfit {f2(comp['model_misfit_share'])},
and the estimand gap only {f2(comp['estimand_gap_share'])}. The error you can see is the error you can
fix. The structural gap, the biggest piece, is mostly invisible, because
nothing in the data says which correlated inputs are causes and which are
mediators or proxies.


[3:45  slides 12-13 - limits and conclusion]  B
Limitations: synthetic worlds with at most ten features, one causal estimand,
and the diagnostic flags risk, it does not repair the explanation. Our
takeaway: SHAP is faithful to the model, not to the world. That gap is large
and structural; a diagnostic can tell you when your model or your explainer
is the problem, but not when SHAP is simply answering a different question.
Code, data, notebook and report are on GitHub. Thank you.
"""

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "one_pager.md").write_text(one_pager, encoding="utf-8")
    (out / "video_script.md").write_text(video, encoding="utf-8")
    try:
        from docx import Document
        from docx.shared import Pt
        for name, text in (("one_pager", one_pager), ("video_script", video)):
            doc = Document()
            st = doc.styles["Normal"]
            st.font.name = "Courier New"
            st.font.size = Pt(9.5)
            for line in text.splitlines():
                p = doc.add_paragraph(line)
                p.paragraph_format.space_after = Pt(0)
            doc.save(out / f"{name}.docx")
    except ImportError:
        pass
    print("wrote", out / "one_pager.md", out / "video_script.md")


if __name__ == "__main__":
    main()
