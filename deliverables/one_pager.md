PROJECT ONE-PAGER

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

    300 SCMs, 60 per graph family (chain-mediation, confounded fork,
    feature collider, Erdos-Renyi DAG, layered = held out), d = 5..10 features,
    2000 observational rows each (60/20/20 split). Mechanisms: linear, tanh, sin,
    quadratic, ReLU, pairwise interactions; Gaussian noise; mean R^2 of the true
    function 0.67. Ground truth per SCM: total effect tau_j under do(X_j)
    (nested Monte Carlo, common random numbers) and exact interventional SHAP of
    the true function on 100 fixed explain rows. Released as data/scms/ (parquet +
    json) with a manifest.


Methodology

    Stage 1-2  SCM generation and ground truth (src/scm).
    Stage 3    XGBoost and MLP per SCM, 3 seeds; explainers: exact SHAP of the
               fitted model, TreeSHAP, KernelSHAP, LIME, permutation importance,
               all on the same explain/background rows (8100 explanations).
    Stage 4    Error = tau -> Phi(f*) -> Phi(f) -> E_hat, reported in raw, share
               (scale-free) and rank (1 - Kendall tau-b) space; 39 observables
               computed without tau or the graph.
    Stage 5    GBM / RF / ridge meta-models vs 7 baselines; grouped CV, LOGFO,
               designated held-out family; paired cluster bootstrap and Wilcoxon.


Results

    Where the error comes from (share-space L2, mean over SCMs and seeds):
      estimand gap 0.31, model misfit 0.12,
      explainer error KernelSHAP 0.00, LIME 0.06,
      permutation 0.16.
      Rank error vs tau: exact SHAP 0.43, KernelSHAP 0.43,
      LIME 0.45, permutation 0.43.
    Estimand gap (rank) by family: chain 0.37, fork 0.34,
      collider 0.34, random DAG 0.26, layered 0.38.
    In chain worlds, 38% of the true effect sits on mediated features;
      exact SHAP of the true function gives them 0%, LIME 11%.
    Diagnostic (target = share error, meta-model = gradient boosting):
      LOGFO pooled Spearman 0.52, AUROC 0.76; best single-signal
      baseline (test error) 0.36. Per held-out family Spearman / AUROC:
      chain_mediation 0.72/0.84 / confounded_fork 0.29/0.69 / erdos_renyi_dag 0.42/0.77 / feature_collider 0.81/0.88 / layered 0.48/0.75.
      Grouped CV (unseen SCMs, seen families) Spearman 0.65, R^2 0.53.
      Held-out layered family: Spearman 0.48 (ties the test-error baseline 0.46).
      Gain over best baseline (LOGFO): delta Spearman 0.17
      [0.08, 0.25], Wilcoxon p < 0.001,
      better on 65% of SCMs.
      Rank error (1 - Kendall tau-b) is only weakly predictable: Spearman 0.22 (LOGFO).
    Which part is predictable (LOGFO Spearman per component): explainer error 0.91
      (from disagreement_l2_mean, explainer_lime, sign_consistency), model misfit 0.70
      (from cross_model_disagreement_l2, model_mlp, top1_share), estimand gap 0.34
      (from mean_abs_corr_xy, mi_xy_mean, attribution_gini).
    Most informative observables: mean_abs_corr_xy, attribution_gini, mi_xy_mean, mi_xy_max, disagreement_l2_mean.


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
