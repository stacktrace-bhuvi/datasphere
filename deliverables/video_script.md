VIDEO SCRIPT  (target 4 minutes; ~150 words/min)
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
We generated 300 structural causal models from five graph families: chains,
confounded forks, colliders, random DAGs and layered graphs, with five to ten
features and linear and non-linear mechanisms. For every feature we computed
the true total interventional effect under do-intervention, and the exact
Shapley values of the true function, the attribution a perfect model and a
perfect explainer would give.


[1:20  slide 5 - models and explainers]  M
On every world we trained XGBoost and an MLP with three seeds, and ran five
explainers on the same rows: brute-force exact SHAP of the fitted model,
TreeSHAP, KernelSHAP, LIME and permutation importance. That is 8100
explanations, each scored against the causal truth.


[1:40  slide 5-6 - decomposition and result 1]  P
We split each explanation's error into three parts. The estimand gap: truth
versus exact SHAP of the true function, what remains with a perfect model.
Model misfit: the fitted model learned a different surface. And explainer
error: KernelSHAP sampling, LIME's surrogate, or shuffling. The headline: most
of the error is structural. In share terms the estimand gap is 0.31,
model misfit 0.12, and KernelSHAP's own approximation error only
0.00. Better SHAP estimators will not fix this.


[2:20  slide 7 - result 2]  P
And the gap depends on the causal structure. In chain worlds, 38% of the
true effect sits on features that act through other features, and exact SHAP
gives them 0%. The gap is largest for chains and layered graphs, smallest
for random DAGs with many direct parents.


[2:45  slides 8-9 - observables and protocol]  D
So can we predict that error without the truth? We computed 39 observables
per explanation: feature dependence and VIF, model fit, attribution
concentration, stability across seeds and under input perturbation, and
disagreement between explainers and between models. None of them sees the
causal effect or the graph. A gradient-boosting meta-model predicts the rank
error from these, and we test it with grouped cross-validation and
leave-one-graph-family-out, against seven single-signal baselines.


[3:15  slides 10-12 - results 3, 4 and 5]  D
Under leave-one-family-out the meta-model reaches a Spearman correlation of
0.52 with the actual share error and an AUROC of 0.76 for flagging
high-error explanations, against 0.36 for the best single-signal baseline,
and the gain is significant under a paired bootstrap over SCMs and a Wilcoxon
test. But when we predict each component separately, the picture is sharper:
explainer error is almost perfectly predictable, 0.91, model misfit 0.70,
and the estimand gap only 0.34. The error you can see is the error you can
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
