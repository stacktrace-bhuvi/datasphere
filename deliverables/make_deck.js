// Build the final project deck from deliverables/numbers.json and results/figures.
//   node deliverables/make_deck.js [numbers.json] [figures_dir] [out.pptx]
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.PPTX_SKILL_DIR + "/scripts/apply_theme.js");

const [numbersPath, figDir, outPath] = [
  process.argv[2] || "deliverables/numbers.json",
  process.argv[3] || "results/figures",
  process.argv[4] || "deliverables/DataSphere_Final_Presentation.pptx",
];
const N = JSON.parse(fs.readFileSync(numbersPath, "utf8"));
const fig = (name) => path.join(figDir, name);
const pct = (x, d = 0) => `${(100 * x).toFixed(d)}%`;
const f2 = (x) => Number(x).toFixed(2);
const f3 = (x) => Number(x).toFixed(3);

// ------------------------------------------------------------------ theme
const THEME = {
  name: "DataSphere",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2432", lt1: "FFFFFF", dk2: "52514E", lt2: "F3F4F6",
    accent1: "2A78D6", accent2: "EB6834", accent3: "1BAF7A", accent4: "EDA100",
    accent5: "E87BA4", accent6: "6250D6", hlink: "2A78D6", folHlink: "6250D6",
  },
};
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = "Team DataSphere";
pres.title = "Predicting Explanation Error Without Causal Ground Truth";
const C = pres.SchemeColor;
const FOOT = "Team DataSphere · Advanced Data Analytics · PES University";

// ------------------------------------------------------------------ layouts
pres.defineSlideMaster({
  title: "DARK", background: { color: THEME.colors.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.5, w: 8.8, h: 1.6, fontSize: 34, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 3.25, w: 8.8, h: 1.6, fontSize: 16, color: "CADCFC", valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "LIGHT", background: { color: THEME.colors.lt1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9.0, h: 0.75, fontSize: 20, bold: true, color: C.text1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: FOOT, options: { x: 0.5, y: 5.25, w: 7.5, h: 0.3, fontSize: 8, color: C.text2, margin: 0 } } },
  ],
  slideNumber: { x: 9.2, y: 5.25, w: 0.4, h: 0.3, fontSize: 8, color: THEME.colors.dk2 },
});

// helpers ------------------------------------------------------------
function statTile(slide, x, y, w, big, small, color, labelColor) {
  slide.addText(big, { x, y, w, h: 0.7, fontSize: 30, bold: true, color, isTextBox: true, margin: 0, fontFace: "Cambria" });
  slide.addText(small, { x, y: y + 0.68, w, h: 0.6, fontSize: 10, color: labelColor || C.text2, isTextBox: true, margin: 0, valign: "top" });
}
function card(slide, x, y, w, h, head, body, opts = {}) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: opts.fill || THEME.colors.lt2 }, line: { color: THEME.colors.lt2 }, rectRadius: 0.08 });
  slide.addText(head, { x: x + 0.15, y: y + 0.1, w: w - 0.3, h: 0.35, fontSize: 12.5, bold: true, color: opts.headColor || C.text1, isTextBox: true, margin: 0 });
  slide.addText(body, { x: x + 0.15, y: y + 0.45, w: w - 0.3, h: h - 0.55, fontSize: 10.5, color: C.text1, isTextBox: true, margin: 0, valign: "top", paraSpaceAfter: 3 });
}
function bullets(items, fontSize = 12) {
  return items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, fontSize, paraSpaceAfter: 5 } }));
}
function numberedRow(slide, x, y, w, n, head, body, color) {
  slide.addShape(pres.shapes.OVAL, { x, y: y + 0.03, w: 0.34, h: 0.34, fill: { color }, line: { color } });
  slide.addText(String(n), { x, y: y + 0.03, w: 0.34, h: 0.34, fontSize: 11, bold: true, color: C.background1, align: "center", valign: "middle", isTextBox: true, margin: 0 });
  slide.addText([{ text: head, options: { bold: true, fontSize: 12, breakLine: true } }, { text: body, options: { fontSize: 10.5, color: C.text2 } }],
    { x: x + 0.45, y, w: w - 0.45, h: 0.75, isTextBox: true, margin: 0, valign: "top" });
}

// ------------------------------------------------------------------ numbers
const B = N.benchmark;
const D = N.diagnostic;
const tgt = "total_error_share_l2";
const logfo = D[tgt].logfo, desig = D[tgt].designated_split, gcv = D[tgt].grouped_cv;
const rank = D["total_error_rank"];
const pooledSp = (t, proto, m) => N.diagnostic_pooled[t][proto][m].spearman;
const chain = N.mediated_share ? N.mediated_share.chain_mediation : null;
const dec = N.decomposition_by_explainer;
const gapFam = N.estimand_gap_by_family;
const famLabel = { chain_mediation: "Chain", confounded_fork: "Fork", feature_collider: "Collider", erdos_renyi_dag: "Random DAG", layered: "Layered*" };
const bestBase = logfo.best_baseline.replace("baseline_", "").replace(/_/g, " ");
const vsb = logfo.vs_best_baseline || {};
const fi = (N.feature_importance[tgt] || []).slice(0, 6);
const oracleR2 = N.oracle_r2 ? N.oracle_r2.mean : null;

// ================================================================== slides
pres.addSection({ title: "Opening" });
let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Opening" });
s.addText("Predicting Explanation Error Without Causal Ground Truth", { placeholder: "title" });
s.addText([
  { text: "A held-out-SCM study of SHAP and LIME", options: { fontSize: 18, color: "CADCFC", breakLine: true } },
  { text: " ", options: { fontSize: 8, breakLine: true } },
  { text: "Team DataSphere · Section G · Advanced Data Analytics (Sem 5), PES University", options: { fontSize: 12, breakLine: true } },
  { text: "Bhuvi S. Katti (PES1UG24AM418) · Mythri S (PES1UG24AM419) · Poojitha L. B. R. (PES1UG24AM384) · Diran Somanna (PES1UG24AM364)", options: { fontSize: 11 } },
], { placeholder: "body" });
s.addText("github.com/stacktrace-bhuvi/datasphere", { x: 0.6, y: 5.0, w: 6, h: 0.3, fontSize: 10, color: "CADCFC", isTextBox: true, margin: 0 });
s.addNotes("Title. One line on what the project is: we ask whether the error of a SHAP or LIME explanation, measured against the causal truth, can be predicted from things you can observe without that truth.");

// --- problem
pres.addSection({ title: "Problem" });
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Problem" });
s.addText("Can we tell when SHAP and LIME mislead, without the true causes?", { placeholder: "title" });
s.addText(bullets([
  "People read SHAP/LIME causally: “this feature drives the outcome”. The methods only describe what the model uses.",
  "In real data the causal truth is unknown, so a misleading explanation cannot be detected.",
  "Our question: is the size of that error predictable from signals you can observe anyway — data dependence, model fit, explainer stability, explainer disagreement?",
], 12.5), { x: 0.5, y: 1.2, w: 5.2, h: 2.6, isTextBox: true, margin: 0, valign: "top" });
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.0, y: 1.2, w: 3.5, h: 3.7, fill: { color: THEME.colors.dk1 }, line: { color: THEME.colors.dk1 }, rectRadius: 0.1 });
s.addText("Chain worlds  X1 → X2 → X3 → X4 → Y", { x: 6.2, y: 1.3, w: 3.1, h: 0.35, fontSize: 11, bold: true, color: "CADCFC", isTextBox: true, margin: 0 });
s.addText(`share of importance on features that cause Y only through other features (mean over ${gapFam.chain_mediation.n} chain SCMs)`, { x: 6.2, y: 1.62, w: 3.1, h: 0.5, fontSize: 8.5, color: "CADCFC", isTextBox: true, margin: 0 });
if (chain) {
  statTile(s, 6.2, 2.15, 1.5, pct(chain.tau), "true causal effect τ", "FFFFFF", "CADCFC");
  statTile(s, 7.8, 2.15, 1.5, pct(chain.oracle), "exact SHAP of the true f*", THEME.colors.accent2, "CADCFC");
  statTile(s, 6.2, 3.45, 1.5, pct(chain.kernelshap), "KernelSHAP (fitted models)", THEME.colors.accent2, "CADCFC");
  statTile(s, 7.8, 3.45, 1.5, pct(chain.lime), "LIME (fitted models)", THEME.colors.accent2, "CADCFC");
}
s.addText("SHAP is faithful to the model, not to the world. That gap is what we measure — and try to predict.", { x: 0.5, y: 4.2, w: 5.2, h: 0.7, fontSize: 12, italic: true, color: C.text2, isTextBox: true, margin: 0 });
s.addNotes("Problem statement, same framing as Review 1. The box on the right is now computed over all 60 chain SCMs of the full run: the share of true effect on mediated features vs what SHAP and LIME give them.");

// --- approach
pres.addSection({ title: "Approach" });
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Approach" });
s.addText("Approach: a causal lab, an error decomposition, a diagnostic", { placeholder: "title" });
const stages = [
  ["1", "Causal worlds", `${B.n_scms} SCMs · 5 graph families · d = 5–10 · linear + non-linear mechanisms`, THEME.colors.accent1],
  ["2", "Ground truth", "Total effect τ_j under do(X_j) · exact Shapley values of the true f*", THEME.colors.accent1],
  ["3", "Models + explainers", "XGBoost, MLP · exact SHAP, TreeSHAP, KernelSHAP, LIME, permutation · 3 seeds", THEME.colors.accent1],
  ["4", "Error + observables", "Estimand gap / model misfit / explainer error · " + B.n_observables + " ground-truth-free observables", THEME.colors.accent2],
  ["5", "Diagnostic", "Meta-model vs 7 baselines · grouped CV · leave-one-graph-family-out", THEME.colors.accent2],
];
stages.forEach(([n, h, b, col], i) => {
  const x = 0.5 + i * 1.84;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.3, w: 1.7, h: 2.3, fill: { color: THEME.colors.lt2 }, line: { color: THEME.colors.lt2 }, rectRadius: 0.08 });
  s.addShape(pres.shapes.OVAL, { x: x + 0.12, y: 1.42, w: 0.36, h: 0.36, fill: { color: col }, line: { color: col } });
  s.addText(n, { x: x + 0.12, y: 1.42, w: 0.36, h: 0.36, fontSize: 12, bold: true, color: C.background1, align: "center", valign: "middle", isTextBox: true, margin: 0 });
  s.addText(h, { x: x + 0.12, y: 1.88, w: 1.5, h: 0.4, fontSize: 12, bold: true, isTextBox: true, margin: 0 });
  s.addText(b, { x: x + 0.12, y: 2.28, w: 1.5, h: 1.25, fontSize: 9.5, color: C.text2, isTextBox: true, margin: 0, valign: "top" });
  if (i < 4) s.addText("›", { x: x + 1.68, y: 2.2, w: 0.2, h: 0.4, fontSize: 18, color: C.text2, isTextBox: true, margin: 0, align: "center" });
});
s.addText(bullets([
  "Each stage writes files the next one reads (docs/SCHEMAS.md); every stage is resumable and tested (pytest, 79 tests).",
  "Stages 1–3 and 4a are the Review-1 pipeline; this review adds the observables, the diagnostic and the held-out evaluation, and fixes the issues raised at Review 1 (outcome SNR, scale-free metrics, stage 2→3 link).",
], 11), { x: 0.5, y: 3.8, w: 9, h: 1.3, isTextBox: true, margin: 0, valign: "top" });
s.addNotes("Pipeline overview. Stress that nothing in stage 4b or 5 sees tau or the graph.");

// --- benchmark
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Approach" });
s.addText("The benchmark: 300 causal worlds with exact ground truth", { placeholder: "title" });
statTile(s, 0.5, 1.25, 2.1, String(B.n_scms), "SCMs · 60 per family · 2000 rows each · 60/20/20 split", THEME.colors.accent1);
statTile(s, 2.7, 1.25, 2.1, String(B.n_attribution_rows), "explanations scored against the causal truth", THEME.colors.accent1);
statTile(s, 4.9, 1.25, 2.1, oracleR2 ? f2(oracleR2) : "—", "mean R² of the true function f* (was 0.33 at Review 1)", THEME.colors.accent2);
statTile(s, 7.1, 1.25, 2.4, `${f2(N.model_fit.xgboost.test_r2_mean)} / ${f2(N.model_fit.mlp.test_r2_mean)}`, "mean test R² of XGBoost / MLP", THEME.colors.accent1);
card(s, 0.5, 2.75, 2.9, 2.3, "Graph families", "Chain-mediation · Confounded fork · Feature collider · Erdős–Rényi DAG · Layered (held out)\n\nMechanisms: linear, tanh, sin, quadratic, ReLU, pairwise interactions. Y is a strict sink; outputs bounded, zero clipping.");
card(s, 3.55, 2.75, 2.9, 2.3, "Causal estimand τ_j", "τ_j = E_x | E[Y | do(X_j = x+σ_j)] − E[Y | do(X_j = x−σ_j)] | / 2\n\nNested Monte-Carlo with common random numbers (50 reference points × 5000 noise draws, 5 repeats, SE stored). Includes mediated effects.");
card(s, 6.6, 2.75, 2.9, 2.3, "Oracle Shapley values", "Exact interventional SHAP of f* on fixed rows: 100 explain (test) × 50 background (train), all 2^d coalitions vectorised.\n\nThe attribution a perfect model and perfect explainer would give.");
s.addNotes("Benchmark facts. The SNR fix: a single multiplier on Y's noise raised oracle R² from 0.33 to about 0.7 without touching the feature distributions.");

// --- decomposition definition
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Approach" });
s.addText("Three places an explanation can go wrong", { placeholder: "title" });
s.addText("τ  →  Φ(f*)  →  Φ(f)  →  Ê(f)", { x: 0.5, y: 1.2, w: 9, h: 0.55, fontSize: 22, bold: true, align: "center", fontFace: "Cambria", isTextBox: true, margin: 0 });
card(s, 0.5, 1.95, 2.9, 2.2, "Estimand gap", "τ vs exact SHAP of the true function.\nWhat remains with a perfect model and an exact explainer: SHAP answers a different question than “what happens if I intervene”.", { headColor: THEME.colors.accent1 });
card(s, 3.55, 1.95, 2.9, 2.2, "Model misfit", "Exact SHAP of f* vs exact SHAP of the fitted f.\nThe model learned a different response surface (noise, inductive bias, proxies).", { headColor: THEME.colors.accent2 });
card(s, 6.6, 1.95, 2.9, 2.2, "Explainer error", "Exact SHAP of f vs what KernelSHAP / LIME / permutation actually output.\nSampling, surrogate and shuffling error.", { headColor: THEME.colors.accent3 });
s.addText(bullets([
  "Feature-wise vectors telescope exactly; the norms do not add up, so components are reported separately.",
  "τ and SHAP live on different scales (Review-1 issue): every component is also computed on importance shares |v|/‖v‖₁ (scale-free) and in rank space (1 − Kendall τ-b, tie-aware). Permutation importance is global, so it is compared in share / rank space only.",
], 11), { x: 0.5, y: 4.3, w: 9, h: 0.9, isTextBox: true, margin: 0, valign: "top" });
s.addNotes("The decomposition. The 'exact' explainer is brute-force SHAP of the fitted model; the error table joins each explainer's output with it, which is the stage 2 to 3 link fixed since Review 1.");

// --- results A: decomposition
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Results" });
s.addText("Result 1 · Most of the error is structural, not the explainer's", { placeholder: "title" });
if (fs.existsSync(fig("fig_error_decomposition_share.png"))) s.addImage({ path: fig("fig_error_decomposition_share.png"), x: 0.4, y: 1.15, w: 6.0, h: 2.9, sizing: { type: "contain", w: 6.0, h: 2.9 } });
const kE = dec.kernelshap, lE = dec.lime, pE = dec.permutation, xE = dec.exact;
s.addText(bullets([
  `Estimand gap ${f2(xE.estimand_gap_share)} vs model misfit ${f2(xE.model_misfit_share)} (share L2, mean over all SCMs and seeds).`,
  `Explainer approximation adds little for KernelSHAP (${f2(kE.explainer_error_share)}), more for LIME (${f2(lE.explainer_error_share)}) and most for permutation importance (${f2(pE.explainer_error_share)}).`,
  `Rank error vs τ is ${f2(xE.total_error_rank)} even for exact SHAP of the fitted model; KernelSHAP ${f2(kE.total_error_rank)}, LIME ${f2(lE.total_error_rank)}, permutation ${f2(pE.total_error_rank)}.`,
  `Top-k recovery of the causal drivers: exact SHAP ${pct(xE.top_k_overlap)}, LIME ${pct(lE.top_k_overlap)}, permutation ${pct(pE.top_k_overlap)}.`,
], 10.5), { x: 6.5, y: 1.2, w: 3.1, h: 3.6, isTextBox: true, margin: 0, valign: "top" });
s.addText("Share L2 = distance between importance-share vectors; rank error = 1 − Kendall τ-b against τ. Means over 300 SCMs × 3 seeds.", { x: 0.5, y: 4.75, w: 9, h: 0.4, fontSize: 8.5, color: C.text2, isTextBox: true, margin: 0 });
s.addNotes("Headline result 1. Even a perfect explainer of a well-fit model is far from the causal ranking; the approximation error of the explainer is the smallest piece. The practical message: better SHAP estimators will not fix this.");

// --- results B: estimand gap by family
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Results" });
s.addText("Result 2 · The estimand gap depends on the causal structure", { placeholder: "title" });
if (fs.existsSync(fig("fig_estimand_gap_by_family.png"))) s.addImage({ path: fig("fig_estimand_gap_by_family.png"), x: 0.4, y: 1.15, w: 6.2, h: 2.6, sizing: { type: "contain", w: 6.2, h: 2.6 } });
const gapRows = Object.keys(famLabel).map((k) => [{ text: famLabel[k] }, { text: f2(gapFam[k].rank_mean) }, { text: f2(gapFam[k].share_mean) }]);
s.addTable([[{ text: "Family", options: { bold: true } }, { text: "Rank gap", options: { bold: true } }, { text: "Share gap", options: { bold: true } }], ...gapRows],
  { x: 6.8, y: 1.2, w: 2.8, colW: [1.3, 0.75, 0.75], fontSize: 9.5, border: { type: "solid", color: "E6E6E3", pt: 0.5 }, fill: { color: "FFFFFF" }, rowH: 0.28, margin: 0.04 });
s.addText(bullets([
  `Computed with the perfect model and exact SHAP: pure definition mismatch. ${pct(N.estimand_gap_overall.frac_scms_rank_gap_gt_0)} of SCMs have a non-zero rank gap.`,
  "Largest where effects are mediated (chain) or where parents of Y are themselves driven by other features (layered, fork); smallest in random DAGs with many direct parents.",
  "*Layered is the held-out family: the diagnostic never trains on it.",
], 10.5), { x: 0.5, y: 3.85, w: 9.1, h: 1.3, isTextBox: true, margin: 0, valign: "top" });
s.addNotes("Result 2: the gap is structural and family-dependent. This is what makes the prediction problem non-trivial under leave-one-family-out.");

// --- observables
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Results" });
s.addText(`What we can measure without the truth: ${B.n_observables} observables`, { placeholder: "title" });
card(s, 0.5, 1.2, 2.9, 3.3, "Data level (train split)", "• max / mean |corr| between features, share of pairs with |r| > 0.3\n• max / mean VIF\n• max / mean mutual information\n• feature–outcome |corr| and MI\n• n_train, d", { headColor: THEME.colors.accent1 });
card(s, 3.55, 1.2, 2.9, 3.3, "Model level", "• test / val MSE, test R², train R², train–test gap\n• prediction variance ratio\n• residual kurtosis\n• agreement between XGBoost and MLP predictions", { headColor: THEME.colors.accent2 });
card(s, 6.6, 1.2, 2.9, 3.3, "Explainer level", "• concentration: Gini, top-1 share, effective #features\n• sign consistency, local dispersion\n• stability across seeds\n• stability + sensitivity under input perturbation (σ = 0.05 sd)\n• disagreement with other explainers, and across models", { headColor: THEME.colors.accent3 });
s.addText("None of these uses τ, the oracle or the graph. The model and explainer names are added as indicators (they are known in practice).", { x: 0.5, y: 4.65, w: 9, h: 0.45, fontSize: 10.5, italic: true, color: C.text2, isTextBox: true, margin: 0 });
s.addNotes("The observable set. Three groups; emphasise the ones that turn out to matter: disagreement, fit, concentration.");

// --- protocol
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Results" });
s.addText("How the diagnostic is tested", { placeholder: "title" });
numberedRow(s, 0.5, 1.25, 4.4, 1, "Grouped CV", "5-fold, groups = SCM, strata = family. Unseen SCMs from seen families.", THEME.colors.accent1);
numberedRow(s, 0.5, 2.15, 4.4, 2, "Leave-one-graph-family-out (LOGFO)", "Train on 4 families, test on the 5th; repeated for all five. The headline protocol.", THEME.colors.accent2);
numberedRow(s, 0.5, 3.05, 4.4, 3, "Designated split", "Train on the four training families, test on the held-out layered family.", THEME.colors.accent3);
numberedRow(s, 0.5, 3.95, 4.4, 4, "Statistics", "Paired cluster bootstrap over SCMs (2000×) and Wilcoxon on per-SCM errors, vs the best baseline.", THEME.colors.accent6);
card(s, 5.3, 1.25, 4.2, 1.7, "Meta-models", "Gradient-boosted trees (200 × depth 3) · random forest · ridge. Inputs: the observables + model / explainer indicators. Targets: rank error (1 − Kendall τ-b vs τ) and share error.");
card(s, 5.3, 3.1, 4.2, 1.95, "Baselines (what a practitioner could do today)", "constant mean · correlation only · max VIF · test error · sample size · explainer disagreement only · explainer stability only — each a ridge on the named columns.");
s.addNotes("Protocol. Rows from the same SCM are never split between train and test. Metrics: Spearman between predicted and actual error, R², and AUROC for flagging rows above the training median.");

// --- results C: diagnostic performance
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Results" });
s.addText("Result 3 · The error is predictable, even on unseen graph families", { placeholder: "title" });
if (fs.existsSync(fig(`fig_logfo_spearman_${tgt}.png`))) s.addImage({ path: fig(`fig_logfo_spearman_${tgt}.png`), x: 0.4, y: 1.1, w: 5.9, h: 2.45, sizing: { type: "contain", w: 5.9, h: 2.45 } });
statTile(s, 6.6, 1.2, 1.45, f2(logfo.meta_spearman), "Spearman, meta-model, LOGFO pooled", THEME.colors.accent1);
statTile(s, 8.15, 1.2, 1.45, f2(logfo.best_baseline_spearman), `best baseline (${bestBase}), LOGFO`, THEME.colors.accent2);
statTile(s, 6.6, 2.45, 1.45, f2(gcv.meta_spearman), "Spearman, grouped CV (seen families)", THEME.colors.accent1);
statTile(s, 8.15, 2.45, 1.45, f2(logfo.meta_auroc_high_error), "AUROC, high-error flag, LOGFO", THEME.colors.accent1);
const famSp = logfo.meta_spearman_by_fold || {};
const famAu = (N.logfo_by_family_auroc[tgt] || {}).meta_gbm || {};
s.addText(bullets([
  `Target: share error ‖share(τ) − share(Ê)‖. Per held-out family, Spearman / AUROC: ${Object.keys(famLabel).map((k) => `${famLabel[k].replace("*", "")} ${f2(famSp[k])} / ${f2(famAu[k])}`).join(" · ")}.`,
  `Gain over the best baseline under LOGFO: ΔSpearman ${f2(vsb.spearman_gain)} [${f2(vsb.spearman_gain_ci_lo)}, ${f2(vsb.spearman_gain_ci_hi)}], Wilcoxon p ${vsb.wilcoxon_p < 0.001 ? "< 0.001" : f3(vsb.wilcoxon_p)}, better on ${pct(vsb.frac_scms_a_better)} of SCMs. On the layered family alone it ties the test-error baseline (${f2(desig.meta_spearman)} vs ${f2(desig.best_baseline_spearman)}).`,
  `Rank error (1 − Kendall τ-b on ≤ 10 features) is much harder: Spearman ${f2(rank.logfo.meta_spearman)} under LOGFO, ${f2(rank.grouped_cv.meta_spearman)} grouped CV — only disagreement and stability carry signal for it.`,
], 10), { x: 0.5, y: 3.8, w: 9.1, h: 1.4, isTextBox: true, margin: 0, valign: "top" });
s.addNotes("Headline result 3. The share error (how much importance mass is misallocated) is predictable and transfers to every family the model never saw; the rank error is not, beyond the disagreement/stability signal. Say both.");

// --- results D (new): which part of the error is predictable
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Results" });
s.addText("Result 4 · The predictable error is the model's and the explainer's", { placeholder: "title" });
if (fs.existsSync(fig("fig_component_predictability_logfo_spearman.png"))) s.addImage({ path: fig("fig_component_predictability_logfo_spearman.png"), x: 0.4, y: 1.1, w: 5.9, h: 2.45, sizing: { type: "contain", w: 5.9, h: 2.45 } });
const comp = (t) => f2(pooledSp(t, "logfo", "meta_gbm"));
const fiTop = (t, n) => (N.feature_importance[t] || []).slice(0, n).map((r) => r.feature).join(", ");
card(s, 6.5, 1.15, 3.1, 1.1, `Explainer error · ρ = ${comp("explainer_error_share")}`, `from ${fiTop("explainer_error_share", 2)}`, { headColor: THEME.colors.accent3 });
card(s, 6.5, 2.35, 3.1, 1.1, `Model misfit · ρ = ${comp("model_misfit_share")}`, `from ${fiTop("model_misfit_share", 3)}`, { headColor: THEME.colors.accent2 });
card(s, 6.5, 3.55, 3.1, 1.1, `Estimand gap · ρ = ${comp("estimand_gap_share")}`, `from ${fiTop("estimand_gap_share", 2)} — the structural part is mostly invisible`, { headColor: THEME.colors.accent1 });
s.addText(bullets([
  "Predicting each component separately (LOGFO, pooled Spearman): the two components a practitioner can act on — fit the model better, use a better explainer — are the ones the observables see. The dominant component, the estimand gap, is only weakly visible because nothing in the data says which correlated inputs are causes and which are mediators or proxies.",
  "Consequence: a diagnostic can tell you when an explanation is unreliable because of the model or the explainer; it cannot tell you when SHAP's question differs from the causal one — which is the most common failure.",
], 10), { x: 0.5, y: 3.7, w: 5.9, h: 1.45, isTextBox: true, margin: 0, valign: "top" });
s.addNotes("The most important slide. The error that is predictable is the error that is fixable; the structural estimand gap dominates and is largely invisible. This is the answer to the Review-2 question: where it works and where it fails.");

// --- results D: what carries the signal
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Results" });
s.addText("Result 5 · Which observables carry the signal", { placeholder: "title" });
if (fs.existsSync(fig(`fig_feature_importance_${tgt}.png`))) s.addImage({ path: fig(`fig_feature_importance_${tgt}.png`), x: 0.4, y: 1.1, w: 4.4, h: 3.5, sizing: { type: "contain", w: 4.4, h: 3.5 } });
const fiRows = fi.map((r) => [{ text: r.feature }, { text: f3(r.importance_mean) }]);
s.addTable([[{ text: "Observable", options: { bold: true } }, { text: "Δ R² when permuted", options: { bold: true } }], ...fiRows],
  { x: 5.1, y: 1.2, w: 4.4, colW: [2.9, 1.5], fontSize: 9.5, border: { type: "solid", color: "E6E6E3", pt: 0.5 }, rowH: 0.27, margin: 0.04 });
const bym = (N.logfo_by_method[tgt] || {}).meta_gbm || {};
const bymStr = Object.keys(bym).slice(0, 6).map((k) => `${k} ${f2(bym[k])}`).join(" · ");
s.addText(bullets([
  "Permutation importance of the meta-model trained on the four training families and scored on the held-out layered family.",
  `Stricter test — rank the SCMs for a fixed (model, explainer) under LOGFO, share error (Spearman): ${bymStr}.`,
  "Concentration (Gini, top-1 share, effective #features) and feature–outcome association are the strongest single signals; disagreement between explainers and between models comes next; max VIF alone carries almost none.",
], 10.5), { x: 5.1, y: 3.3, w: 4.4, h: 1.8, isTextBox: true, margin: 0, valign: "top" });
s.addNotes("What the diagnostic uses. Disagreement between explainers and between models, test fit and attribution concentration dominate; VIF alone is a weak signal, which matches the baseline table.");

// --- limitations
pres.addSection({ title: "Closing" });
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Closing" });
s.addText("Limitations and what we would do next", { placeholder: "title" });
card(s, 0.5, 1.2, 4.4, 3.8, "Limitations", "• Synthetic SCMs: d ≤ 10, additive Gaussian noise, one continuous outcome, five hand-designed families.\n• τ is one causal estimand (unit-sd shift, total effect); other choices (direct effect, ATE of a binary treatment) would move the estimand gap.\n• The diagnostic predicts how large the error is, not what the correct attribution is.\n• Calibrated to this benchmark; a real-data deployment needs re-validation.", { headColor: THEME.colors.accent2 });
card(s, 5.1, 1.2, 4.4, 3.8, "Next steps", "• Semi-synthetic checks on IHDP, Adult Income and California Housing (planned, not run).\n• Larger d with sampled-coalition oracles; non-additive noise; classification outcomes.\n• Causal / asymmetric Shapley values with a partially known graph as additional explainers.\n• Turn the diagnostic into a reusable warning: “this explanation is likely far from the causal ranking”.", { headColor: THEME.colors.accent1 });
s.addNotes("Be candid about scope. The strongest caveat: the diagnostic flags risk, it does not repair the explanation.");

// --- conclusion
s = pres.addSlide({ masterName: "DARK", sectionTitle: "Closing" });
s.addText("Conclusion", { placeholder: "title" });
s.addText([
  { text: `1. Even perfect SHAP of a perfect model ranks features differently from the causal effect — the estimand gap (${f2(xE.estimand_gap_share)} share L2) dwarfs the explainers' own approximation error.`, options: { breakLine: true, paraSpaceAfter: 6 } },
  { text: `2. How much importance mass an explanation misallocates is predictable from ground-truth-free observables: Spearman ${f2(logfo.meta_spearman)} and AUROC ${f2(logfo.meta_auroc_high_error)} under leave-one-graph-family-out, beating every single-signal baseline (Wilcoxon p < 0.001).`, options: { breakLine: true, paraSpaceAfter: 6 } },
  { text: `3. But the predictable part is the model's and the explainer's (ρ ${comp("model_misfit_share")} and ${comp("explainer_error_share")}); the structural gap is mostly invisible (ρ ${comp("estimand_gap_share")}). A diagnostic can warn about a bad model or explainer, not about SHAP asking a different question than "what if I intervene".`, options: { breakLine: true, paraSpaceAfter: 6 } },
  { text: "Code · data · notebook · report: github.com/stacktrace-bhuvi/datasphere", options: { fontSize: 11, color: "CADCFC" } },
], { placeholder: "body" });
s.addText("Bhuvi — SCMs, ground truth, oracle · Mythri — models, explainers · Poojitha — metrics, decomposition, error tables · Diran — observables, diagnostic, evaluation, figures", { x: 0.6, y: 5.0, w: 8.8, h: 0.35, fontSize: 9, color: "CADCFC", isTextBox: true, margin: 0 });
s.addNotes("Close with the three sentences. Then questions.");

(async () => {
  await pres.writeFile({ fileName: outPath });
  await applyTheme(outPath, THEME);
  console.log("wrote", outPath);
})();
