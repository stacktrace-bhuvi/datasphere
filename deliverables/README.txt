DataSphere -- course project deliverables (deadline 16/11/2026 EOD)
====================================================================

What goes in the Google Drive folder "DataSphere"
-------------------------------------------------

1. Final one-pager
     deliverables/one_pager.md / one_pager.docx
     -> paste the sections into the one-pager template that was shared earlier.
        Section order matches the usual template: title, team, problem, objectives,
        dataset, methodology, results, conclusion, references.

2. Code (.ipynb)
     notebooks/DataSphere_pipeline.ipynb   (executed, outputs included)
     Runs end to end from the repo root; falls back to results_small/ if the full
     results/ folder is not present. The rest of the code is in the repo
     (github.com/stacktrace-bhuvi/datasphere, branch feat/diagnostic-stage).

3. Dataset
     deliverables/datasphere_benchmark_v1.zip
       data/scms/manifest.csv and 300 folders scm_XXXX_<family>_d<d>/ with
       data.parquet, scm.json, meta.json, ground_truth.json  (see docs/SCHEMAS.md)
     deliverables/datasphere_results_v1.zip
       results/errors.csv, observables.csv, models.csv, tables/, figures/,
       diagnostic_summary.json and the per-explanation attribution files.
     Upload both zips (or unzip into the Drive folder) and put the Drive link in
     the one-pager / report if the template asks for it.

4. Project PPT
     deliverables/DataSphere_Final_Presentation.pptx  (13 slides, speaker notes)
     -> when the course template arrives, copy slide by slide into it; the
        content per slide is self-contained. Regenerate after any change with
        the commands below.

5. Video presentation (3-5 min)
     deliverables/video_script.md  (about 4 minutes at normal pace, split
     B / M / P / D with slide cues). Record over the PPT.

6. IEEE report
     report/main.pdf  (IEEEtran conference format, 2 columns)
     source: report/main.tex + report/tables.tex + report/figures/


Regenerating everything from the results
----------------------------------------

  python deliverables/collect_numbers.py --scms-dir data/scms --results-dir results --out deliverables/numbers.json
  python deliverables/make_text_deliverables.py          # one_pager.*, video_script.*
  PPTX_SKILL_DIR=<path to pptx skill> node deliverables/make_deck.js   # or edit the pptx directly
  python deliverables/fill_report.py && (cd report && pdflatex main && pdflatex main)
  python notebooks/build_notebook.py --execute

All numbers in the one-pager, deck, script and report come from
deliverables/numbers.json, so they are consistent with each other.


What changed in the code since Review 1 (for the "implementation" slide / report)
----------------------------------------------------------------------------------

  - stage 2 -> 3 link: the error table now reads Phi(f) from the `exact` sidecar
  - outcome SNR: SCM.build(outcome_noise_scale=0.4); R^2 of f* 0.33 -> ~0.7
  - scale-free metrics: every error component also on importance shares and ranks
  - src/observables: 39 ground-truth-free observables (stage 4b) + sharded runner
  - src/diagnostic: meta-models, baselines, grouped CV / LOGFO / designated split,
    paired cluster bootstrap + Wilcoxon, permutation importance, figures, report
  - scripts: run_attributions --shard, build_observables.py, run_diagnostic.py
  - tasks.py: pipeline-small, pipeline-full, merge-shards
  - tests: 79 passing (was 63)
  - README.md and docs/SCHEMAS.md updated


Still open (not required for submission)
----------------------------------------

  - semi-synthetic checks on IHDP / Adult / California Housing
  - give the TAs and Bhaskarjyoti Das sir view access to the Drive folder
  - submit the Drive link through the Google Form
