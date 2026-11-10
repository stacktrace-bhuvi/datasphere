#!/usr/bin/env bash
# Package the dataset and results for the Google Drive folder.
#   bash deliverables/package.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p deliverables
rm -f deliverables/datasphere_benchmark_v1.zip deliverables/datasphere_results_v1.zip
# dataset: manifest + 300 SCM folders (parquet + json), ~55 MB
zip -q -r deliverables/datasphere_benchmark_v1.zip data/scms -x '*.DS_Store'
# results: csv tables, figures, summary, attribution files (parquet + json sidecars)
zip -q -r deliverables/datasphere_results_v1.zip results/errors.csv results/observables.csv results/models.csv \
    results/diagnostic_summary.json results/tables results/figures results/attributions
ls -la deliverables/*.zip
