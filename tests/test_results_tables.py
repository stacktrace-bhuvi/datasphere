import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.metrics.results_tables import METRICS, REQUIRED, load_errors, summarize_errors


def _errors(path, rows=None):
    rows = rows if rows is not None else [
        # scm, model, explainer, seed, l2, rank, gap, misfit, expl
        ("a", "xgboost", "lime", 0, 0.5, 0.4, 0.3, 0.1, 0.2),
        ("a", "xgboost", "lime", 1, 0.7, 0.6, 0.3, 0.3, 0.4),
        ("a", "xgboost", "exact", 0, 0.4, 0.4, 0.3, 0.1, 0.0),
        ("b", "xgboost", "lime", 0, 0.9, 0.2, 0.5, 0.2, 0.3),
        ("b", "xgboost", "exact", 0, 0.8, 0.2, 0.5, 0.2, 0.0),
        ("c", "xgboost", "lime", 0, 0.1, 0.0, 0.1, 0.1, 0.1),
        ("a", "xgboost", "permutation", 0, np.nan, 0.5, 0.3, 0.1, np.nan),
        ("b", "xgboost", "permutation", 0, np.nan, 0.7, 0.5, 0.2, np.nan),
    ]
    pd.DataFrame(rows, columns=REQUIRED).to_csv(path, index=False)
    return path


def test_seeds_are_averaged_within_scm_then_aggregated_across_scms(tmp_path):
    t = summarize_errors(_errors(tmp_path / "e.csv"), tmp_path / "t")
    lime = t["by_model_explainer"].set_index("explainer").loc["lime"]
    per_scm = [np.mean([0.5, 0.7]), 0.9, 0.1]               # a (2 seeds), b, c
    assert lime["n_scms"] == 3 and lime["n_rows"] == 4
    assert lime["total_error_l2_mean"] == pytest.approx(np.mean(per_scm))
    assert lime["total_error_l2_std"] == pytest.approx(np.std(per_scm, ddof=1))
    sem = np.std(per_scm, ddof=1) / np.sqrt(3)
    assert lime["total_error_l2_sem"] == pytest.approx(sem)
    assert lime["total_error_l2_ci95_high"] - lime["total_error_l2_mean"] == pytest.approx(stats.t.ppf(0.975, 2) * sem)


def test_nan_metrics_for_global_explainer_are_not_counted(tmp_path):
    t = summarize_errors(_errors(tmp_path / "e.csv"), tmp_path / "t")
    perm = t["by_model_explainer"].set_index("explainer").loc["permutation"]
    assert perm["n_scms"] == 2 and perm["total_error_l2_n"] == 0
    assert np.isnan(perm["total_error_l2_mean"]) and perm["total_error_rank_n"] == 2
    assert perm["total_error_rank_mean"] == pytest.approx(0.6)


def test_single_scm_group_has_no_spread_estimates(tmp_path):
    t = summarize_errors(_errors(tmp_path / "e.csv", [("a", "mlp", "lime", 0, .1, .2, .3, .4, .5)]), tmp_path / "t")
    r = t["by_explainer"].iloc[0]
    assert r["n_scms"] == 1 and r["total_error_l2_mean"] == pytest.approx(0.1)
    assert np.isnan(r["total_error_l2_std"]) and np.isnan(r["total_error_l2_ci95_low"])


def test_components_are_deduplicated_over_explainers(tmp_path):
    t = summarize_errors(_errors(tmp_path / "e.csv"), tmp_path / "t")
    c = t["components_by_model"].iloc[0]
    # (a,seed0): gap .3 misfit .1 ; (a,seed1): .3/.3 ; (b,seed0): .5/.2 ; (c,seed0): .1/.1  -> per SCM
    assert c["n_scms"] == 3
    assert c["estimand_gap_mean"] == pytest.approx(np.mean([0.3, 0.5, 0.1]))
    assert c["model_misfit_mean"] == pytest.approx(np.mean([np.mean([0.1, 0.3]), 0.2, 0.1]))


def test_by_seed_and_files_written(tmp_path):
    t = summarize_errors(_errors(tmp_path / "e.csv"), tmp_path / "t")
    assert set(t["by_seed"]["seed"]) == {0, 1}
    for f in ["by_model_explainer", "by_explainer", "by_seed", "components_by_model"]:
        back = pd.read_csv(tmp_path / "t" / f"error_summary_{f}.csv")
        assert len(back) == len(t[f])
    assert "by_family" not in t


def test_family_table_uses_manifest(tmp_path):
    scms = tmp_path / "scms"
    scms.mkdir()
    pd.DataFrame({"scm_id": ["a", "b", "c"], "family": ["chain", "chain", "layered"]}).to_csv(scms / "manifest.csv", index=False)
    t = summarize_errors(_errors(tmp_path / "e.csv"), tmp_path / "t", scms_dir=scms)
    fam = t["by_family"].set_index(["family", "explainer"])
    assert fam.loc[("chain", "lime"), "n_scms"] == 2 and fam.loc[("layered", "lime"), "n_scms"] == 1
    assert (tmp_path / "t" / "error_summary_by_family.csv").exists()


def test_family_table_rejects_unknown_scm(tmp_path):
    scms = tmp_path / "scms"
    scms.mkdir()
    pd.DataFrame({"scm_id": ["a"], "family": ["chain"]}).to_csv(scms / "manifest.csv", index=False)
    with pytest.raises(ValueError, match="absent"):
        summarize_errors(_errors(tmp_path / "e.csv"), tmp_path / "t", scms_dir=scms)
    with pytest.raises(ValueError, match="manifest not found"):
        summarize_errors(_errors(tmp_path / "e.csv"), tmp_path / "t", scms_dir=tmp_path / "none")


def test_invalid_inputs(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_errors(tmp_path / "missing.csv")
    (tmp_path / "empty.csv").write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_errors(tmp_path / "empty.csv")
    pd.DataFrame(columns=REQUIRED).to_csv(tmp_path / "norows.csv", index=False)
    with pytest.raises(ValueError, match="no rows"):
        load_errors(tmp_path / "norows.csv")
    pd.DataFrame({"scm_id": ["a"]}).to_csv(tmp_path / "cols.csv", index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_errors(tmp_path / "cols.csv")
    rows = [("a", "m", "e", 0, 1, 1, 1, 1, 1)] * 2
    with pytest.raises(ValueError, match="duplicate"):
        load_errors(_errors(tmp_path / "dup.csv", rows))
    with pytest.raises(ValueError, match="infinite"):
        load_errors(_errors(tmp_path / "inf.csv", [("a", "m", "e", 0, np.inf, 1, 1, 1, 1)]))
    bad = _errors(tmp_path / "bad.csv", [("a", "m", "e", 0, 1, 1, 1, 1, 1)])
    bad.write_text(bad.read_text().replace(",1,1,1,1,1", ",x,1,1,1,1"))
    with pytest.raises(ValueError):
        load_errors(bad)


def test_every_metric_has_all_statistics(tmp_path):
    t = summarize_errors(_errors(tmp_path / "e.csv"), tmp_path / "t")
    for m in METRICS:
        for suf in ["mean", "std", "sem", "ci95_low", "ci95_high", "n"]:
            assert f"{m}_{suf}" in t["by_explainer"].columns
