import json
import os
import sys

import pandas as pd
import pytest

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import simulate_studies as sim  # noqa: E402
import integrate  # noqa: E402
import analysis  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def ensure_pipeline_ran():
    sim.simulate()
    integrate.consolidate()
    yield


def test_all_three_studies_present():
    unified = pd.read_csv(os.path.join(integrate.PROCESSED_DIR, "consolidated.csv"))
    studies = unified["study"].unique()
    assert len(studies) == 3
    assert set(studies) == {"ONC-101", "ONC-202", "ONC-303"}
    for s in studies:
        assert (unified["study"] == s).sum() == sim.N_PER_STUDY


def test_patient_ids_unique_across_studies():
    unified = pd.read_csv(os.path.join(integrate.PROCESSED_DIR, "consolidated.csv"))
    assert unified["patient_id"].is_unique


def test_consolidated_integrity_in_reasonable_band():
    """3 injected issues out of 600 records -> 99.5% integrity by design;
    allow a reasonable band around that in case simulation parameters change."""
    unified = pd.read_csv(os.path.join(integrate.PROCESSED_DIR, "consolidated.csv"))
    integrity_pct = 100 * unified["is_valid"].mean()
    assert 98.0 <= integrity_pct <= 100.0
    assert integrity_pct == pytest.approx(99.5, abs=0.5)


def test_injected_bad_records_are_flagged_invalid():
    """Sanity check: the deliberately-inconsistent records injected by
    simulate_studies.py (bad age, non-physical tumor size, unmapped sex
    code) must actually be caught by validation, not silently pooled in."""
    with open(os.path.join(sim.RAW_DIR, "injected_issues.json")) as f:
        injected = json.load(f)

    unified = pd.read_csv(os.path.join(integrate.PROCESSED_DIR, "consolidated.csv"))
    unified = unified.set_index("patient_id")

    prefix_map = {"study_a": "A-", "study_b": "B-", "study_c": "C-"}
    bad_ids = [prefix_map[study] + raw_id for study, ids in injected.items() for raw_id in ids]
    assert len(bad_ids) == 3

    for pid in bad_ids:
        assert pid in unified.index
        assert unified.loc[pid, "is_valid"] == False  # noqa: E712
        assert unified.loc[pid, "issue_reason"] != ""


def test_valid_records_pass_all_checks():
    unified = pd.read_csv(os.path.join(integrate.PROCESSED_DIR, "consolidated.csv"))
    valid = unified[unified["is_valid"]]
    assert valid["age"].between(18, 90).all()
    assert valid["sex"].isin(["M", "F"]).all()
    assert valid["arm"].isin(["treatment", "control"]).all()
    assert (valid["baseline_tumor_mm"] > 0).all()
    assert (valid["week12_tumor_mm"] > 0).all()


def test_unit_conversion_study_b_cm_to_mm():
    """Study B reports tumor size in cm; confirm the harmonized mm value
    is consistent with a 10x conversion from the raw cm value."""
    raw_b = pd.read_csv(os.path.join(sim.RAW_DIR, "study_b.csv"))
    unified = pd.read_csv(os.path.join(integrate.PROCESSED_DIR, "consolidated.csv"))
    merged = raw_b.merge(
        unified, left_on=raw_b["subj_id"].apply(lambda x: f"B-{x}"), right_on="patient_id"
    )
    assert (merged["baseline_tumor_mm"].round(1) == (merged["tumor_baseline_cm"] * 10).round(1)).all()


def test_exploratory_analysis_runs_on_consolidated_data():
    df = analysis.load_valid()
    summary = analysis.pooled_summary(df)
    effects = analysis.study_effects(df)
    assert len(summary) > 0
    # 3 per-study effects + 1 pooled fixed-effect row
    assert len(effects) == 4
    assert effects["effect"].notna().all()
    # Treatment should reduce tumor size relative to control in every study
    # (by simulation design) and in the pooled estimate.
    assert (effects["effect"] < 0).all()


def test_kaplan_meier_curve_is_monotonically_non_increasing():
    df = analysis.load_valid()
    t, s = analysis.kaplan_meier(df["time_to_event_days"].values, df["event_occurred"].astype(int).values)
    assert (pd.Series(s).diff().dropna() <= 1e-9).all()
    assert s[0] == 1.0
