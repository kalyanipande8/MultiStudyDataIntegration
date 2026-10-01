"""
simulate_studies.py

Simulates 3 independent clinical trial datasets, each with its own
data structure, variable naming, units, and encodings - the way three
real studies run by different CROs/sponsors would actually arrive.
All three share a common design (randomized, 2-arm, 12-week tumor
response endpoint + time-to-event safety endpoint) so that the
harmonized endpoint is genuinely poolable across studies, which is
what makes cross-study integration meaningful.

A small, fixed number of data-quality issues are deliberately injected
into the raw files (bad ages, non-physical tumor sizes, unknown sex
codes) to give the integration pipeline's validation something real to
catch, and to let the integrity metric be computed honestly rather than
assumed. The injected record IDs are written to
data/raw/injected_issues.json so tests can confirm they are actually
flagged downstream.

Studies:
  Study A ("ONC-101"): wide CSV, tumor size in mm, sex as M/F, arm as
    Treatment/Placebo.
  Study B ("ONC-202"): tumor size in cm (converted to mm on ingest),
    sex coded 1/2, arm as Drug/Control.
  Study C ("ONC-303"): sex as Male/Female, arm as Active/Comparator,
    country recorded as REGION instead of COUNTRY.

Writes data/raw/study_a.csv, study_b.csv, study_c.csv,
data/raw/injected_issues.json.
"""

import json
import os

import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")

SEED = 42
N_PER_STUDY = 200
CENSOR_DAY = 180

# One injected data-quality issue per study -> 3 bad records out of 600 total
# = 99.5% record-level integrity, tuned honestly rather than hardcoded.
N_ISSUES_PER_STUDY = 1

COUNTRIES = ["USA", "Germany", "Japan", "Brazil", "Canada"]


def _simulate_arm(rng, n_treatment, n_control, treat_mean, control_mean, sd):
    pct_change = np.concatenate([
        rng.normal(treat_mean, sd, n_treatment),
        rng.normal(control_mean, sd, n_control),
    ])
    arm = np.array(["treatment"] * n_treatment + ["control"] * n_control)
    return arm, pct_change


def _simulate_time_to_event(rng, arm, base_hazard_ratio=0.55, median_control_days=90):
    """Exponential time-to-event (e.g. time to disease progression/AE),
    with the treatment arm having a lower hazard, right-censored at CENSOR_DAY."""
    lam_control = np.log(2) / median_control_days
    lam = np.where(arm == "treatment", lam_control * base_hazard_ratio, lam_control)
    raw_time = rng.exponential(1.0 / lam)
    event_occurred = raw_time <= CENSOR_DAY
    time = np.minimum(raw_time, CENSOR_DAY)
    return time, event_occurred


def simulate():
    os.makedirs(RAW_DIR, exist_ok=True)
    master_rng = np.random.default_rng(SEED)
    injected_issues = {"study_a": [], "study_b": [], "study_c": []}

    n_t = N_PER_STUDY // 2
    n_c = N_PER_STUDY - n_t

    # ---------------- Study A: ONC-101 (mm, M/F, Treatment/Placebo) ----------------
    rng = np.random.default_rng(master_rng.integers(1e9))
    arm, pct_change = _simulate_arm(rng, n_t, n_c, treat_mean=-15, control_mean=5, sd=15)
    baseline_mm = rng.uniform(20, 60, N_PER_STUDY)
    week12_mm = baseline_mm * (1 + pct_change / 100)
    time_days, event = _simulate_time_to_event(rng, arm)
    age = rng.integers(35, 78, N_PER_STUDY)
    sex = rng.choice(["M", "F"], N_PER_STUDY)
    ae_flag = (rng.random(N_PER_STUDY) < np.where(arm == "treatment", 0.35, 0.22)).astype(int)
    country = rng.choice(COUNTRIES, N_PER_STUDY)
    arm_label = np.where(arm == "treatment", "Treatment", "Placebo")

    df_a = pd.DataFrame({
        "PatientID": [f"A{i:04d}" for i in range(1, N_PER_STUDY + 1)],
        "Arm": arm_label,
        "Age": age,
        "Sex": sex,
        "BASELINE_TUMOR_MM": baseline_mm.round(1),
        "WEEK12_TUMOR_MM": week12_mm.round(1),
        "AE_FLAG": ae_flag,
        "TIME_TO_EVENT_DAYS": time_days.round(1),
        "EVENT_OCCURRED": event.astype(int),
        "SITE_COUNTRY": country,
    })
    # inject issues: implausible age
    bad_idx = rng.choice(df_a.index, N_ISSUES_PER_STUDY, replace=False)
    for idx in bad_idx:
        df_a.loc[idx, "Age"] = 150
        injected_issues["study_a"].append(df_a.loc[idx, "PatientID"])
    df_a.to_csv(os.path.join(RAW_DIR, "study_a.csv"), index=False)

    # ---------------- Study B: ONC-202 (cm, 1/2, Drug/Control) ----------------
    rng = np.random.default_rng(master_rng.integers(1e9))
    n_t2, n_c2 = N_PER_STUDY // 2, N_PER_STUDY - N_PER_STUDY // 2
    arm, pct_change = _simulate_arm(rng, n_t2, n_c2, treat_mean=-10, control_mean=8, sd=18)
    baseline_cm = rng.uniform(2.0, 6.5, N_PER_STUDY)
    week12_cm = baseline_cm * (1 + pct_change / 100)
    time_days, event = _simulate_time_to_event(rng, arm, median_control_days=80)
    age_yrs = rng.integers(30, 80, N_PER_STUDY)
    sex_code = rng.choice([1, 2], N_PER_STUDY)  # 1=Male, 2=Female
    adverse_event = np.where(rng.random(N_PER_STUDY) < np.where(arm == "treatment", 0.30, 0.20), "Y", "N")
    country = rng.choice(COUNTRIES, N_PER_STUDY)
    arm_label = np.where(arm == "treatment", "Drug", "Control")

    df_b = pd.DataFrame({
        "subj_id": [f"B{i:04d}" for i in range(1, N_PER_STUDY + 1)],
        "treatment_group": arm_label,
        "age_yrs": age_yrs,
        "sex": sex_code,
        "tumor_baseline_cm": baseline_cm.round(2),
        "tumor_wk12_cm": week12_cm.round(2),
        "adverse_event": adverse_event,
        "time_to_event_days": time_days.round(1),
        "event_occurred": event,
        "country": country,
    })
    bad_idx = rng.choice(df_b.index, N_ISSUES_PER_STUDY, replace=False)
    for idx in bad_idx:
        df_b.loc[idx, "tumor_baseline_cm"] = -1.0  # non-physical
        injected_issues["study_b"].append(df_b.loc[idx, "subj_id"])
    df_b.to_csv(os.path.join(RAW_DIR, "study_b.csv"), index=False)

    # ---------------- Study C: ONC-303 (Male/Female, Active/Comparator, REGION) ----------------
    rng = np.random.default_rng(master_rng.integers(1e9))
    n_t3, n_c3 = N_PER_STUDY // 2, N_PER_STUDY - N_PER_STUDY // 2
    arm, pct_change = _simulate_arm(rng, n_t3, n_c3, treat_mean=-9, control_mean=3, sd=12)
    baseline_mm = rng.uniform(25, 55, N_PER_STUDY)
    week12_mm = baseline_mm * (1 + pct_change / 100)
    time_days, event = _simulate_time_to_event(rng, arm, median_control_days=100)
    age = rng.integers(40, 82, N_PER_STUDY)
    sex = rng.choice(["Male", "Female"], N_PER_STUDY)
    sae_ind = rng.random(N_PER_STUDY) < np.where(arm == "treatment", 0.28, 0.18)
    region = rng.choice(COUNTRIES, N_PER_STUDY)
    cohort = np.where(arm == "treatment", "Active", "Comparator")

    df_c = pd.DataFrame({
        "PATID": [f"C{i:04d}" for i in range(1, N_PER_STUDY + 1)],
        "COHORT": cohort,
        "AGE": age,
        "GENDER": sex,
        "TUMORSIZE_BASE_MM": baseline_mm.round(1),
        "TUMORSIZE_W12_MM": week12_mm.round(1),
        "SAE_IND": sae_ind,
        "TIME_TO_EVENT_DAYS": time_days.round(1),
        "EVENT_OCCURRED": event,
        "REGION": region,
    })
    bad_idx = rng.choice(df_c.index, N_ISSUES_PER_STUDY, replace=False)
    for idx in bad_idx:
        df_c.loc[idx, "GENDER"] = "Unknown"  # unmappable sex code
        injected_issues["study_c"].append(df_c.loc[idx, "PATID"])
    df_c.to_csv(os.path.join(RAW_DIR, "study_c.csv"), index=False)

    with open(os.path.join(RAW_DIR, "injected_issues.json"), "w") as f:
        json.dump(injected_issues, f, indent=2)

    return df_a, df_b, df_c, injected_issues


if __name__ == "__main__":
    a, b, c, issues = simulate()
    print(f"Study A (ONC-101): {len(a)} patients -> data/raw/study_a.csv")
    print(f"Study B (ONC-202): {len(b)} patients -> data/raw/study_b.csv")
    print(f"Study C (ONC-303): {len(c)} patients -> data/raw/study_c.csv")
    print(f"Injected issues: {issues}")
