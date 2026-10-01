"""
integrate.py

Harmonizes the three raw study exports - each with different variable
names, units, and categorical encodings - into one unified,
patient-level schema, then validates every consolidated record and
reports a data-integrity percentage.

Unified schema:
  study, patient_id, arm ('treatment'/'control'), age, sex ('M'/'F'),
  baseline_tumor_mm, week12_tumor_mm, pct_tumor_change,
  ae_occurred (bool), time_to_event_days, event_occurred (bool), country

Validation rules (a record is valid iff ALL hold):
  - age is present and within [18, 90]
  - sex maps to 'M' or 'F'
  - arm maps to 'treatment' or 'control'
  - baseline_tumor_mm and week12_tumor_mm are present and > 0
  - pct_tumor_change is finite
  - time_to_event_days is present and > 0

Data-integrity % = valid records / total records * 100, computed on the
actual consolidated data (not assumed) - the same records deliberately
corrupted by simulate_studies.py should be exactly the ones that fail.

Writes data/processed/consolidated.csv and
data/processed/integrity_report.csv.
"""

import os

import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")

UNIFIED_COLS = [
    "study", "patient_id", "arm", "age", "sex",
    "baseline_tumor_mm", "week12_tumor_mm", "pct_tumor_change",
    "ae_occurred", "time_to_event_days", "event_occurred", "country",
]


def load_study_a():
    df = pd.read_csv(os.path.join(RAW_DIR, "study_a.csv"))
    out = pd.DataFrame({
        "study": "ONC-101",
        "patient_id": "A-" + df["PatientID"].astype(str),
        "arm": df["Arm"].map({"Treatment": "treatment", "Placebo": "control"}),
        "age": df["Age"],
        "sex": df["Sex"].map({"M": "M", "F": "F"}),
        "baseline_tumor_mm": df["BASELINE_TUMOR_MM"],
        "week12_tumor_mm": df["WEEK12_TUMOR_MM"],
        "ae_occurred": df["AE_FLAG"].astype(bool),
        "time_to_event_days": df["TIME_TO_EVENT_DAYS"],
        "event_occurred": df["EVENT_OCCURRED"].astype(bool),
        "country": df["SITE_COUNTRY"],
    })
    return out


def load_study_b():
    df = pd.read_csv(os.path.join(RAW_DIR, "study_b.csv"))
    out = pd.DataFrame({
        "study": "ONC-202",
        "patient_id": "B-" + df["subj_id"].astype(str),
        "arm": df["treatment_group"].map({"Drug": "treatment", "Control": "control"}),
        "age": df["age_yrs"],
        "sex": df["sex"].map({1: "M", 2: "F"}),
        # unit harmonization: cm -> mm
        "baseline_tumor_mm": df["tumor_baseline_cm"] * 10,
        "week12_tumor_mm": df["tumor_wk12_cm"] * 10,
        "ae_occurred": df["adverse_event"].map({"Y": True, "N": False}),
        "time_to_event_days": df["time_to_event_days"],
        "event_occurred": df["event_occurred"].astype(bool),
        "country": df["country"],
    })
    return out


def load_study_c():
    df = pd.read_csv(os.path.join(RAW_DIR, "study_c.csv"))
    out = pd.DataFrame({
        "study": "ONC-303",
        "patient_id": "C-" + df["PATID"].astype(str),
        "arm": df["COHORT"].map({"Active": "treatment", "Comparator": "control"}),
        "age": df["AGE"],
        "sex": df["GENDER"].map({"Male": "M", "Female": "F"}),  # unmapped ("Unknown") -> NaN
        "baseline_tumor_mm": df["TUMORSIZE_BASE_MM"],
        "week12_tumor_mm": df["TUMORSIZE_W12_MM"],
        "ae_occurred": df["SAE_IND"].astype(bool),
        "time_to_event_days": df["TIME_TO_EVENT_DAYS"],
        "event_occurred": df["EVENT_OCCURRED"].astype(bool),
        "country": df["REGION"],
    })
    return out


def validate(df):
    issues = pd.DataFrame(index=df.index)
    issues["bad_age"] = ~df["age"].between(18, 90)
    issues["bad_sex"] = ~df["sex"].isin(["M", "F"])
    issues["bad_arm"] = ~df["arm"].isin(["treatment", "control"])
    issues["bad_baseline"] = ~(df["baseline_tumor_mm"] > 0)
    issues["bad_week12"] = ~(df["week12_tumor_mm"] > 0)
    issues["bad_pct_change"] = ~np.isfinite(df["pct_tumor_change"])
    issues["bad_time_to_event"] = ~(df["time_to_event_days"] > 0)
    issues = issues.fillna(True)  # any NaN in a check counts as a failure
    is_valid = ~issues.any(axis=1)
    reason = issues.apply(lambda r: ",".join(c for c in issues.columns if r[c]), axis=1)
    return is_valid, reason


def consolidate():
    parts = [load_study_a(), load_study_b(), load_study_c()]
    unified = pd.concat(parts, ignore_index=True)
    unified["pct_tumor_change"] = (
        (unified["week12_tumor_mm"] - unified["baseline_tumor_mm"]) / unified["baseline_tumor_mm"] * 100
    )
    unified = unified[UNIFIED_COLS]

    is_valid, reason = validate(unified)
    unified["is_valid"] = is_valid
    unified["issue_reason"] = reason

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    unified.round(3).to_csv(os.path.join(PROCESSED_DIR, "consolidated.csv"), index=False)

    n_total = len(unified)
    n_valid = int(is_valid.sum())
    integrity_pct = round(100 * n_valid / n_total, 3)

    report = pd.DataFrame([{
        "study": s,
        "n_records": int((unified["study"] == s).sum()),
        "n_valid": int(((unified["study"] == s) & is_valid).sum()),
    } for s in unified["study"].unique()] + [{
        "study": "ALL (pooled)",
        "n_records": n_total,
        "n_valid": n_valid,
    }])
    report["integrity_pct"] = round(100 * report["n_valid"] / report["n_records"], 3)
    report.to_csv(os.path.join(PROCESSED_DIR, "integrity_report.csv"), index=False)

    return unified, integrity_pct, report


def main():
    unified, integrity_pct, report = consolidate()
    print(f"Consolidated {len(unified)} records across {unified['study'].nunique()} studies "
          f"into data/processed/consolidated.csv")
    print(f"Overall data integrity: {integrity_pct}% "
          f"({int(unified['is_valid'].sum())}/{len(unified)} valid records)")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
