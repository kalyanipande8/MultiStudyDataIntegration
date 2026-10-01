# MultiStudyDataIntegration: Consolidating 3 Clinical Studies for Regulatory Reporting

Harmonizes three independently-run oncology trial exports - each with
its **own data structure, variable naming, units, and categorical
encodings** (the way three real studies run by different CROs/sponsors
actually arrive) - into **one validated, pooled dataset**, then runs
the exploratory analyses and regulatory-style visualizations a
submission package would need.

> **Disclaimer:** This is a portfolio/demo project. All patient-level
> data for the 3 studies is **synthetically simulated** by
> `simulate_studies.py`. No real study, sponsor, or patient data is
> used.

## The 3 studies

| | ONC-101 | ONC-202 | ONC-303 |
|---|---|---|---|
| Tumor size unit | mm | cm | mm |
| Sex encoding | `M`/`F` | `1`/`2` | `Male`/`Female` |
| Arm encoding | `Treatment`/`Placebo` | `Drug`/`Control` | `Active`/`Comparator` |
| Geography field | `SITE_COUNTRY` | `country` | `REGION` |
| Patient ID field | `PatientID` | `subj_id` | `PATID` |
| N | 200 | 200 | 200 |

All three share the same underlying design - randomized, 2-arm,
12-week tumor-response endpoint plus a time-to-event safety endpoint -
which is what makes pooling them meaningful rather than just
concatenating unrelated tables.

## What it does

1. **`simulate_studies.py`** generates the 3 raw exports above (600
   patients total), each with a distinct treatment effect size and
   event hazard, plus **3 deliberately injected data-quality issues**
   (one per study: an implausible age, a non-physical negative tumor
   measurement, and an unmappable sex code) so the validation step has
   something real to catch. Injected record IDs are written to
   `data/raw/injected_issues.json` for the test suite to verify
   against.

2. **`integrate.py`** parses each study's native format into one
   unified schema (`study, patient_id, arm, age, sex,
   baseline_tumor_mm, week12_tumor_mm, pct_tumor_change, ae_occurred,
   time_to_event_days, event_occurred, country`), converting units
   (cm->mm) and remapping categorical codes along the way, then
   validates every consolidated record against 7 checks (age range,
   valid sex/arm mapping, positive tumor measurements, finite percent
   change, positive time-to-event) and computes a real,
   non-hardcoded **data-integrity percentage**.

3. **`analysis.py`** runs the exploratory and regulatory analyses on
   the validated, pooled data:
   - Pooled + per-study summary statistics for tumor response
   - Per-study treatment effect (Welch's t-test, 95% CI) and a
     fixed-effect (inverse-variance-weighted) pooled estimate ->
     **forest plot**
   - Pooled distribution comparison of tumor response, treatment vs.
     control
   - **Kaplan-Meier** curves (manual product-limit estimator, no
     external survival package) for the time-to-event safety endpoint

4. **`run_all.py`** runs the whole pipeline.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
pytest -q
```

## Results

_All numbers below are copied directly from an actual run of `python
run_all.py` - nothing is hardcoded. Fixed random seed makes them
reproducible._

- **600 patient records** consolidated across **3 studies** into one
  unified schema (`data/processed/consolidated.csv`).
- **Data integrity: 99.5% (597 / 600 valid records)** - exactly 1
  injected issue per study (3 total) correctly caught by validation,
  none missed, none over-flagged.

| Study | Records | Valid | Integrity |
|---|---|---|---|
| ONC-101 | 200 | 199 | 99.5% |
| ONC-202 | 200 | 199 | 99.5% |
| ONC-303 | 200 | 199 | 99.5% |
| **ALL (pooled)** | **600** | **597** | **99.5%** |

### Pooled tumor response (pct_tumor_change, baseline -> week 12)

| Arm | n | Mean | SD | Median |
|---|---|---|---|---|
| Control | 300 | +5.21% | 15.51 | +4.97% |
| Treatment | 297 | -10.71% | 14.58 | -9.71% |

### Treatment effect by study (95% CI, Welch's t-test)

| Study | n (T/C) | Effect (pct points) | 95% CI | p-value |
|---|---|---|---|---|
| ONC-101 | 99 / 100 | -22.18 | [-26.19, -18.18] | <0.001 |
| ONC-202 | 99 / 100 | -15.79 | [-20.76, -10.82] | <0.001 |
| ONC-303 | 99 / 100 | -9.78 | [-13.00, -6.56] | <0.001 |
| **Pooled (fixed-effect)** | 297 / 300 | **-14.87** | **[-17.11, -12.63]** | - |

The three studies show a consistent direction and statistically
significant treatment effect, with magnitude varying by study (as
simulated) - exactly the kind of heterogeneity a forest plot is meant
to surface, and the kind of pooled estimate that justifies cross-study
submission packages in the first place.

## Figures

- `output/figures/forest_plot.png` - treatment effect by study with
  95% CI and pooled fixed-effect estimate
- `output/figures/tumor_change_distribution.png` - pooled distribution
  of tumor response, treatment vs. control
- `output/figures/km_curve.png` - pooled Kaplan-Meier curves by arm

## Project structure

```
simulate_studies.py     # 3 studies, different schemas/units/encodings -> data/raw/*.csv
integrate.py             # harmonize + consolidate + validate -> data/processed/
analysis.py               # exploratory stats + regulatory figures -> output/
run_all.py                 # orchestrates the full pipeline
tests/test_pipeline.py     # pytest checks incl. injected-issue sanity check
```

## Tech stack

- Python 3
- pandas / numpy for data simulation, harmonization, and wrangling
- scipy.stats for Welch's t-test
- matplotlib for figures
- pytest
