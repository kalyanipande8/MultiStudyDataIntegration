"""
analysis.py

Exploratory analyses and regulatory-style visualizations on the
consolidated, cross-study dataset. Only records that pass validation
(is_valid == True) are analyzed - exactly what a real regulatory
submission would do with a flagged data-quality issue.

Analyses:
  1. Pooled + per-study summary statistics for the tumor-response
     endpoint (pct_tumor_change), by arm.
  2. Per-study treatment effect (treatment - control mean pct change)
     with 95% CI, plus a fixed-effect (inverse-variance-weighted)
     pooled estimate -> forest plot.
  3. Pooled distribution comparison of pct_tumor_change, treatment vs
     control (regulatory-style overlaid histograms/violin).
  4. Kaplan-Meier survival curves (time to event) pooled across
     studies, treatment vs control, computed with a manual
     product-limit estimator (no external survival-analysis
     dependency).

Writes output/exploratory_summary.csv and
output/figures/{forest_plot,tumor_change_distribution,km_curve}.png.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")


def load_valid():
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "consolidated.csv"))
    return df[df["is_valid"]].copy()


def pooled_summary(df):
    summary = (
        df.groupby(["study", "arm"])["pct_tumor_change"]
        .agg(n="count", mean="mean", sd="std", median="median")
        .round(3)
        .reset_index()
    )
    pooled = (
        df.groupby("arm")["pct_tumor_change"]
        .agg(n="count", mean="mean", sd="std", median="median")
        .round(3)
        .reset_index()
    )
    pooled.insert(0, "study", "ALL (pooled)")
    return pd.concat([summary, pooled], ignore_index=True)


def study_effects(df):
    """Per-study mean difference (treatment - control) in pct_tumor_change,
    with SE/95% CI from Welch's t-test, plus a fixed-effect pooled estimate."""
    rows = []
    for study, g in df.groupby("study"):
        t = g.loc[g["arm"] == "treatment", "pct_tumor_change"]
        c = g.loc[g["arm"] == "control", "pct_tumor_change"]
        diff = t.mean() - c.mean()
        se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
        ci_lo, ci_hi = diff - 1.96 * se, diff + 1.96 * se
        tstat, pval = stats.ttest_ind(t, c, equal_var=False)
        rows.append({"study": study, "n_treatment": len(t), "n_control": len(c),
                      "effect": diff, "se": se, "ci_lo": ci_lo, "ci_hi": ci_hi, "p_value": pval})
    effects = pd.DataFrame(rows)

    # Fixed-effect (inverse-variance-weighted) meta-analytic pooled estimate
    w = 1 / effects["se"] ** 2
    pooled_effect = float((effects["effect"] * w).sum() / w.sum())
    pooled_se = float(np.sqrt(1 / w.sum()))
    pooled_row = pd.DataFrame([{
        "study": "Pooled (fixed-effect)", "n_treatment": int(effects["n_treatment"].sum()),
        "n_control": int(effects["n_control"].sum()), "effect": pooled_effect, "se": pooled_se,
        "ci_lo": pooled_effect - 1.96 * pooled_se, "ci_hi": pooled_effect + 1.96 * pooled_se,
        "p_value": np.nan,
    }])
    return pd.concat([effects, pooled_row], ignore_index=True)


def kaplan_meier(times, events):
    """Manual product-limit (Kaplan-Meier) estimator.
    Returns arrays of time points and survival probability."""
    order = np.argsort(times)
    times, events = np.asarray(times)[order], np.asarray(events)[order]
    unique_times = np.unique(times[events == 1])
    surv = 1.0
    t_out, s_out = [0.0], [1.0]
    for t in unique_times:
        n_at_risk = np.sum(times >= t)
        n_events = np.sum((times == t) & (events == 1))
        surv *= (1 - n_events / n_at_risk)
        t_out.append(t)
        s_out.append(surv)
    return np.array(t_out), np.array(s_out)


def plot_forest(effects, path):
    plot_df = effects.copy()
    fig, ax = plt.subplots(figsize=(7, 4))
    ypos = np.arange(len(plot_df))[::-1]
    colors = ["#4C72B0"] * (len(plot_df) - 1) + ["#C44E52"]
    for y, (_, row), color in zip(ypos, plot_df.iterrows(), colors):
        ax.plot([row["ci_lo"], row["ci_hi"]], [y, y], color=color, lw=2)
        ax.plot(row["effect"], y, "s" if color == "#C44E52" else "o", color=color, ms=8)
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(plot_df["study"])
    ax.set_xlabel("Treatment effect on tumor % change (treatment - control, 95% CI)")
    ax.set_title("Forest plot: treatment effect on tumor response by study")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_distribution(df, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for arm, color in [("control", "#C44E52"), ("treatment", "#4C72B0")]:
        vals = df.loc[df["arm"] == arm, "pct_tumor_change"]
        ax.hist(vals, bins=25, alpha=0.55, density=True, label=f"{arm} (n={len(vals)})", color=color)
    ax.set_xlabel("Tumor size % change, baseline to week 12")
    ax.set_ylabel("Density")
    ax.set_title("Pooled distribution of tumor response by arm (3 studies consolidated)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_km(df, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for arm, color in [("control", "#C44E52"), ("treatment", "#4C72B0")]:
        g = df[df["arm"] == arm]
        t, s = kaplan_meier(g["time_to_event_days"].values, g["event_occurred"].astype(int).values)
        ax.step(t, s, where="post", label=f"{arm} (n={len(g)})", color=color, lw=2)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Event-free probability")
    ax.set_title("Pooled Kaplan-Meier curve by arm (3 studies consolidated)")
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    df = load_valid()
    summary = pooled_summary(df)
    effects = study_effects(df)

    summary.to_csv(os.path.join(OUTPUT_DIR, "exploratory_summary.csv"), index=False)
    effects.round(3).to_csv(os.path.join(OUTPUT_DIR, "treatment_effects.csv"), index=False)

    plot_forest(effects, os.path.join(FIG_DIR, "forest_plot.png"))
    plot_distribution(df, os.path.join(FIG_DIR, "tumor_change_distribution.png"))
    plot_km(df, os.path.join(FIG_DIR, "km_curve.png"))

    print(f"Analyzed {len(df)} valid records across {df['study'].nunique()} studies")
    print("\nPooled summary (pct_tumor_change by arm):")
    print(summary[summary["study"] == "ALL (pooled)"].to_string(index=False))
    print("\nTreatment effects by study (95% CI):")
    print(effects.round(3).to_string(index=False))
    print("\nWrote output/exploratory_summary.csv, output/treatment_effects.csv")
    print("Wrote output/figures/forest_plot.png, tumor_change_distribution.png, km_curve.png")


if __name__ == "__main__":
    main()
