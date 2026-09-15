"""
Univariate association of each feature with the continuous 6MWT change score
(6MWT_post - 6MWT_pre, raw metres).

WHAT THIS IS
------------
A marginal sanity check: does each feature, ON ITS OWN, track the size of
improvement? This is NOT a predictive model and NOT independent contribution.
Because your locomotor features are collinear, several will look predictive
for the same underlying reason. Do not sum the R2s; do not read them as a
combined model. For "how well could a simple model do", fit ONE multivariate
regression instead (noted at the bottom).

TWO TRAPS, HANDLED
------------------
1. Mathematical coupling: the change score = post - pre. Any feature that IS
   the pre term (6MWT_m_pre) appears, negated, inside the outcome. A negative
   slope then arises by arithmetic + regression-to-the-mean, not biology.
   Such features are FLAGGED, not silently trusted.
2. Categoricals (text): a slope "equation" is meaningless for BM/AVC/Autre or
   M/F. These get a group-difference test (eta^2, ANOVA p) instead of a line.

Continuous features -> OLS line: change = intercept + slope * feature, with
R2, adjusted R2, slope p-value, n used (per-feature missing-drop).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
from amelio_medullo import Calculus  # adjust import to your module path

# ============================ CONFIG =========================================
# DATA_PATH = "/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/CHUNantes collaboration/donnees/data_from_dpi/final_data_matrix_sessions_separated.xlsx"
DATA_PATH = "/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/CHUNantes collaboration/donnees/data_from_dpi/merged_data_final.xlsx"

# Column names for the change score components.
COL_6MWT_PRE = "6MWT_m_pre"
COL_6MWT_POST = "6MWT_m_post"

# Features to test.
# FEATURES = [
#     "Neurol_cond", "Lesion_num", "Nb sessions", "Sex", "Age", "BMI",
#     "6MWT_m_pre", "10MWT_pas_pre", "10MWT_sec_pre", "delay_injury",
#     "delay_loko", "functional_level", "speed",
# ]
FEATURES = [
    "duration",
    "Durée_min",
    "Vitesse_kmh_MOY",
    "BWS_%_MOY",
    "step_length",
    "Guidage_%_MOY",
    "sessions_per_week",
    "Neurol_cond",
    "Sex",
    "Nb sessions",
    "BMI",
]

# Features arithmetically tied to the outcome (the 'pre' term of the change).
# Their association is partly mechanical -> reported but flagged.
COUPLED = {COL_6MWT_PRE}
CATEGORICAL = {"Neurol_cond", "Sex"}  # text columns -> group test, not a line
# =============================================================================


def load():
    data = pd.read_excel(DATA_PATH)
    df = data.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def fit_continuous(x, y):
    m = (~x.isna()) & (~y.isna())
    x1, y1 = x[m].astype(float), y[m].astype(float)
    if len(x1) < 3 or x1.nunique() < 2:
        return None
    X = sm.add_constant(x1)
    res = sm.OLS(y1, X).fit()
    slope = res.params.iloc[1]
    intercept = res.params.iloc[0]
    return {
        "n": int(len(x1)),
        "equation": f"post = {intercept:.3f} + {slope:.4f} * feature",
        "r2": res.rsquared,
        "adj_r2": res.rsquared_adj,
        "slope_p": res.pvalues.iloc[1],
    }


def fit_categorical(x, y):
    """Group-difference test: how much change variance the category explains."""
    m = (~x.isna()) & (~y.isna())
    x1, y1 = x[m].astype(str), y[m].astype(float)
    if x1.nunique() < 2 or len(y1) < 3:
        return None
    groups = [y1[x1 == lvl] for lvl in x1.unique()]
    grand = y1.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((y1 - grand) ** 2).sum()
    eta2 = ss_between / ss_total if ss_total > 0 else np.nan
    # one-way ANOVA p
    from scipy import stats

    fval, pval = stats.f_oneway(*groups)
    means = "; ".join(f"{lvl}: {y1[x1==lvl].mean():.1f}" for lvl in x1.unique())
    return {"n": int(len(y1)), "equation": f"group means -> {means}", "r2": eta2, "adj_r2": np.nan, "slope_p": pval}


def plot_grid(df, y, features):
    """One panel per feature. Continuous -> scatter + OLS line + R2.
    Categorical -> boxplot per group + eta2. Coupled features flagged red."""
    n = len(features)
    ncols = 3 if n > 4 else 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 3.6))
    axes = np.atleast_1d(axes).ravel()

    for ax, f in zip(axes, features):
        m = (~df[f].isna()) & (~y.isna())
        yv = y[m].astype(float)
        coupled = f in COUPLED

        if f in CATEGORICAL:
            xv = df[f][m].astype(str)
            levels = list(xv.unique())
            data = [yv[xv == lvl] for lvl in levels]
            ax.boxplot(data, labels=levels)
            grand = yv.mean()
            ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in data)
            ss_t = ((yv - grand) ** 2).sum()
            eta2 = ss_b / ss_t if ss_t > 0 else np.nan
            _, p = stats.f_oneway(*data)
            ax.set_title(f"{f}\n$\\eta^2$={eta2:.3f}, p={p:.3f}", fontsize=9)
            ax.set_ylabel("6MWT post (m)", fontsize=8)
        else:
            xv = df[f][m].astype(float)
            ax.scatter(xv, yv, s=18, alpha=0.6, edgecolor="none")
            if xv.nunique() >= 2:
                X = sm.add_constant(xv)
                res = sm.OLS(yv, X).fit()
                b0, b1 = res.params.iloc[0], res.params.iloc[1]
                xs = np.linspace(xv.min(), xv.max(), 50)
                ax.plot(xs, b0 + b1 * xs, color="crimson", lw=1.5)
                eq = f"y={b0:.1f}{'+' if b1>=0 else '-'}{abs(b1):.3f}x"
                title = f"{f}\n{eq}  $R^2$={res.rsquared:.3f}"
            else:
                title = f"{f}\n(constant)"
            ax.set_title(title, fontsize=9, color="crimson" if coupled else "black")
            ax.set_xlabel(f, fontsize=8)
            ax.set_ylabel("6MWT post (m)", fontsize=8)
            if coupled:
                ax.text(
                    0.5,
                    0.02,
                    "may be coupled to outcome",
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="crimson",
                )

        ax.tick_params(labelsize=7)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Univariate association with 6MWT post (marginal, not independent)", fontsize=11, y=1.0)
    fig.tight_layout()
    fig.legend()
    fig.show()
    fig.savefig("univariate_regression_grid_post.png", dpi=150, bbox_inches="tight")
    print("\nSaved figure -> univariate_regression_grid_post.png")


def main():
    df = load()
    y = df["6MWT_m_post"]
    print(f"Outcome: 6MWT post. n with outcome = {y.notna().sum()}")
    print(f"6MWT post: mean {y.mean():.1f} m, sd {y.std():.1f}, range [{y.min():.0f}, {y.max():.0f}]\n")

    rows = []
    for f in FEATURES:
        if f not in df.columns:
            print(f"  ! {f} not in data — skipped")
            continue
        if f in CATEGORICAL:
            r = fit_categorical(df[f], y)
            kind = "categorical (eta2, ANOVA p)"
        else:
            r = fit_continuous(df[f], y)
            kind = "linear"
        if r is None:
            print(f"  ! {f} — too few values, skipped")
            continue
        r["feature"] = f
        r["kind"] = kind
        r["flag"] = "may be coupled to outcome (distrust)" if f in COUPLED else ""
        rows.append(r)

    out = pd.DataFrame(rows).sort_values("r2", ascending=False)
    cols = ["feature", "kind", "n", "r2", "adj_r2", "slope_p", "equation", "flag"]
    pd.set_option("display.width", 200, "display.max_colwidth", 60)
    print(out[cols].to_string(index=False))
    # out[cols].to_csv("univariate_regression_results.csv", index=False)

    print("\nREAD THIS:")
    print("- R2 here is MARGINAL and per-feature. Do NOT sum across features.")
    print("- Collinear locomotor features (6MWT, 10MWT, speed) may share signal;")
    print("  high R2 on several = the same axis counted repeatedly.")
    print(f"- {COL_6MWT_PRE} is flagged: its slope is partly arithmetic")
    print("  (it's the 'pre' inside post-minus-pre) + regression to the mean.")
    print("- For a real 'how good is a simple model' number, fit ONE OLS on all")
    print("  chosen features together and read that adjusted R2 — not these.")

    plot_grid(df, y, [f for f in FEATURES if f in df.columns])


if __name__ == "__main__":
    main()
