"""
Multivariable association between features and the binary responder class (MCID).

Two models, because "OLS on a binary outcome" is a specific choice you should
make deliberately:

  - fit_logit()  : logistic regression. The correct default for a 0/1 outcome.
                   Reports odds ratios + 95% CI, pseudo-R2, and AUC (in-sample).
                   AUC is directly comparable to your CatBoost numbers.

  - fit_lpm()    : linear probability model (OLS on 0/1) WITH robust HC3 SEs.
                   Only use if you specifically want coefficients as absolute
                   changes in probability. Robust SEs patch the built-in
                   heteroscedasticity; they do NOT fix predictions outside [0,1].

FIXES relative to the old script:
  * Neurol_cond is ONE-HOT encoded, not coded 1/2/3 (it's nominal: BM/AVC/Autre;
    integer codes impose a fake order and corrupt every coefficient).
  * `data = data.apply(lesion_level_to_num, ...)`  -- the assignment was missing
    before, so the cleaning was silently discarded.
  * `.copy()` on the feature frame -- avoids the SettingWithCopy view bug.
  * AUC is in-sample here (all rows used to fit). It measures fit, not
    generalization -- do not report it as predictive performance. Your
    resampled/nested estimate is the performance number; this is a descriptive fit.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from amelio_medullo import DataCleaning, Calculus

# Nominal categoricals -> must be one-hot, never integer-coded.
CATEGORICAL = ["Neurol_cond"]      # Sex is 2-level; encoded 0/1 below (order irrelevant)


def load_data(data_path, cols_to_keep):
    data = pd.read_excel(data_path)
    data = data.apply(DataCleaning.lesion_level_to_num, axis=1)   # <-- assignment restored
    y = pd.Series(Calculus.calculate_MCID_2(data, 45)["MCID_classes"], name="MCID")

    X = data[cols_to_keep].copy()
    # Sex: 2 levels -> single 0/1 column (arbitrary but monotonic, so fine)
    if "Sex" in X.columns:
        X["Sex"] = X["Sex"].map({"M": 0, "F": 1})
    # Nominal categoricals present in the list -> one-hot, drop first level
    cats = [c for c in CATEGORICAL if c in X.columns]
    if cats:
        X = pd.get_dummies(X, columns=cats, drop_first=True, dtype=float)

    # align + listwise-drop on features AND target together
    df = X.copy()
    df["MCID"] = y.values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df.drop(columns="MCID"), df["MCID"].astype(int), len(df)


def fit_logit(X, y):
    Xc = sm.add_constant(X.astype(float))
    res = sm.Logit(y, Xc).fit(disp=0)
    print(res.summary())
    # odds ratios + CI
    or_tab = pd.DataFrame({
        "odds_ratio": np.exp(res.params),
        "ci_low": np.exp(res.conf_int()[0]),
        "ci_high": np.exp(res.conf_int()[1]),
        "p": res.pvalues,
    })
    print("\nOdds ratios (exp(coef)):")
    print(or_tab.round(3).to_string())
    # in-sample AUC (fit quality, NOT generalization)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, res.predict(Xc))
    print(f"\nPseudo-R2 (McFadden): {res.prsquared:.3f} | in-sample AUC: {auc:.3f}")
    return res


def fit_lpm(X, y):
    """OLS on 0/1 with HC3 robust SEs. Only if you want probability-scale coefs."""
    Xc = sm.add_constant(X.astype(float))
    res = sm.OLS(y, Xc).fit(cov_type="HC3")   # robust SEs for the built-in heterosked.
    print(res.summary())
    pred = res.predict(Xc)
    n_out = int(((pred < 0) | (pred > 1)).sum())
    print(f"\nPredictions outside [0,1]: {n_out}/{len(pred)} "
          f"(LPM artifact; can't be fixed by robust SEs)")
    return res


def main(data_path, features, model="logit"):
    X, y, n = load_data(data_path, features)
    print(f"n = {n} complete cases | {X.shape[1]} design columns "
          f"| responders = {int(y.sum())} ({y.mean():.1%})")
    ratio = n / X.shape[1]
    if ratio < 10:
        print(f"WARNING: {ratio:.1f} rows per coefficient (<10). Estimates unstable; "
              f"consider fewer features or penalized regression.")
    print()
    return fit_logit(X, y) if model == "logit" else fit_lpm(X, y)

if __name__ == "__main__":
    DATA_PATH = "/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/CHUNantes collaboration/donnees/data_from_dpi/final_data_matrix_sessions_separated.xlsx"
    # DATA_PATH = "/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/CHUNantes collaboration/donnees/data_from_dpi/merged_data_final.xlsx"

    FEATURES = [
        "Lesion_num", "Nb sessions", "Sex", "BMI",
        "6MWT_m_pre", "delay_loko", "functional_level", "speed",
    ]
    # FEATURES = [
    #         "duration", "Durée_min", "Vitesse_kmh_MOY", "BWS_%_MOY",
    #         "step_length", "Guidage_%_MOY", "sessions_per_week",
    #         "Neurol_cond", "Sex", "Nb sessions", "BMI"
    #     ]

    main(DATA_PATH, FEATURES, model="logit")
