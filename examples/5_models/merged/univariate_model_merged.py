from sklearn.metrics import roc_auc_score
import scipy.stats as stats
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from amelio_medullo import DataCleaning, Calculus
import datetime


def load_data(data_path, cols_to_keep):
    data = pd.read_excel(data_path)
    data["Neurol_cond"] = data["Neurol_cond"].replace(["BM", "AVC", "Autre"], [1, 2, 3])
    data["Sex"] = data["Sex"].replace(["M", "F"], [1, 2])
    data.apply(DataCleaning.lesion_level_to_num, axis=1)
    X = data[cols_to_keep]
    y = Calculus.calculate_MCID_2(data, 45)
    X["MCID"] = y["MCID_classes"]
    # Drops rows where the target or the specific feature is missing
    clean_X = X.dropna(axis=0)
    print(f"Number of participants included: {len(X)}")

    return clean_X.drop(columns=["MCID"], axis=1), clean_X[["MCID"]]


def simple_stats(data, y, feature):
    group0 = data[y["MCID"] == 0][feature]
    group1 = data[y["MCID"] == 1][feature]
    return stats.mannwhitneyu(group0, group1)


def main(data_path, cols_to_keep):
    print("*" * 10)
    print(datetime.datetime.now())

    results = []
    X, y = load_data(data_path, cols_to_keep)
    X_log = np.log1p(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    for col in cols_to_keep:
        # 1. Prepare data
        X_sm = sm.add_constant(X_scaled[col])

        # 2. Fit Logistic Regression to get P-Value and Odds Ratio
        try:
            model = sm.Logit(y, X_sm).fit(disp=0)
            p_val = model.pvalues[col]
            odds_ratio = np.exp(model.params[col])
            probs = model.predict(X_sm)
            auc = roc_auc_score(y, probs)
        except:
            # Handles cases where the model fails to converge
            p_val, odds_ratio, auc = np.nan, np.nan, np.nan

        _, mw_p = simple_stats(data=X, y=y, feature=col)

        results.append(
            {"Biomarker": col, "AUC": auc, "Logit_P_Value": p_val, "MW_U_P_Value": mw_p, "Odds_Ratio": odds_ratio}
        )
        print("\n" + "* " * 10)
        print(f"Biomarker: {col}\nAUC: {auc}\nLogit_P_Value: {p_val}\nMW_U_P_Value: {mw_p}\nOdds_Ratio: {odds_ratio}")

    univariate_df = pd.DataFrame(results).sort_values("AUC", ascending=False)
    print(univariate_df.to_markdown())
    univariate_df.to_excel("results/uni_multi_variate/univariate_results_merged_dataset.xlsx")


if __name__ == "__main__":
    data_path = "/Volumes/SP UFD U2/PhD/Stage Nantes/data/datasets/final/merged_data_final.xlsx"

    # SELECTED FEATURES
    cols_to_keep = [
        "nb_sessions",
        "duration",
        "Durée_min",
        "Vitesse_kmh_MOY",
        "BWS_%_MOY",
        "cadence",
        "step_length",
        "Guidage_%_MOY",
        "sessions_per_week",
        "Neurol_cond",
        "Sex",
        "Age",
        "Nb sessions",
        "functional_level",
        "Lesion_num",
        "BMI",
    ]

    main(data_path, cols_to_keep)
