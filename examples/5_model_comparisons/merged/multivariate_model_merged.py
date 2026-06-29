from sklearn.metrics import roc_auc_score
import scipy.stats as stats
import pandas as pd
import numpy as np
import statsmodels.api as sm
from amelio_medullo import DataCleaning, Calculus

def load_data(data_path, cols_to_keep):
    data = pd.read_excel(data_path)
    data["Neurol_cond"] = data["Neurol_cond"].replace(["BM", "AVC", "Autre"], [1, 2, 3])
    data["Sex"] = data["Sex"].replace(["M", "F"], [1, 2])
    data["lesion_num"] = data.apply(DataCleaning.lesion_level_to_num, axis=1)
    X = data[cols_to_keep].copy()
    y = Calculus.calculate_MCID_2(data, 45)
    X['MCID'] = y['MCID_classes']
    # Drops rows where the target or the specific feature is missing
    clean_X = X.dropna(axis=0)

    return clean_X.drop(columns=["MCID"], axis=1), clean_X[["MCID"]]

def simple_stats(data, y, feature):
    group0 = data[y["MCID"] == 0][feature]
    group1 = data[y["MCID"] == 1][feature]
    return stats.mannwhitneyu(group0, group1)


def main(data_path, top_features):

    X, y = load_data(data_path, top_features)
    X = sm.add_constant(X) # Don't forget the intercept!
    
    model = sm.OLS(y, X).fit()
    
    print(model.summary())

    return model

if __name__ == "__main__":
    data_path = "/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/CHUNantes collaboration/donnees/data_from_dpi/merged_data_final.xlsx"

    # SELECTED FEATURES
    cols_to_keep = [
        "step_length",
        "Durée_min",
        "BWS_%_MOY",
        "duration",
        "Neurol_cond",
        "sessions_per_week"
    ]
    # All features with high correlated removed (i.e., cadence, speed)
    # cols_to_keep_2 = [
    #     "nb_sessions",
    #     # "duration",
    #     "Durée_min",
    #     "Vitesse_kmh_MOY",
    #     "BWS_%_MOY",
    #     "step_length",
    #     "Guidage_%_MOY",
    #     "sessions_per_week",
    #     "Neurol_cond",
    #     "Sex",
    #     "Age",
    #     "Nb sessions",
    #     "functional_level",
    #     "Lesion_num",
    #     "BMI",
    # ]

    main(data_path, cols_to_keep)