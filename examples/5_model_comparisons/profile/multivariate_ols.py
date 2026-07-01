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
    data.apply(DataCleaning.lesion_level_to_num, axis=1)
    X = data[cols_to_keep]
    y = Calculus.calculate_MCID_2(data, 45)
    X["MCID"] = y["MCID_classes"]
    # Drops rows where the target or the specific feature is missing
    clean_X = X.dropna(axis=0)

    return clean_X.drop(columns=["MCID"], axis=1), clean_X[["MCID"]]


def simple_stats(data, y, feature):
    group0 = data[y["MCID"] == 0][feature]
    group1 = data[y["MCID"] == 1][feature]
    return stats.mannwhitneyu(group0, group1)


def main(data_path, top_features):

    X, y = load_data(data_path, top_features)
    X = sm.add_constant(X)  # Don't forget the intercept!

    model = sm.OLS(y, X).fit()

    print(model.summary())

    return model


if __name__ == "__main__":
    data_path = "/Volumes/SP UFD U2/PhD/Stage Nantes/data/datasets/final/final_data_matrix_sessions_separated.xlsx"

    # SELECTED FEATURES
    cols_to_keep = [
        "10MWT_pas_pre",
        "delay_injury",
        "6MWT_m_pre",
        "Neurol_cond",
        "Lesion_num",
    ]

    main(data_path, cols_to_keep)
