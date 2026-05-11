import os
import pandas as pd
import numpy as np
from scipy.stats import shapiro
from tableone import TableOne
import matplotlib.pyplot as plt
from amelio_medullo import Calculus


def merge_data_and_mcid(data):
    MCID = Calculus.calculate_MCID_2(data, 30)
    data = data.merge(MCID, on="IPP", how="left")
    return data


def find_non_normal_col(data, cont_cols):
    """Finds the columns that are not normal, to ensure the right stat tests therefore.

    Parameters
    ----------
    data : DataFrame
        Table with all data.
    cont_cols : list
        List of columns that are continuous (non cat variables).

    Return
    ------
    stat: pd.Series
        List of stats from Shapiro-Wilk tests.
    p_value_shapiro: pd.Series
        List of p-values from Shapiro-Wilk tests.
    variables_non_normal: list
        List of variables that are not normal.
    """

    variables_non_normal = []
    shapiro_results = []

    # 1. Testing normality of each continuous variables
    for col in cont_cols:
        # Drop all NaN (to ensure SW test to work)
        data_clean = data[col].dropna()

        # Shapiro-Wilk test
        stat, p_value_shapiro = shapiro(data_clean)
        shapiro_results.append({"Variable": col, "W_Statistic": stat, "p_value": p_value_shapiro})

        # If p < 0.05, distribution NOT normal
        if p_value_shapiro < 0.05:
            variables_non_normal.append(col)

        df_shapiro = pd.DataFrame(shapiro_results)

    print(f"NON normal variables: {variables_non_normal}")

    return df_shapiro, variables_non_normal


def print_and_save_shapiro(df_shapiro, group_by, output_path=None):
    """Saves the Shapiro stats to Excel and plots the p-values."""

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # 1. Save the raw stats for the reviewers
    excel_path = os.path.join(output_path, f"shapiro_stats_for_merged_dataset.xlsx")
    df_shapiro.to_excel(excel_path, index=False)

    print(f"\n[Shapiro-Wilk Test Results for {group_by}]")
    print(df_shapiro.to_markdown(index=False))
    print("\n")


def main(data_path, cont_cols, cols_to_keep, cat_cols, group_by, output_path=None):
    np.random.seed(42)

    data = pd.read_excel(data_path)
    # data = merge_data_and_mcid(data)
    df_shapiro, variables_non_normal = find_non_normal_col(data, cont_cols)

    print_and_save_shapiro(df_shapiro, group_by, output_path)

    table = TableOne(
        data, columns=cols_to_keep, categorical=cat_cols, groupby=group_by, pval=True, nonnormal=variables_non_normal
    )

    # Affichage propre
    print(table.tabulate(tablefmt="github"))

    if output_path:
        table.to_excel(f"{output_path}/table_demographique_by_{group_by}.xlsx")


if __name__ == "__main__":

    data_path = "/Volumes/SP UFD U2/PhD/Stage Nantes/data/datasets/final/merged_data_final.xlsx"
    output_path = "/Volumes/SP UFD U2/PhD/Stage Nantes/data"
    # mapping_dict = {
    #     "MCID_classes": {
    #         "cols_to_keep": ["Age", "Height", "Sex", "Weight", "functional_level", "Neurol_cond"],
    #         "cat_cols": ["Sex", "functional_level", "Neurol_cond"],
    #         "cont_cols": ["Age", "Height", "Weight"],
    #     },
    #     "Neurol_cond": {
    #         "cols_to_keep": ["Age", "Height", "Sex", "Weight", "functional_level", "MCID_classes"],
    #         "cat_cols": ["Sex", "functional_level", "MCID_classes"],
    #         "cont_cols": ["Age", "Height", "Weight"],
    #     },
    #     "Sex": {
    #         "cols_to_keep": ["Age", "Height", "Weight", "functional_level", "Neurol_cond", "MCID_classes"],
    #         "cat_cols": ["functional_level", "Neurol_cond", "MCID_classes"],
    #         "cont_cols": ["Age", "Height", "Weight"],
    #     },
    # }
    # Mapping dict for merged data
    mapping_dict = {
        "MCID_classes": {
            "cols_to_keep": ["Sex", "functional_level", "Neurol_cond", "Lesion_num", "Age", "Height", "Weight", "BMI",  "nb_sessions",
                        "duration",
                        "Durée_min",
                        "Vitesse_kmh_MOY",
                        "BWS_%_MOY",
                        "step_length",
                        "Guidage_%_MOY",
                        "sessions_per_week",
                        "Nb sessions"],
            "cat_cols": ["Sex", "functional_level", "Neurol_cond", "Lesion_num","Nb sessions"],
            "cont_cols": ["Age", "Height", "Weight", "BMI",  "nb_sessions",
                        "duration",
                        "Durée_min",
                        "Vitesse_kmh_MOY",
                        "BWS_%_MOY",
                        "step_length",
                        "Guidage_%_MOY",
                        "sessions_per_week"],
            },
        "Neurol_cond": {
            "cols_to_keep": ["Sex", "functional_level", "MCID_classes", "Lesion_num", "Age", "Height", "Weight", "BMI",  "nb_sessions",
                        "duration",
                        "Durée_min",
                        "Vitesse_kmh_MOY",
                        "BWS_%_MOY",
                        "step_length",
                        "Guidage_%_MOY",
                        "sessions_per_week",
                        "Nb sessions"],
            "cat_cols": ["Sex", "functional_level", "MCID_classes", "Lesion_num", "Nb sessions"],
            "cont_cols": ["Age", "Height", "Weight", "BMI",  "nb_sessions",
                        "duration",
                        "Durée_min",
                        "Vitesse_kmh_MOY",
                        "BWS_%_MOY",
                        "step_length",
                        "Guidage_%_MOY",
                        "sessions_per_week"],
            },
        "Sex": {
            "cols_to_keep": ["MCID_classes", "functional_level", "Neurol_cond", "Lesion_num", "Age", "Height", "Weight", "BMI",  "nb_sessions",
                        "duration",
                        "Durée_min",
                        "Vitesse_kmh_MOY",
                        "BWS_%_MOY",
                        "step_length",
                        "Guidage_%_MOY",
                        "sessions_per_week",
                        "Nb sessions"],
            "cat_cols": ["MCID_classes", "functional_level", "Neurol_cond", "Lesion_num", "Nb sessions"],
            "cont_cols": ["Age", "Height", "Weight", "BMI",  "nb_sessions",
                        "duration",
                        "Durée_min",
                        "Vitesse_kmh_MOY",
                        "BWS_%_MOY",
                        "step_length",
                        "Guidage_%_MOY",
                        "sessions_per_week",
                        ],
            },
    }

    # 2. Loop through the dictionary to run the analysis for each group
    for group_by_var, config in mapping_dict.items():
        print(f"\n{'='*50}")
        print(f"Generating Table 1 grouped by: {group_by_var.upper()}")
        print(f"{'='*50}\n")

        # Extract the specific lists for this iteration
        current_cols_to_keep = config["cols_to_keep"]
        current_cat_cols = config["cat_cols"]
        current_cont_cols = config["cont_cols"]

        # Call your main function with the extracted variables
        main(
            data_path=data_path,
            cont_cols=current_cont_cols,
            cols_to_keep=current_cols_to_keep,
            cat_cols=current_cat_cols,
            group_by=group_by_var,
            output_path=output_path,
        )
