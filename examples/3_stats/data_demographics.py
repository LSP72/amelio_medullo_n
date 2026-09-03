import os
import pandas as pd
import numpy as np
from scipy.stats import shapiro
from tableone import TableOne
import matplotlib.pyplot as plt
from amelio_medullo import Calculus


def merge_data_and_mcid(data):
    MCID = Calculus.calculate_MCID_2(data, 45)
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
    excel_path = os.path.join(output_path, f"shapiro_stats_for_profile_dataset.xlsx")
    df_shapiro.to_excel(excel_path, index=False)

    print(f"\n[Shapiro-Wilk Test Results for {group_by}]")
    print(df_shapiro.to_markdown(index=False))
    print("\n")


def main(data_path, cont_cols, cols_to_keep, cat_cols, group_by, output_path=None):
    np.random.seed(42)

    data = pd.read_excel(data_path)
    data = merge_data_and_mcid(data)
    df_shapiro, variables_non_normal = find_non_normal_col(data, cont_cols)

    print_and_save_shapiro(df_shapiro, group_by, output_path)

    mean_table = TableOne(data, columns=cols_to_keep, categorical=cat_cols, groupby=group_by, pval=False)
    table = TableOne(
        data, columns=cols_to_keep, categorical=cat_cols, groupby=group_by, pval=True, nonnormal=variables_non_normal
    )

    # Affichage propre
    print(table.tabulate(tablefmt="github"))

    if output_path:
        mean_table.to_excel(f"{output_path}/table_demographics_by_{group_by}_all_with_mean_and_std_profile_data.xlsx")
        table.to_excel(f"{output_path}/table_demographics_by_{group_by}_profile_data.xlsx")


if __name__ == "__main__":

    data_path = "/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/CHUNantes collaboration/donnees/data_from_dpi/final_data_matrix_sessions_separated.xlsx"
    output_path = "results/stats"
    # mapping dict for profile data
    mapping_dict = {
        "MCID_classes": {
            "cols_to_keep": [
                "Neurol_cond",
                "Lesion_num",
                "Nb sessions",
                "Sex",
                "Age",
                "BMI",
                "6MWT_m_pre",
                "6MWT_m_post",
                "10MWT_pas_pre",
                "10MWT_sec_pre",
                "delay_injury",
                "delay_loko",
                "functional_level",
                "speed",
            ],
            "cat_cols": ["Sex", "functional_level", "Neurol_cond", "Nb sessions", "Lesion_num"],
            "cont_cols": [
                "Age",
                "BMI",
                "6MWT_m_pre",
                "6MWT_m_post",
                "10MWT_pas_pre",
                "10MWT_sec_pre",
                "delay_injury",
                "delay_loko",
                "speed",
            ],
        },
        "Neurol_cond": {
            "cols_to_keep": [
                "Neurol_cond",
                "Lesion_num",
                "Nb sessions",
                "Sex",
                "Age",
                "BMI",
                "6MWT_m_pre",
                "6MWT_m_post",
                "10MWT_pas_pre",
                "10MWT_sec_pre",
                "delay_injury",
                "delay_loko",
                "functional_level",
                "speed",
            ],
            "cat_cols": ["Sex", "functional_level", "Nb sessions", "Lesion_num"],
            "cont_cols": [
                "Age",
                "BMI",
                "6MWT_m_pre",
                "6MWT_m_post",
                "10MWT_pas_pre",
                "10MWT_sec_pre",
                "delay_injury",
                "delay_loko",
                "speed",
            ],
        },
        "Sex": {
            "cols_to_keep": [
                "Neurol_cond",
                "Lesion_num",
                "Nb sessions",
                "Sex",
                "Age",
                "BMI",
                "6MWT_m_pre",
                "6MWT_m_post",
                "10MWT_pas_pre",
                "10MWT_sec_pre",
                "delay_injury",
                "delay_loko",
                "functional_level",
                "speed",
            ],
            "cat_cols": ["functional_level", "Neurol_cond", "Nb sessions", "Lesion_num"],
            "cont_cols": [
                "Age",
                "BMI",
                "6MWT_m_pre",
                "6MWT_m_post",
                "10MWT_pas_pre",
                "10MWT_sec_pre",
                "delay_injury",
                "delay_loko",
                "speed",
            ],
        },
    }

    # Mapping dict for merged data
    # mapping_dict = {
    #     "MCID_classes": {
    #         "cols_to_keep": ["Sex", "functional_level", "Neurol_cond", "Lesion_num", "Age", "Height", "Weight", "BMI",  "nb_sessions",
    #                     "duration",  "Durée_min", "Vitesse_kmh_MOY", "BWS_%_MOY", "step_length",
    #                     "Guidage_%_MOY",
    #                     "sessions_per_week",
    #                     "Nb sessions", "6MWT_m_pre",
    #                     "6MWT_m_post",],
    #         "cat_cols": ["Sex", "functional_level", "Neurol_cond", "Lesion_num","Nb sessions"],
    #         "cont_cols": ["Age", "Height", "Weight", "BMI",  "nb_sessions",
    #                     "duration",
    #                     "Durée_min",
    #                     "Vitesse_kmh_MOY",
    #                     "BWS_%_MOY",
    #                     "step_length",
    #                     "Guidage_%_MOY",
    #                     "sessions_per_week",
    #                     "6MWT_m_pre",
    #                     "6MWT_m_post",],
    #         },
    #     "Neurol_cond": {
    #         "cols_to_keep": ["Sex", "functional_level", "MCID_classes", "Lesion_num", "Age", "Height", "Weight", "BMI",  "nb_sessions",
    #                     "duration",
    #                     "Durée_min",
    #                     "Vitesse_kmh_MOY",
    #                     "BWS_%_MOY",
    #                     "step_length",
    #                     "Guidage_%_MOY",
    #                     "sessions_per_week",
    #                     "Nb sessions",
    #                     "6MWT_m_pre",
    #                     "6MWT_m_post",],
    #         "cat_cols": ["Sex", "functional_level", "MCID_classes", "Lesion_num", "Nb sessions"],
    #         "cont_cols": ["Age", "Height", "Weight", "BMI",  "nb_sessions",
    #                     "duration",
    #                     "Durée_min",
    #                     "Vitesse_kmh_MOY",
    #                     "BWS_%_MOY",
    #                     "step_length",
    #                     "Guidage_%_MOY",
    #                     "sessions_per_week",
    #                     "6MWT_m_pre",
    #                     "6MWT_m_post",],
    #         },
    #     "Sex": {
    #         "cols_to_keep": ["MCID_classes", "functional_level", "Neurol_cond", "Lesion_num", "Age", "Height", "Weight", "BMI",  "nb_sessions",
    #                     "duration",
    #                     "Durée_min",
    #                     "Vitesse_kmh_MOY",
    #                     "BWS_%_MOY",
    #                     "step_length",
    #                     "Guidage_%_MOY",
    #                     "sessions_per_week",
    #                     "Nb sessions", "6MWT_m_pre",
    #                     "6MWT_m_post",],
    #         "cat_cols": ["MCID_classes", "functional_level", "Neurol_cond", "Lesion_num", "Nb sessions"],
    #         "cont_cols": ["Age", "Height", "Weight", "BMI",  "nb_sessions",
    #                     "duration",
    #                     "Durée_min",
    #                     "Vitesse_kmh_MOY",
    #                     "BWS_%_MOY",
    #                     "step_length",
    #                     "Guidage_%_MOY",
    #                     "sessions_per_week",
    #                     "6MWT_m_pre",
    #                     "6MWT_m_post",
    #                     ],
    #         },
    # }

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
