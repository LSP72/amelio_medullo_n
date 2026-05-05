import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from amelio_medullo import DemographicData, DataCleaning, MuscleScore, DemographicData, ProcessExcel, Calculus, FunctionalLevel, LegSplit

"""
    This scripts builds the matrix to be then able to use it in ML model testsings.
    It takes different files as inputs:
        - file_path: path to the raw excel file in which we will collect the different data (i.e., demo, funct tests, etc.)
        - 
"""

def collect_initial_file(keys: list, file_path: str = None):
    """Function to collect and select the participants.

    Parameters
    ----------
    keys : list
        List of keys (i.e., tests) to filter the participants.
    file_path : str, optional
        Path to the Excel file containing the data, by default None

    Returns
    -------
    pd.DataFrame
        DataFrame containing only the participants with complete data for the specified tests.
    """
    if file_path is None:
        root = tk.Tk()
        root.withdraw()  # Hide the main window

        file_path = filedialog.askopenfilename(
            title="Select a file", filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )

    data = pd.read_excel(file_path)

    data_final = DataCleaning.select_patients_with_complete_data(data, keys)
    data_final = DataCleaning.clean_lesion_type(data_final, "Neurol_cond")

    return data_final, file_path


def clean_data(df: pd.DataFrame):
    """Function to clean the data:
        - fix the values of the different tests columns depending on walked perimeter
        - to clean the data from the str values (i.e., to put integers instaed of "0?")

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to be cleaned.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    df = DataCleaning.fix_value_of_test(df, ["6MWT_m_pre", "10MWT_pas_pre", "10MWT_sec_pre"], "Perim_marche_m_pre", 0)
    df = DataCleaning.fix_value_of_test(
        df, ["6MWT_m_post", "10MWT_pas_post", "10MWT_sec_post"], "Perim_marche_m_post", 0
    )

    # TODO: see if 10MWT_sec and 10MWT_sec are not redundant
    df["6MWT_m_pre"] = DataCleaning.clean_string(df["6MWT_m_pre"])
    df["6MWT_m_post"] = DataCleaning.clean_string(df["6MWT_m_post"])
    df["10MWT_pas_pre"] = DataCleaning.clean_string(df["10MWT_pas_pre"])
    df["10MWT_pas_post"] = DataCleaning.clean_string(df["10MWT_pas_post"])
    df["10MWT_sec_pre"] = DataCleaning.clean_string(df["10MWT_sec_pre"])
    df["10MWT_sec_post"] = DataCleaning.clean_string(df["10MWT_sec_post"])

    df.drop(columns=["tests"], inplace=True)

    print("Data has been cleaned from strings.")

    return df


def calculate_days(df: pd.DataFrame, dict_days: dict):
    """Function to calculate the delay between:
    - the date of injury and the date of 1st session of RAGT
    - the date of entry in rehab and the date of 1st session of RAGT

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to be cleaned.
    dict_days : dict
        Dictionary containing the column names for the dates to be used in the calculation.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the calculated delays and the original date columns dropped.
    """
    for key, dates in dict_days.items():
        df = DemographicData.calculate_day_btwn_2_cols(df, dates[0], dates[1], key)
        df.drop(columns=dates[1], inplace=True)

    df.drop(columns=dict_days[list(dict_days.keys())[0]][0], inplace=True)
    print("Delays have been calculated and date columns have been dropped.")

    return df


def add_muscular_scores(df: pd.DataFrame, muscular_scores_to_remove: list = None, selected_leg = False, combined_muscular_path: str = None):
    """Function that adds the muscular scores to the matrices.
        This function supposes that the combined_muscular_scores was already
        created and has a path (i.e., )

    Parameters
    ----------
    df : pd.DataFrame
        Matrix in creation.
    list_of_scores : list
        List of muscular scores to remove.

    Returns
    -------
    df : pd.DataFrame
        Matrix with the muscular scores added.
    """

    # STEP 1: Create or look for the matrix
    if combined_muscular_path is not None:
        # combined_muscular_path = ProcessExcel.collect_excel_file_path()
        muscular_data = pd.read_excel(combined_muscular_path)
        # muscular_data = df.merge(muscular_data, on="IPP", how="left")
    
    else:
        muscular_data = MuscleScore().add_muscle_scores(df, selected_leg)

    # STEP 2: Clean the df from any muscular scores.
    if muscular_scores_to_remove is not None:
        df = df.drop(columns=muscular_scores_to_remove, errors="ignore")

    # STEP 3: merge the muscular matrix with the df
    df = df.merge(muscular_data, on="IPP", how="left")

    print("Muscular scores have been added to the dataframe.")

    return df


def functional_score(df, file_path=None):
    """Function to calculate the functional score.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to be cleaned.
    file_1 : str, optional
        Path to the Excel file containing the mapping dictionary for the population 1, by default None
    file_2 : str, optional
        Path to the Excel file containing the mapping dictionary for the population 2, by default None

    Returns
    -------
    pd.DataFrame
        DataFrame containing the calculated functional scores.
    """
    if file_path is None:
        file_path = ProcessExcel.collect_excel_file_path()
    file = pd.read_excel(file_path, sheet_name="Demographics")
    
    functional_levels = FunctionalLevel.functional_categories(file)
    functional_df = file[["IPP"]].assign(functional_level=functional_levels)
    df = df.merge(functional_df, on="IPP", how="left")

    print("Functional scores have been added to the original dataframe.")

    return df


def clean_assessments(df: pd.DataFrame, joints_assessment_to_remove: list, other_cols_to_remove: list):
    """Function to clean the angular assessment data
        and remove all the 'post' columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to be cleaned.
    cols : list
        List of columns to be cleaned.

    Returns
    -------
    pd.DataFrame
        DataFrame with the nonuse columns dropped.
    """
    df.drop(columns=joints_assessment_to_remove + other_cols_to_remove, inplace=True, errors="ignore")
    df.drop(columns=[col for col in df.columns if col.endswith("_post") and col != "6MWT_m_post"], inplace=True)
    print("Selected assessment columns and all 'post' columns have been dropped.")

    return df


def main(
    keys: list,
    dict_days: dict,
    muscular_scores_to_remove: list,
    joints_assessments_to_remove: list,
    other_cols_to_remove: list,
    selected_leg = False,
    file_path: str = None,
    output_dir: str = None,
):
    df, file_path = collect_initial_file(keys, file_path)
    df_cleaned = clean_data(df)
    df_with_delays = calculate_days(df_cleaned, dict_days)
    df_with_muscular_scores = add_muscular_scores(df=df_with_delays, selected_leg=selected_leg, muscular_scores_to_remove=muscular_scores_to_remove)
    df_with_func_scores = functional_score(df_with_muscular_scores, file_path)
    if selected_leg == False:
        df_with_selected_leg = LegSplit().split_legs(df_with_func_scores, arranged_with_muscular_grps=True)
    else:
        df_with_selected_leg = df_with_func_scores.copy()
    df_final = clean_assessments(df_with_selected_leg, joints_assessments_to_remove, other_cols_to_remove)

    if output_dir is None:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        output_dir = filedialog.askdirectory(title="Select a Folder to Save")
    output_file = os.path.join(output_dir, "final_data_matrix_sessions_separated.xlsx")

    df_final.to_excel(os.path.join(output_file), index=False)
    print(f"Final data matrix has been saved to '{output_file}'.")


if __name__ == "__main__":
    keys = ["TOUT", "6MWT"]
    dict_days = {"delay_injury": ["date", "Date of injury"], "delay_loko": ["date", "Entrée en MPR"]}
    joints_to_remove = [
        "Artic_hip_abd_D_pre",
        "Artic_hip_abd_G_pre",
        "Artic_hip_add_D_pre",
        "Artic_hip_add_G_pre",
        "Artic_hip_rot_ext_D_pre",
        "Artic_hip_rot_ext_G_pre",
        "Artic_hip_rot_int_D_pre",
        "Artic_hip_rot_int_G_pre",
        "Ank_ext_D_pre",
        "Ank_ext_G_pre",
    ]
    muscular_scores_to_remove = [
        "H_flex_ass_D_pre",
        "H_flex_ass_G_pre",
        "H_flex_GT_D_pre",
        "H_flex_GT_G_pre",
        "H_ext_PP_D_pre",
        "H_ext_PP_G_pre",
        "H_ext_GF_D_pre",
        "H_ext_GF_G_pre",
        "H_abd_D_pre",
        "H_abd_G_pre",
        "H_add_D_pre",
        "H_add_G_pre",
        "H_rot_int_D_pre",
        "H_rot_int_G_pre",
        "H_rot_ext_D_pre",
        "H_rot_ext_G_pre",
        "K_ext_D_pre",
        "K_ext_G_pre",
        "K_flex_D_pre",
        "K_flex_G_pre",
        "A_dorsiflex_GF_D_pre",
        "A_dorsiflex_GF_G_pre",
        "A_dorsiflex_GT_D_pre",
        "A_dorsiflex_GT_G_pre",
        "A_plantarflex_D_pre",
        "A_plantarflex_G_pre",
        "A_eversion_D_pre",
        "A_eversion_G_pre",
        "Sartorius_D_pre",
        "Sartorius_G_pre",
        "Iliopsoas_D_pre",
        "Iliopsoas_G_pre",
        "Adductor_D_pre",
        "Adductor_G_pre",
        "RF_D_pre",
        "RF_G_pre",
        "QF_D_pre",
        "QF_G_pre",
        "Gracilis_D_pre",
        "Gracilis_G_pre",
        "TA_D_pre",
        "TA_G_pre",
        "TP_D_pre",
        "TP_G_pre",
        "GM_D_pre",
        "GM_G_pre",
        "Gmin_D_pre",
        "Gmin_G_pre",
        "TFL_D_pre",
        "TFL_G_pre",
        "Ext_Hall_D_pre",
        "Ext_Hall_G_pre",
        "Ext_Dig_D_pre",
        "Ext_Dig_G_pre",
        "Ext_Dig_Brev_D_pre",
        "Ext_Dig_Brev_G_pre",
        "SmTD_D_pre",
        "SmTD_G_pre",
        "Smbr_D_pre",
        "Smbr_G_pre",
        "Fibu_long_D_pre",
        "Fibu_long_G_pre",
        "Gastroc_D_pre",
        "Gastroc_G_pre",
        "Sol_D_pre",
        "Sol_G_pre",
        "Fib_Brev_D_pre",
        "Fib_Brev_G_pre",
        "Gmax_D_pre",
        "Gmax_G_pre",
        "FHL_D_pre",
        "FHL_G_pre",
        "FDL_D_pre",
        "FDL_G_pre",
        "Bic_Fem_D_pre",
        "Bic_Fem_G_pre",
        "Intris_D_pre",
        "Intris_G_pre",
    ]

    assessment_to_remove = [
        "MIF_pre",
        "sub_SCIM_pre",
        "SCIM_pre",
        "TUG_sec_pre",
        "Perim_marche_m_pre",
        "Aide technique_pre",
        "BBS_pre",
        "MIF_Loco_pre",
        "Unnamed: 12",
    ]

    main(keys, dict_days, muscular_scores_to_remove, joints_to_remove, assessment_to_remove, test)
    main(keys=keys, dict_days=dict_days, muscular_scores_to_remove=muscular_scores_to_remove,
         joints_assessments_to_remove=joints_to_remove, other_cols_to_remove=assessment_to_remove,
         file_path=file_path)
