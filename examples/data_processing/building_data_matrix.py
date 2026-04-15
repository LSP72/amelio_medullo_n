import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog

from amelio_medullo import DemographicData, DataCleaning, MuscleScore, DemographicData, ProcessExcel, Calculus


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

    return data_final


def clean_data(df: pd.DataFrame):
    """Function to clean the data.

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
    df = DataCleaning.clean_string(df["6MWT_m_pre"])
    df = DataCleaning.clean_string(df["6MWT_m_post"])
    df = DataCleaning.clean_string(df["10MWT_pas_pre"])
    df = DataCleaning.clean_string(df["10MWT_pas_post"])
    df = DataCleaning.clean_string(df["10MWT_sec_pre"])
    df = DataCleaning.clean_string(df["10MWT_sec_post"])

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
        df.drop(columns=dates, inplace=True)

    print("Delays have been calculated and date columns have been dropped.")

    return df


def motricity_score(df, file_path_1=None, file_path_2=None):
    """Function to calculate the motricity score.

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
        DataFrame containing the calculated motricity scores.
    """
    if file_path_1 is None:
        file_1 = collect_and_read_file()

    if file_path_2 is None:
        file_2 = collect_and_read_file()

    # func that norm scores to match the same scale (i.e., 0-5)

    df.merge(file_1, on="IPP", how="right")
    df.merge(file_2, on="IPP", how="right")

    print("Motricity scores have been normalised and merged with the original dataframe.")

    def collect_and_read_file():
        file_path = ProcessExcel.collect_excel_file_path()
        return pd.read_excel(file_path)

    return df


def calculate_MCID(df: pd.DataFrame, test: dict):
    """Function to calculate the MCID for a given test.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to be cleaned.
    test : str
        Name of the test for which to calculate the MCID.

    Returns
    -------
    pd.dataframe
        df with the MCID column.
    """
    MCID = Calculus.calculate_MCID(df[test + "_m_pre"], df[test + "_m_post"], threshold=30)
    df["MCID"] = MCID

    print("MCID has been calculated and added to the dataframe.")

    return df


def clean_assessments(df: pd.DataFrame, cols: list):
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
    df.drop(columns=cols, inplace=True)
    df.drop(columns=[col for col in df.columns if col.endswith("_post")], inplace=True)
    print("Selected assessment columns and all 'post' columns have been dropped.")

    return df


def main(keys: list, dict_days: dict, assessments_to_clean: list, test: str):
    df = collect_initial_file(keys)
    df_cleaned = clean_data(df)
    df_with_delays = calculate_days(df_cleaned, dict_days)
    df_with_motricity = motricity_score(df_with_delays)
    df_with_MCID = calculate_MCID(df_with_motricity, test)
    df_final = clean_assessments(df_with_MCID, assessments_to_clean)

    root = tk.Tk()
    root.withdraw()  # Hide the main window
    output_folder_path = filedialog.askdirectory(title="Select a Folder to Save")
    output_file = os.path.join(output_folder_path, "final_data_matrix.xlsx")

    df_final.to_excel(os.path.join(output_file), index=False)
    print(f"Final data matrix has been saved to '{output_file}'.")


if __name__ == "__main__":
    test = "6MWT"
    keys = ["TOUT", "6MWT"]
    dict_days = {"delay_injury": ["date", "Date of injury"], "delay_loko": ["date", "Entrée en MPR"]}
    assessment_to_clean = ["A_Ever", "etc."]
