import os
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
from amelio_medullo import ProcessDataLokomat, Calculus

"""
    This script generates a table with for each row:
    ID's patients, info on their Loko training
"""


def collect_files(reports_folder_path=None):
    """Collect the folder and extract the files' paths

    Parameters
    ----------
    reports_folder_path : str
        Path to the folder with all reports.

    Returns
    -------
    list
        List with all the paths of the files.
    """
    if reports_folder_path is None:
        root = tk.Tk()
        root.withdraw()
        reports_folder_path = filedialog.askdirectory(title="Select a folder")
    list_of_files = []
    for folder, dirs, files in os.walk(reports_folder_path):
        for file in files:
            if file.endswith(".xlsx") and not file.startswith("._"):
                list_of_files.append(os.path.join(folder, file))
    return list_of_files


def load_and_clean_data(names_file, data_path=None):
    # Select the file with the patients' data

    if data_path is None:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        data_path = filedialog.askopenfilename(title="Select Excel File", filetypes=[("Excel files", "*.xlsx")])
        root.destroy()

    # collect the table and the name of the sheet (i.e., pp's name)
    data, sheet_name = ProcessDataLokomat.load_and_preprocess_data(data_path)
    # dealing with the 2 headers => to one header
    data = ProcessDataLokomat.clean_data_columns(data)
    # select the only the sessions "Robotisé"
    data.drop(data[data["Type"] != "Robotisé"].index, inplace=True)

    # identify the ID with the sheet_name
    ID = ProcessDataLokomat.find_patient_id(sheet_name, names_file, name_col="names", id_col="IPP")

    return data, ID


def load_names_file(names_id_path=None):
    if names_id_path is None:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        names_id_path = filedialog.askopenfilename(title="Select Names ID File", filetypes=[("Excel files", "*.xlsx")])
        root.destroy()

    names_file = pd.read_excel(names_id_path)

    return names_file


def load_functional_table(file_path=None):
    if file_path is None:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        file_path = filedialog.askopenfilename(title="Select Names ID File", filetypes=[("Excel files", "*.xlsx")])
        root.destroy()

    file = pd.read_excel(file_path)
    return file[["IPP", "functional_level"]]


def count_nb_sessions(data, nb_of_sessions=1):
    start_date = data["Date"].iloc[0]
    end_date = data["Date"].iloc[nb_of_sessions - 1]
    delta = (end_date - start_date).days
    return delta


def count_and_calculate(data, final_data, ID):
    print("* " * 20)
    print(f"Process report {ID}")
    data = ProcessDataLokomat.merge_same_day(data)

    nb_of_sessions = int(data["Date"].nunique())
    final_data.at[ID, "nb_sessions"] = nb_of_sessions
    print(f"Number of sessions: {nb_of_sessions}")

    final_data.at[ID, "duration"] = count_nb_sessions(data, nb_of_sessions)
    print(f"Duration of the intervention (days): {final_data.at[ID, 'duration']} days")

    data["cadence"] = data["Distance_pas"] / data["Durée_min"]
    data["step_length"] = data["Distance_m"] / data["Distance_pas"]
    data["Guidage_%_MOY"] = data[["Guidage_G_%_MOY", "Guidage_D_%_MOY"]].mean(axis=1)

    for col in data.columns.to_list()[1:]:
        final_data.at[ID, col] = data[col].mean()
        print(f"Mean {col}: {final_data.at[ID, col]}")

    nb_of_weeks, nb_sessions_per_week = ProcessDataLokomat.calculate_sessions_metrics(data)
    final_data.at[ID, "sessions_per_week"] = nb_sessions_per_week
    print(f"Frequency of sessions: {nb_sessions_per_week} session/week")

    return final_data


def load_MCID_table(mcid_path=None):
    if mcid_path is None:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        mcid_path = filedialog.askopenfilename(title="Select MCID File", filetypes=[("Excel files", "*.xlsx")])
        root.destroy()

    mcid_data = pd.read_excel(mcid_path)
    end_data = mcid_data[["IPP", "6MWT_m_pre", "6MWT_m_post"]]
    MCID = Calculus.calculate_MCID_2(mcid_data)

    data = end_data.merge(MCID, on="IPP", how="left")

    return data


def associate_and_calculate_MCID(mcid_data, final_data, ID):
    if ID in mcid_data["IPP"].values:
        MCID_value = mcid_data["MCID_classes"][mcid_data["IPP"] == ID].values[0]
        final_data.at[ID, "MCID_6MWT"] = MCID_value
        print(f"MCID for 6MWT: {MCID_value}")
    else:
        final_data.at[ID, "MCID_6MWT"] = np.nan
        print("No MCID value found for this patient.")

    return final_data


def main(reports_folder_path=None, names_id_file=None, mcid_file=None):
    final_table = pd.DataFrame()
    list_of_files = collect_files(reports_folder_path)
    names_file = load_names_file(names_id_file)
    mcid_table = load_MCID_table(mcid_file)
    functional_table = load_functional_table(mcid_file)
    for file in list_of_files:
        data, ID = load_and_clean_data(names_file, file)  # data = training data
        final_table = count_and_calculate(data, final_table, ID)
        # final_table = associate_and_calculate_MCID(mcid_table, final_table, ID)
    final_table.reset_index(names="IPP", inplace=True)
    final_table = final_table.merge(mcid_table, on="IPP", how="left")
    final_table_with_func_level = final_table.merge(
        functional_table, on="IPP", how="right"
    )  # 'right' bcs want only the concerned participants (one session of Loko)

    print(final_table_with_func_level.head())
    final_table_with_func_level.to_excel(
        os.path.join(reports_folder_path, "loko_final_table_sessions_separated.xlsx"), index=True
    )


if __name__ == "__main__":
    reports_folder_path = 
    names_id_file = 
    mcid_file = 
    main(reports_folder_path, names_id_file, mcid_file)
