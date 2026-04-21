import os
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
from amelio_medullo import ProcessDataLokomat, Calculus

def collect_files(reports_folder_path):
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
    if data_path is None:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        data_path = filedialog.askopenfilename(title="Select Excel File", filetypes=[("Excel files", "*.xlsx")])
        root.destroy()

    data, sheet_name = ProcessDataLokomat.load_and_preprocess_data(data_path)
    data = ProcessDataLokomat.clean_data_columns(data)
    data.drop(data[data['Type'] != 'Robotisé'].index, inplace=True)

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
   
def count_nb_sessions(data, nb_of_sessions=1):
    start_date = data['Date'].iloc[0]
    end_date = data['Date'].iloc[nb_of_sessions-1]
    delta = (end_date-start_date).days
    return delta

def count_and_calculate(data, final_data, ID):

    print("* " * 20)
    print(f'Process report {ID}')
    data = ProcessDataLokomat.merge_same_day(data)

    nb_of_sessions = int(data['Date'].nunique())
    final_data.at[ID, 'nb_sessions'] = nb_of_sessions
    print(f"Number of sessions: {nb_of_sessions}")

    final_data.at[ID, 'duration'] = count_nb_sessions(data, nb_of_sessions)
    print(f"Duration of the intervention: {final_data.at[ID, 'duration']} days")

    for col in data.columns.to_list()[1:]:
        final_data.at[ID, col] = data[col].mean()
        print(f"Mean {col}: {final_data.at[ID, col]}")

    return final_data

def load_MCID_table(mcid_path=None):
    if mcid_path is None:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        mcid_path = filedialog.askopenfilename(title="Select MCID File", filetypes=[("Excel files", "*.xlsx")])
        root.destroy()

    mcid_data = pd.read_excel(mcid_path)
    MCID = Calculus.calculate_MCID(mcid_data['6MWT_m_pre'], mcid_data['6MWT_m_post'], threshold=45)

    return pd.concat([mcid_data['IPP'], MCID], axis=1)

def associate_and_calculate_MCID(mcid_data, final_data, ID):
    if ID in mcid_data['IPP'].values:
        MCID_value = mcid_data.loc[mcid_data['IPP'] == ID, 0].values[0]
        final_data.at[ID, 'MCID_6MWT'] = MCID_value
        print(f"MCID for 6MWT: {MCID_value}")
    else:
        final_data.at[ID, 'MCID_6MWT'] = np.nan
        print("No MCID value found for this patient.")

    return final_data

def main(reports_folder_path=None, names_id_file=None, mcid_file=None):
    final_table = pd.DataFrame()
    list_of_files = collect_files(reports_folder_path)
    names_file = load_names_file(names_id_file)
    MCID_table = load_MCID_table(mcid_file)
    for file in list_of_files:
        data, ID = load_and_clean_data(names_file, file)
        final_table = count_and_calculate(data, final_table, ID)
        final_table = associate_and_calculate_MCID(MCID_table, final_table, ID)

    print(final_table.head())
    final_table.to_excel(os.path.join(reports_folder_path, "final_table.xlsx"), index=False)

if __name__ == "__main__":
    reports_folder_path = "/Volumes/SP UFD U2/PhD/Stage Nantes/LOKOMAT/Reports" 
    names_id_file = "/Users/mathildetardif/Documents/Documents/PhD/Nantes/autres/names_ipp.xlsx"
    mcid_file = "/Users/mathildetardif/Documents/Documents/PhD/Nantes/data_drafts/prelim_table_2.xlsx"
    main(reports_folder_path, names_id_file, mcid_file)