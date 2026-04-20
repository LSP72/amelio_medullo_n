import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from amelio_medullo import ProcessDataLokomat

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

def collect_ids(list_of_files, names_file_path):
    ids = {}
    names_file = pd.read_excel(names_file_path)
    names_file.dropna(axis=0, inplace=True)
    names_file['IPP'] = names_file['IPP'].astype(int)
    print(names_file.head())
    for file in list_of_files:
        data, sheet_name = ProcessDataLokomat.load_and_preprocess_data(file)
        data = ProcessDataLokomat.clean_data_columns(data)
        ID = ProcessDataLokomat.find_patient_id(sheet_name, names_file)
        ids[ID] = data
    return ids

def combine_reports(dict_of_files):
    dfs = []
    for ID, data in dict_of_files.items():
        data["ID"] = ID  
        dfs.append(data)
    combined = pd.concat(dfs, axis=0, ignore_index=True)
    return combined

def main(names_file_path,reports_folder_path=None):
    files = collect_files(reports_folder_path)
    dict_of_files = collect_ids(files, names_file_path)
    combined_df = combine_reports(dict_of_files)
    if reports_folder_path is None:
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askdirectory(title="Select a folder")
        output_path = os.path.join(path, "combined.xlsx")
    else:
        output_path = f"{reports_folder_path}/combined.xlsx"
    combined_df.to_excel(output_path, index=False)

    print(f"Combined report saved at: {output_path}")

if __name__ == "__main__":
    reports_folder_path = "/Volumes/SP UFD U2/PhD/Stage Nantes/LOKOMAT/Reports/"
    names_file_path = "/Users/mathildetardif/Documents/Documents/PhD/Nantes/autres/names_ipp.xlsx"
    main(names_file_path, reports_folder_path)


# # Now you can groupby that column
# grouped = combined.groupby("source_file").sum()  # or .mean(), .count(), etc.

# combined.to_excel("combined.xlsx", index=False)