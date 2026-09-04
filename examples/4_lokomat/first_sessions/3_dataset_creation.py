import pandas as pd
import tkinter as tk
from tkinter import filedialog


def put_regression_info_in_line(data):
    """This function takes as input the df of all the info from the linear regressions, all in lines.
    It therefore groups the data by ID and creates a new df with one line per ID, containing the slopes
    and slope SE for the parametersVitesse and BWS.
    #TODO: add the other BWS in the future and a way to handle the adding/removing of parameters in the future.

    Parameters
    ----------
    data : dataframe
        The input dataframe containing the linear regression information.

    Returns
    -------
    dataframe
        A new dataframe with one line per ID, containing the slopes and slope SE for the specified parameters.
    """
    df = dict()
    for id, subdf in data.groupby("ID"):
        # print(f"Key: {key}")
        df[id] = {
            "Vitesse_slope": subdf.loc[data["Feature"] == "Vitesse_kmh_MOY", "Slope"].astype(float).values[0],
            "Vitesse_slope_SE": subdf.loc[data["Feature"] == "Vitesse_kmh_MOY", "Slope SE"].astype(float).values[0],
            "BWS_slope": subdf.loc[data["Feature"] == "BWS_%_MOY", "Slope"].astype(float).values[0],
            "BWS_slope_SE": subdf.loc[data["Feature"] == "BWS_%_MOY", "Slope SE"].astype(float).values[0],
        }
    df = pd.DataFrame.from_dict(df, orient="index")
    print(df)
    return df

def merge_3rd_session(all_reports_data, data, variable_list):
    
    for id, subdf in all_reports_data.groupby("ID"):
            if id in data.index:
                for variable in variable_list:
                    data.loc[id, f"{variable}_3rd"] = subdf.iloc[2][f"{variable}_MOY"].astype(float)
            else:
                print(f"ID {id} not found in 'data'")
    return data

def main(data_path, all_reports_data_path, variable_list=["Vitesse_kmh", "BWS_%"]):
    data = pd.read_excel(data_path)
    all_reports_data = pd.read_excel(all_reports_data_path)
    clean_data = put_regression_info_in_line(data)
    merged_data = merge_3rd_session(all_reports_data, clean_data, variable_list)

    merged_data.to_excel("results/loko_results/fits_over_first_5_sessions_with_3rd_session.xlsx", index=True)
    
    print(merged_data.to_markdown())

if __name__ == "__main__":
    data_path = "results/loko_results/fits_over_first_5_sessions.xlsx"
    root = tk.Tk()
    root.withdraw()
    all_reports_data_path = filedialog.askopenfilename(
        title="Select a File",
        filetypes=[("Excel Files", "*.xlsx"), ("All Files", "*.*")]
    )
    variable_list = ["Vitesse_kmh", "BWS_%"]
    main(data_path, all_reports_data_path, variable_list)