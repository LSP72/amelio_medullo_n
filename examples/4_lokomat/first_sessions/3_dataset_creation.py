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

def merge_3rd_session(all_reports_data, data, variable_list, blocks):

    for id in data.index.tolist():
        nb_seances = 2  # Default value for the session to extract, i.e., the 3rd session 
        if id > 10**9:  # Check if the ID is greater than 10^9 => would mean that the patient had several blocks of Lokomat and that the ID has been adjusted in the data file
            block = int(str(id)[-1])  # Get the last digit of the ID, which corresponds to the block number
            true_id = int(str(id)[:-5])  # Remove the last 5 digits to get the original ID
            for i in range(int(block) - 1):
                nb_seances += blocks[true_id][i]  # Add the number of sessions in the previous blocks to get the correct index for the 3rd session of the current block
            print(f"> {true_id} had followed several blocks: processing block n°{block}")

        subdf = all_reports_data.groupby("ID").get_group(true_id if id > 10**9 else id).sort_values("Session(s)")

        for variable in variable_list:
            data.loc[id, f"{variable}_3rd"] = subdf.iloc[nb_seances][f"{variable}_MOY"].astype(float)
    return data

def main(data_path, all_reports_data_path, nb_sessions, variable_list=["Vitesse_kmh", "BWS_%"], patient_blocks=None):
    data = pd.read_excel(data_path)
    all_reports_data = pd.read_excel(all_reports_data_path)
    all_reports_data["Guidage_%_MOY"] = all_reports_data[["Guidage_G_%_MOY", "Guidage_D_%_MOY"]].mean(axis=1)
    clean_data = put_regression_info_in_line(data)
    merged_data = merge_3rd_session(all_reports_data, clean_data, variable_list, patient_blocks)

    merged_data.to_excel(f"results/loko_results/fits_over_first_{nb_sessions}_sessions_with_3rd_session.xlsx", index=True)
    print(merged_data.to_markdown())

if __name__ == "__main__":
    nb_sessions = 8
    data_path = f"results/loko_results/complete_data_with_mcids_{nb_sessions}_sessions.xlsx"
    root = tk.Tk()
    root.withdraw()
    all_reports_data_path = filedialog.askopenfilename(
        title="Select the Excel file containing all the Lokomat reports data",
        filetypes=[("Excel Files", "*.xlsx"), ("All Files", "*.*")]
    )
    variable_list = ["Vitesse_kmh", "BWS_%", "Guidage_%"]

    patient_blocks = {
            5750370: [19, 20],
            20047255: [20, 19],
            24190250: [16, 18],
            25801189: [17, 20],
            27522095: [20, 8, 19, 20, 22, 8],
            28373638: [5, 13, 5, 4, 6],
            30312319: [3, 13, 24, 17],
            30528453: [21, 19],
            31022187: [20, 20],
            32548837: [21, 20],
        }
    
    main(data_path, all_reports_data_path, nb_sessions, variable_list, patient_blocks)