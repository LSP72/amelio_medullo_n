from amelio_medullo import MuscleScore
import pandas as pd
import os

import tkinter as tk
from tkinter import filedialog


def combine_muscle_scores(df, mapping_dict, side, cols_to_add):

    side_dict = MuscleScore.transform_dict_to_side(mapping_dict, side)
    movement_scores = MuscleScore.convert_muscles_to_movements(df, side_dict)

    for col in cols_to_add:
        if col not in movement_scores.columns.to_list():
            movement_scores = movement_scores.merge(df[["IPP", col]], on="IPP", how="right")

    return movement_scores


def combine_both_sides(df, mapping_dict, cols_to_add):
    right_scores = combine_muscle_scores(df, mapping_dict, "right", cols_to_add)
    left_scores = combine_muscle_scores(df, mapping_dict, "left", cols_to_add)

    combined_scores_df = right_scores.combine_first(left_scores)
    print(combined_scores_df.head())

    return combined_scores_df


def main(dict_mvt_BM, cols_to_add, file_path=None):
    if file_path is None:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        file_path = filedialog.askopenfilename(
            title="Select a file", filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )

    data = pd.read_excel(file_path)

    combined_scores_df = combine_both_sides(data, dict_mvt_BM, cols_to_add)

    directory_path = os.path.dirname(file_path)
    output_file = directory_path + "/combined_movement_scores.xlsx"
    combined_scores_df.to_excel(output_file, index=False)
    print("Combined movement scores have been saved to 'combined_movement_scores.xlsx'.")


if __name__ == "__main__":

    dict_mvt_BM = {
        "H_Flex_ass": ["Sartorius", "Iliopsoas"],
        "H_Ext_PP": ["Gmax"],
        "H_abd": ["GM"],
        "H_add": ["Adductor"],
        "H_rot_int": ["Gm"],
        "K_Flex": ["SmTD", "Smbr", "Bic_Fem"],
        "K_Ext": ["RF", "QF", "Gracilis"],
        "A_Dorsiflex_GT": ["TA"],
        "A_Plantarflex": ["Gastroc", "Sol"],
        "A_Ever": ["Fibu_long"],
        "A_Inver": ["TP"],
    }
    cols_to_add = [
        "H_Ext_GF_D_pre_pre",
        "H_Ext_GF_G_pre_pre",
        "A_Dorsiflex_GF_D_pre_pre",
        "A_Dorsiflex_GF_G_pre_pre",
    ]  # Example columns to add, adjust as needed
    file_path = "/Users/mathildetardif/Documents/Documents/PhD/Nantes/m_a_testings_pre_post_data.xlsx"

    main(dict_mvt_BM, cols_to_add, file_path)
