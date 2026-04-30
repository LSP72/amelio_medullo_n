from amelio_medullo import MuscleScore
import pandas as pd
import os

import tkinter as tk
from tkinter import filedialog


def combine_muscle_scores(df, mapping_dict, cols_to_add, side=None):
    if side :
        mapping_dict = MuscleScore.transform_dict_to_side(mapping_dict, side)
    
    movement_scores = MuscleScore.convert_muscles_to_movements(df, mapping_dict)

    for col in cols_to_add:
        if col not in movement_scores.columns.to_list():
            movement_scores = movement_scores.merge(df[["IPP", col]], on="IPP", how="right")

    return movement_scores


def combine_both_sides(df, mapping_dict, cols_to_add, splitted_legs:bool=True):
    if splitted_legs == True :
        right_scores = combine_muscle_scores(df, mapping_dict, cols_to_add, "right")
        left_scores = combine_muscle_scores(df, mapping_dict, cols_to_add, "left")

        combined_scores_df = right_scores.combine_first(left_scores)
        print(combined_scores_df.head())

    elif splitted_legs == False :
        scores = combine_muscle_scores(df, mapping_dict, cols_to_add)

    return combined_scores_df


def main(dict_mvt_BM, cols_to_add, file_path=None):
    if file_path is None:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select the data file", filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )

    data = pd.read_excel(file_path)

    splitted_legs = False
    combined_scores_df = combine_both_sides(data, dict_mvt_BM, cols_to_add, splitted_legs)

    directory_path = os.path.dirname(file_path)
    output_file = directory_path + "/combined_movement_scores.xlsx"
    combined_scores_df.to_excel(output_file, index=False)
    print(f"Combined movement scores have been saved to {output_file}.")


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
    ] 


    file_path = "/Users/mathildetardif/Documents/Documents/PhD/Nantes/m_a_testings_pre_post_data.xlsx"

    main(dict_mvt_BM, cols_to_add, file_path)
