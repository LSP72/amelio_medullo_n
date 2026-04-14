from amelio_medullo import MuscleScore
import pandas as pd
import os

import tkinter as tk
from tkinter import filedialog


def combine_muscle_scores(df, mapping_dict, side):

    side_dict = MuscleScore.transform_dict_to_side(mapping_dict, side)
    movement_scores = MuscleScore.convert_muscles_to_movements(df, side_dict)

    return movement_scores


def combine_both_sides(df, mapping_dict):
    right_scores = combine_muscle_scores(df, mapping_dict, "right")
    left_scores = combine_muscle_scores(df, mapping_dict, "left")

    combined_scores_df = right_scores.combine_first(left_scores)
    print(combined_scores_df.head())

    return combined_scores_df


def main(file_path, dict_mvt_BM):
    data = pd.read_excel(file_path)

    combined_scores_df = combine_both_sides(data, dict_mvt_BM)

    directory_path = os.path.dirname(file_path)
    output_file = directory_path + "/combined_movement_scores.xlsx"
    combined_scores_df.to_excel(output_file, index=False)
    print("Combined movement scores have been saved to 'combined_movement_scores.xlsx'.")


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_path = filedialog.askopenfilename(
        title="Select a file", filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
    )

    dict_mvt_BM = {
        "H_Flex_ass": ["Sartorius", "Iliopsoas"],
        "H_abd": ["GM", "Gmax"],
        "H_add": ["Adductor"],
        "H_rot_int": ["Gm"],
        "K_Flex": ["SmTD", "Smbr", "Bic_Fem"],
        "K_Ext": ["RF", "QF", "Gracilis"],
        "A_Dorsiflex_GT": ["TA"],
        "A_Plantarflex": ["Gastroc", "Sol"],
        "A_Ever": ["Fibu_long"],
        "A_Inver": ["TP"],
    }

    main(file_path, dict_mvt_BM)
