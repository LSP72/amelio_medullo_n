import pandas as pd
import numpy as np
from amelio_medullo import LegSplit

"""
    This script splits the legs depending on the neurological condition of the participant:
        BM, others = mean of both legs
        AVC = affected leg
"""


def main(data_file_path, col_to_complete, output_path=None):
    sheet_name = "Sheet1"
    data = pd.read_excel(data_file_path, sheet_name=sheet_name)

    for col in col_to_complete:
        data[col] = data.apply(lambda row: LegSplit.calculate_legs(row, col), axis=1)

    print(data.columns.to_list())
    if output_path:
        data.to_excel(f"{output_path}/pre_post_data_final_with_legs_separated.xlsx")


if __name__ == "__main__":

    data_file_path = "/Users/mathildetardif/Documents/pre_post_data_final.xlsx"
    col_to_complete = [
        "Artic_hip_flex",
        "Artic_hip_ext",
        "Artic_hip_add",
        "Artic_hip_abd",
        "Artic_hip_add",
        "Artic_hip_rot_ext",
        "Artic_hip_rot_int",
        "Knee_flex",
        "Knee_ext",
        "Ank_flex_90",
        "Ank_flex_180",
        "Ank_ext",
        "Sartorius",
        "Iliopsoas",
        "Adductor",
        "RF",
        "QF",
        "Gracilis",
        "TA",
        "TP",
        "GM",
        "Gmin",
        "TFL",
        "Ext_Hall",
        "Ext_Dig",
        "Ext_Dig_Brev",
        "SmTD",
        "Smbr",
        "Fibu_long",
        "Gastroc",
        "Sol",
        "Fib_Brev",
        "Gmax",
        "FHL",
        "FDL",
        "Bic_Fem",
        "Intris",
        "H_flex_ass",
        "H_flex_GT",
        "H_ext_PP",
        "H_ext_GF",
        "H_abd",
        "H_add",
        "H_rot_int",
        "H_rot_ext",
        "K_ext",
        "K_flex",
        "A_flex_dors_GF",
        "A_flex_dors_GT",
        "A_flex_plant",
        "A_eversion",
    ]
    output_path = "/Users/mathildetardif/Documents"

    main(data_file_path, col_to_complete, output_path)
