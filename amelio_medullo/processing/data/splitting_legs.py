import numpy as np
import pandas as pd


class LegSplit:
    def __init__(self):
        self.col_to_complete = [
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
        self.col_to_complete_in_df_arranged_with_muscular_groups = [
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
        "H_Flex_ass",
        "H_Ext_PP",
        "H_abd",
        "H_add",
        "H_rot_int",
        "K_Flex",
        "K_Ext",
        "A_Dorsiflex_GT",
        "A_Plantarflex",
        "A_Ever",
        "A_Inver",
        "H_ext_GF",
        "A_dorsiflex_GF",
    ]

    @staticmethod
    def _calculate_legs(row, feature):
        val_D = pd.to_numeric(row[feature + "_D_pre"], errors="coerce")
        val_G = pd.to_numeric(row[feature + "_G_pre"], errors="coerce")

        if row["Neurol_cond"] == "AVC":
            if row["Affected_side"] == "D":
                return val_D
            else:
                return val_G

        else:
            if pd.isna(val_D) and pd.isna(val_G):
                return np.nan
            return np.nanmean([val_D, val_G])
    
    def split_legs(self, data, arranged_with_muscular_grps):
        if arranged_with_muscular_grps:
            col_to_complete = self.col_to_complete_in_df_arranged_with_muscular_groups
        else:
            col_to_complete = self.col_to_complete
        for col in col_to_complete:
            data[col] = data.apply(lambda row: LegSplit._calculate_legs(row, col), axis=1)
            print(col)

        print(data.columns.to_list())
        return data