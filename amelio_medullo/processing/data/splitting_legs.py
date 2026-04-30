import numpy as np
import pandas as pd


class LegSplit:
    def __init__(self):
        pass

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
