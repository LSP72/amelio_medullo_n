import pandas as pd
import numpy as np


class Calculus:
    def __init__(self, value):
        pass

    @staticmethod
    def calculate_MCID(pre_data, post_data, threshold) -> list:
        delta = post_data - pre_data
        MCID = []
        for i in delta:
            if i >= threshold:
                MCID.append(1)
            else:
                MCID.append(0)
        return pd.Series(MCID, index=pre_data.index)

    @staticmethod
    def calculate_MCID_2(data, default_threshold=30.0) -> pd.DataFrame:
        # 1. Calculating the difference
        data["MCID"] = data["6MWT_m_post"] - data["6MWT_m_pre"]

        # 2. Defining thresholds
        threshold_map = {"BM": 45.8, "AVC": 44.0, "Autre": default_threshold}

        # 3. Creating a Series of dynamic thresholds matching each row
        dynamic_thresholds = data["Neurol_cond"].map(threshold_map).fillna(default_threshold)

        # 4. Comparing  MCID and  dynamic threshold, then converting to 1/0
        mcid_binary = (data["MCID"] >= dynamic_thresholds).astype(int)
        mcid_binary.name = "MCID_classes"

        # 5. Returning 'IPP' column joined with MCID column
        return pd.concat([data["IPP"], mcid_binary], axis=1)

    @staticmethod
    def categorise(data, default_threshold=30):
        df = data.copy()

        # Change in 6MWT distance
        df["delta_6MWT"] = df["6MWT_m_post"] - df["6MWT_m_pre"]

        # Per-condition MCID thresholds; unknown conditions fall back to default
        threshold_map = {"BM": 45.8, "AVC": 44.0, "Autre": default_threshold}
        thresholds = df["Neurol_cond"].map(threshold_map).fillna(default_threshold)

        conditions = [
            df["delta_6MWT"] >= thresholds,  # met/exceeded MCID
            (df["delta_6MWT"] > 0) & (df["delta_6MWT"] < thresholds),  # improved, below MCID
        ]
        df["category"] = np.select(conditions, ["Responder", "Improved"], default="Non-Responder")

        return df[["IPP", "category"]]
