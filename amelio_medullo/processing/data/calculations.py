import pandas as pd


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
        mcid_binary = (data["MCID"] > dynamic_thresholds).astype(int)
        mcid_binary.name = "MCID_classes"

        # 5. Returning 'IPP' column joined with MCID column
        return pd.concat([data["IPP"], mcid_binary], axis=1)

    @staticmethod
    def categorise(data, default_threshold=30):
        # 1. Calculating the difference
        data["MCID"] = data["6MWT_m_post"] - data["6MWT_m_pre"]

        # 2. Defining thresholds
        threshold_map = {"BM": 45.8, "AVC": 44.0, "Autre": default_threshold}

        # 3. Creating a Series of dynamic thresholds matching each row
        dynamic_thresholds = data["Neurol_cond"].map(threshold_map).fillna(default_threshold)

        # 4. Comparing  MCID and  dynamic threshold, then converting to 1/0
        mcid_binary = (data["MCID"] > dynamic_thresholds).astype(int)
        mcid_binary.name = "categories"

        # 5. Comparing MCID with 0, if above "Improved", if not "Remains NResp"
        not_mcid_filter = (data["MCID"] > 0) & (data["MCID"] < dynamic_thresholds)
        data["categories"][not_mcid_filter] = "Non-Responder Improved"
        data["categories"].replace({"1": "Responder", "0": "Non-Responder Not Improved"})

        # 5. Returning 'IPP' column joined with MCID column
        return pd.concat([data["IPP"], mcid_binary], axis=1)
