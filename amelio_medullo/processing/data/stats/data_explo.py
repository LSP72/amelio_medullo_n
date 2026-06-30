import pandas as pd
import numpy as np


class Explo:
    def __init__(self):
        pass

    @staticmethod
    def count_phases(data):
        acute_filter = data["delay_injury"] <= 7
        subacute_filter = (data["delay_injury"] > 7) & (data["delay_injury"] <= 42)
        chronic_filter = data["delay_injury"] > 42
        missing_filter = data["delay_injury"].isna()

        df_phases = {
            "acute": {"count": acute_filter.sum()},
            "subacute": {"count": subacute_filter.sum()},
            "chronic": {"count": chronic_filter.sum()},
            "missing": {"count": missing_filter.sum()},
        }
        df = pd.DataFrame.from_dict(df_phases, orient="index")
        print(df)

    @staticmethod
    def count_improve_but_blw_mcid(data):
        data["MCID"] = data["6MWT_m_post"] - data["6MWT_m_pre"]
        improve_but_blw_mcid_avc_filter = (data["Neurol_cond"] == "AVC") & (data["MCID"] < 44) & (data["MCID"] > 0)
        improve_but_blw_mcid_bm_filter = (data["Neurol_cond"] == "BM") & (data["MCID"] < 45.8) & (data["MCID"] > 0)
        improve_but_blw_mcid_autres_filter = (data["Neurol_cond"] == "Autre") & (data["MCID"] < 45) & (data["MCID"] > 0)
        df_improve_but_blw = {
            "Stroke": {"count": improve_but_blw_mcid_avc_filter.sum()},
            "SCI": {"count": improve_but_blw_mcid_bm_filter.sum()},
            "Other": {"count": improve_but_blw_mcid_autres_filter.sum()},
            "Total": {
                "count": improve_but_blw_mcid_avc_filter.sum()
                + improve_but_blw_mcid_bm_filter.sum()
                + improve_but_blw_mcid_autres_filter.sum()
            },
        }
        df = pd.DataFrame.from_dict(df_improve_but_blw, orient="index")
        print("Count of patients not reaching the MCID value but who have improved:")
        print(df)

    @staticmethod
    def count_no_improve(data):
        data["MCID"] = data["6MWT_m_post"] - data["6MWT_m_pre"]
        stay_at_0_avc_filter = (data["Neurol_cond"] == "AVC") & (data["MCID"] <= 0)
        stay_at_0_bm_filter = (data["Neurol_cond"] == "BM") & (data["MCID"] <= 0)
        stay_at_0_autres_filter = (data["Neurol_cond"] == "Autre") & (data["MCID"] <= 0)

        df_stay_at_0 = {
            "Stroke": {"count": stay_at_0_avc_filter.sum()},
            "SCI": {"count": stay_at_0_bm_filter.sum()},
            "Other": {"count": stay_at_0_autres_filter.sum()},
            "Total": {"count": stay_at_0_avc_filter.sum() + stay_at_0_bm_filter.sum() + stay_at_0_autres_filter.sum()},
        }
        df = pd.DataFrame.from_dict(df_stay_at_0, orient="index")
        print("Count of patients who have not improved (Δ6MWT <= 0):")
        print(df)

    @staticmethod
    def info_about_endurance(df_improvements, neurol_cond=""):

        mean_improvements = df_improvements.mean()
        min_improvements = df_improvements.min()
        max_improvements = df_improvements.max()
        std_improvements = df_improvements.std()
        median_improvements = df_improvements.median()
        quant_improvements = df_improvements.quantile([0.25, 0.75])

        df_results_improvements = {
            "mean": mean_improvements,
            "std": std_improvements,
            "min": min_improvements,
            "max": max_improvements,
            "median": median_improvements,
            "[Q1, Q3]": [quant_improvements.loc[0.25], quant_improvements.loc[0.75]],
        }
        df = pd.DataFrame.from_dict(df_results_improvements, orient="index")
        print(f"Improvements in 6MWT in {neurol_cond} ({len(df_improvements)}):")
        print(df)
