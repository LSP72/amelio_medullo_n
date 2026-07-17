import pandas as pd
from amelio_medullo import Calculus, Explo


def load_data(data_path):
    data = pd.read_excel(data_path)
    data["MCID"] = data["6MWT_m_post"] - data["6MWT_m_pre"]
    data = data.merge(Calculus.calculate_MCID_2(data, 45), on="IPP", how="left")
    return data


def give_count(data):
    # Patients who have improved but have not reached the MCID
    Explo.count_improve_but_blw_mcid(data)
    # Patients who have not improved (Δ6MWT <= 0)
    Explo.count_no_improve(data)


def endurance_filter(data, category, cond=None):
    """Boolean masks for each categories, can be restricted to
    1 "nerological condition" (cond=None => toutes conditions confondues).

    - "responders"             : MCID_classes == 1
    - "nonresponders_improved" : MCID_classes == 0 & MCID > 0
    - "all_improved"           : MCID > 0
    """

    if category == "responders":
        mask = data["MCID_classes"] == 1
    elif category == "nonresponders_improved":
        mask = (data["MCID_classes"] == 0) & (data["MCID"] > 0)
    elif category == "all_improved":
        mask = data["MCID"] > 0
    else:
        raise ValueError(f"Unknown category : {category}")

    if cond is not None:
        mask = mask & (data["Neurol_cond"] == cond)
    return mask


def give_info_on_endurance(data, category, cond=None, label=""):
    mask = endurance_filter(data, category, cond)
    df_improvements = data.loc[mask, "MCID"]
    Explo.info_about_endurance(df_improvements, neurol_cond=label)


def main(data_path, conds, eng_conds):
    data = load_data(data_path)

    # 1. Count, all conditions, all categories
    give_count(data)

    # 2. Stats of endurance for each category
    categories = [
        ("responders", "répondants"),
        ("nonresponders_improved", "non-répondants améliorés"),
        ("all_improved", "tous améliorés"),
    ]

    for cat_key, cat_label in categories:
        # all conditions
        give_info_on_endurance(
            data, cat_key, cond=None,
            label=f"{cat_label} - toutes conditions",
        )
        # by neurol. condition
        for cond, eng in zip(conds, eng_conds):
            give_info_on_endurance(
                data, cat_key, cond=cond,
                label=f"{cat_label} - {eng}",
            )


if __name__ == "__main__":
    data_path = (
        "/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/"
        "Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/"
        "CHUNantes collaboration/donnees/data_from_dpi/"
        "final_data_matrix_sessions_separated.xlsx"
    )
    conds = ["AVC", "BM", "Autre"]
    eng_conds = ["stroke", "SCI", "other"]
    main(data_path, conds, eng_conds)