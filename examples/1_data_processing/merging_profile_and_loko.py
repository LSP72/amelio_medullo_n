import pandas as pd


def main(profile_data_path, loko_data_path, output_path=None):
    profile_data = pd.read_excel(profile_data_path)
    loko_data = pd.read_excel(loko_data_path)
    profile_data = profile_data.loc[:, ~profile_data.columns.str.endswith("_pre")]
    profile_data = profile_data.loc[:, ~profile_data.columns.str.endswith("_post")]
    profile_data = profile_data.loc[:, ~profile_data.columns.str.startswith("ASIA")]
    profile_data.drop(
        columns=[
            "cadence",
            "Unnamed: 39",
            "PRE_POST_BOTH",
            "Bilan entre",
            "Comm_lesion",
            "Durée intervalle",
            "Muscle_assessment",
            "Joint_assessment",
        ],
        inplace=True,
    )
    loko_data.drop(columns=["functional_level", "Unnamed: 0"], inplace=True)

    all_data = loko_data.merge(profile_data, on="IPP", how="left")

    print(all_data.columns.to_list())
    print(all_data.head())

    if output_path:
        all_data.to_excel(f"{output_path}/merged_data_final.xlsx", index=False)


if __name__ == "__main__":
    profile_data_path = (
        "/Volumes/SP UFD U2/PhD/Stage Nantes/data/datasets/final/final_data_matrix_sessions_separated.xlsx"
    )
    loko_data_path = "/Volumes/SP UFD U2/PhD/Stage Nantes/LOKOMAT/loko_final_table_sessions_separated.xlsx"
    output_path = "/Volumes/SP UFD U2/PhD/Stage Nantes/data/datasets/final"
    main(profile_data_path, loko_data_path, output_path)
