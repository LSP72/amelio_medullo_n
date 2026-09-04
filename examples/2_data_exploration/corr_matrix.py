import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog
from amelio_medullo import DataCleaning


def plot_and_save_corr_matrix(data, output_name, output_file_path, corr_method="pearson"):
    correlation_matrix = data.corr(numeric_only=True, method=corr_method)
    plt.figure(figsize=(16, 12))

    sns.heatmap(
        correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 10}, square=True
    )

    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"{output_file_path}/corr_matrix_{output_name}.png", dpi=300, bbox_inches="tight")

    plt.show()


def main(file_path, cols_to_keep, output_name, output_file_path, corr_method="pearson", num=True):
    data = pd.read_excel(file_path)
    data_kept = data[cols_to_keep]
    if num:
        data["Neurol_cond"] = data["Neurol_cond"].replace(["BM", "AVC", "Autre"], [1, 2, 3])
        data["Sex"] = data["Sex"].replace(["M", "F"], [1, 2])
        data.apply(DataCleaning.lesion_level_to_num, axis=1)

    plot_and_save_corr_matrix(data_kept, output_name, output_file_path, corr_method=corr_method)


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select a file", filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
    )
    cols_to_keep_profile_1 = [
        "Sex",
        "Age",
        "Height",
        "Weight",
        "6MWT_m_pre",
        "10MWT_pas_pre",
        "10MWT_sec_pre",
        "6MWT_m_post",
        "delay_injury",
        "delay_loko",
        "functional_level",
        "Artic_hip_flex",
        "Artic_hip_ext",
        "Artic_hip_add",
        "Artic_hip_abd",
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
    ]
    # + Lesion & Neurol_cond
    cols_to_keep_profile_2 = [
        "Sex",
        "Age",
        "Height",
        "Weight",
        "6MWT_m_pre",
        "10MWT_pas_pre",
        "10MWT_sec_pre",
        "6MWT_m_post",
        "delay_injury",
        "delay_loko",
        "functional_level",
    ]
    cols_to_keep_loko_1 = [
        "nb_sessions",
        "duration",
        "Distance_m",
        "Distance_pas",
        "Durée_min",
        "Vitesse_kmh_MIN",
        "Vitesse_kmh_MAX",
        "Vitesse_kmh_MOY",
        "BWS_%_MIN",
        "BWS_%_MAX",
        "BWS_%_MOY",
        "BWS_kg_MIN",
        "BWS_kg_MAX",
        "BWS_kg_MOY",
        "Guidage_G_%_MIN",
        "Guidage_G_%_MAX",
        "Guidage_G_%_MOY",
        "Guidage_D_%_MIN",
        "Guidage_D_%_MAX",
        "Guidage_D_%_MOY",
        "sessions_per_week",
        "6MWT_m_pre",
        "6MWT_m_post",
        "MCID_classes",
        "functional_level",
    ]
    cols_to_keep_loko_2 = [
        "nb_sessions",
        "duration",
        "Distance_m",
        "Distance_pas",
        "Durée_min",
        "Vitesse_kmh_MOY",
        "BWS_%_MOY",
        "Guidage_G_%_MOY",
        "Guidage_D_%_MOY",
        "sessions_per_week",
        "6MWT_m_pre",
        "6MWT_m_post",
        "MCID_classes",
        "functional_level",
    ]
    cols_to_keep_loko_3 = [
        "nb_sessions",
        "duration",
        "Durée_min",
        "Vitesse_kmh_MOY",
        "BWS_%_MOY",
        "step_length",
        "Guidage_%_MOY",
        "sessions_per_week",
        "6MWT_m_pre",
        "6MWT_m_post",
        "MCID_classes",
        "functional_level",
        "cadence",
    ]
    cols_to_keep_merged_1 = [
        "nb_sessions",
        "duration",
        "Durée_min",
        "Vitesse_kmh_MOY",
        "BWS_%_MOY",
        "cadence",
        "step_length",
        "Guidage_%_MOY",
        "sessions_per_week",
        "Neurol_cond",
        "Sex",
        "Age",
        "Nb sessions",
        "functional_level",
        "Lesion_num",
        "BMI",
        "cadence",
    ]

    # DATA_PATH = "/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/CHUNantes collaboration/donnees/data_from_dpi/final_data_matrix_sessions_separated.xlsx"
    DATA_PATH = "/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/CHUNantes collaboration/donnees/data_from_dpi/merged_data_final.xlsx"

    # Features to test.
    # FEATURES = [
    #     "Neurol_cond", "Lesion_num", "Nb sessions", "Sex", "Age", "BMI",
    #     "6MWT_m_pre", "10MWT_pas_pre", "10MWT_sec_pre", "delay_injury",
    #     "delay_loko", "functional_level", "speed",
    # ]
    FEATURES = [
        "duration",
        "Durée_min",
        "Vitesse_kmh_MOY",
        "BWS_%_MOY",
        "step_length",
        "Guidage_%_MOY",
        "sessions_per_week",
        "Neurol_cond",
        "Sex",
        "Nb sessions",
        "BMI",
    ]

    output_name = "profile_selected_features_Spearman"
    output_file_path = "results/data_exploration/correlation_matrices/"

    main(DATA_PATH, FEATURES, output_name, output_file_path, corr_method="spearman", num=True)
