import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog


def plot_and_save_corr_matrix(data, output_name, output_file_path):
    correlation_matrix = data.corr(numeric_only=True)
    plt.figure(figsize=(16, 12))

    sns.heatmap(
        correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 10}, square=True
    )

    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"{output_file_path}/corr_matrix_{output_name}.png", dpi=300, bbox_inches="tight")

    plt.show()


def main(file_path, cols_to_keep, output_name, output_file_path):
    data = pd.read_excel(file_path)
    data_kept = data[cols_to_keep]

    plot_and_save_corr_matrix(data_kept, output_name, output_file_path)


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

    output_name = "loko_selected_features_3_with_cadence"
    output_file_path = "results/data_exploration/correlation_matrices/"

    main(file_path, cols_to_keep_loko_3, output_name, output_file_path)
