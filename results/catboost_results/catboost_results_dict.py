import pickle as pkl
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc
from amelio_medullo import DataCleaning

# dict_path_1 = "/Users/mathildetardif/Documents/Python/Biomarkers/amelio_medullo_n/results/catboost_results/catboost_results_True.pkl"
# dict_path_2 = "/Users/mathildetardif/Documents/Python/Biomarkers/amelio_medullo_n/results/catboost_results/catboost_results_True_selected_features.pkl"
# with open(dict_path_1, "rb") as file:
#     dict_1 = pkl.load(file)

# with open(dict_path_2, "rb") as file:
#     dict_2 = pkl.load(file)

# acc_1, acc_2 = [], []
# for key in dict_1:
#     y_pred = dict_1[key]["predictions"]
#     y_true = dict_1[key]["true_values"]
#     acc_1.append(accuracy_score(y_true, y_pred))

# for key in dict_2:
#     y_pred = dict_2[key]["predictions"]
#     y_true = dict_2[key]["true_values"]
#     acc_2.append(accuracy_score(y_true, y_pred))

# print("Results for all features:")
# print(f"Accuracy: {np.mean(acc_1):.2f} ± {np.std(acc_1):.2f}")
# print("Results for selected features:")
# print(f"Accuracy: {np.mean(acc_2):.2f} ± {np.std(acc_2):.2f}")


## ---------- Avec SHAP et sur 100 itérations ----------
data_path = "/Volumes/SP UFD U2/PhD/Stage Nantes/data/datasets/final/final_data_matrix_sessions_separated.xlsx"
data = pd.read_excel(data_path)
data["Neurol_cond"] = data["Neurol_cond"].replace(["BM", "AVC", "Autre"], [1, 2, 3])
data["Sex"] = data["Sex"].replace(["M", "F"], [1, 2])
data.apply(DataCleaning.lesion_level_to_num, axis=1)
# cols_to_keep = ['Neurol_cond', 'Lesion_num', 'Nb sessions', 'Sex', 'Age', 'Height', 'Weight', '6MWT_m_pre', '10MWT_pas_pre', '10MWT_sec_pre', 'delay_injury', 'delay_loko', 'functional_level']
# cols_to_keep = ['Neurol_cond', 'Lesion_num', 'Nb sessions', 'Sex', 'Age', 'BMI', '6MWT_m_pre', '10MWT_pas_pre', '10MWT_sec_pre', 'delay_injury', 'delay_loko', 'functional_level']
cols_to_keep = [
    "Neurol_cond",
    "Lesion",
    "Sex",
    "Age",
    "Height",
    "Weight",
    "6MWT_m_pre",
    "10MWT_pas_pre",
    "10MWT_sec_pre",
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
# cols_to_keep = ['Neurol_cond', 'Sex', 'Age', 'BMI', '6MWT_m_pre', '10MWT_pas_pre', '10MWT_sec_pre', 'delay_injury', 'delay_loko', 'functional_level', 'Artic_hip_flex', 'Artic_hip_abd', 'Ank_flex_90', 'Ank_flex_180', 'H_abd', 'Lesion_num']
X = data[cols_to_keep]

dict_path_3 = "/Users/mathildetardif/Documents/Python/Biomarkers/amelio_medullo_n/results/catboost_results/catboost_results_True_all_features_numerical_100it_with_shap.pkl"
with open(dict_path_3, "rb") as file:
    dict_3 = pkl.load(file)

# 1. --- Initialisation ---
acc_3 = []
aucs_3 = []
tprs = []
mean_fpr = np.linspace(0, 1, 100)
shap_records, feature_imp_records = [], []

# 2. --- Loop to collect all info ---
for rdm_state, res in dict_3.items():
    # a. --- Extracting data from the it ---
    test_idx = res["index_test"]
    y_pred = res["predictions"]
    # y_proba = res["proba_predictions"]
    y_true = res["true_values"]
    # roc_auc = res["auc_test"]
    shap_vals = res["shap_values"]

    acc_3.append(accuracy_score(y_true, y_pred))
    # aucs_3.append(roc_auc)

    # b. --- Calculatin & interpolating for ROC curve ---
    # fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    # interp_tpr = np.interp(mean_fpr, fpr, tpr)
    # interp_tpr[0] = 0.0
    # tprs.append(interp_tpr)

    # c. --- Saving for SHAP ---
    df_shap = pd.DataFrame(shap_vals, index=test_idx, columns=cols_to_keep)
    shap_records.append(df_shap)

    # d. --- Saving for Feature Importance ---
    feature_imp_df = res["model_fts_imp"]
    feature_imp_df = feature_imp_df.set_index("Feature Id")
    feature_imp_df = feature_imp_df.rename(columns={"Importances": rdm_state})
    feature_imp_records.append(feature_imp_df)

# --- Printing metrics ---
print("Results for selected features (100 it.):")
print(f"Accuracy: {np.mean(acc_3):.3f} ± {np.std(acc_3):.3f}")
print(f"AUC:      {np.mean(aucs_3):.3f} ± {np.std(aucs_3):.3f}")

# Concaténer toutes les itérations
all_shap = pd.concat(shap_records)
all_feature_imp = pd.concat(feature_imp_records, axis=1)

# Moyenne des valeurs SHAP pour chaque patient (sur toutes les fois où il était dans X_test)
mean_shap_per_patient = all_shap.groupby(all_shap.index).mean()
mean_feature_imp = all_feature_imp.mean(axis=1).sort_values(ascending=False)
std_feature_imp = all_feature_imp.std(axis=1)

# Moyenne globale sur tous les patients
mean_shap_global = mean_shap_per_patient.mean(axis=0).sort_values(key=abs, ascending=False)
print(mean_shap_global)

# Récupérer X correspondant (moyenne des X_test aussi par patient)
X_test_mean = X.loc[mean_shap_per_patient.index]

# ====================
# SHAP visualisations
# ====================
shap.summary_plot(
    mean_shap_per_patient.values, X_test_mean, plot_type="bar", title="Importance SHAP moyenne (100 itérations)"
)

shap.summary_plot(mean_shap_per_patient.values, X_test_mean, title="Importance SHAP moyenne (100 itérations)")

# ====================================================
# Feature importance from Catboost model visualisation
# ====================================================

fig, ax = plt.subplots(figsize=(8, 6))

bars = ax.barh(
    mean_feature_imp.index,
    mean_feature_imp.values,
    xerr=std_feature_imp[mean_feature_imp.index],  # barre d'erreur
    color="steelblue",
    ecolor="gray",
    capsize=3,
)

# Adding info texts (Mean ± SD) at the end of each bar
for i, (feature, mean_val) in enumerate(mean_feature_imp.items()):
    std_val = std_feature_imp[feature]

    text_str = f"{mean_val:.2f} ± {std_val:.2f}"

    # x position:
    marge = 0.02 * max(mean_feature_imp.values)  # 2% de la valeur max pour aérer
    x_pos = mean_val + std_val + marge

    ax.text(x_pos, i, text_str, va="center", ha="left", fontsize=9, color="black")

ax.invert_yaxis()
ax.set_xlabel("Mean Importance")
ax.set_title("Feature Importance CatBoost (mean on 100 iterations)")

# Adjusting X-axis
xmax = max(mean_feature_imp.values + std_feature_imp[mean_feature_imp.index])
ax.set_xlim(0, xmax * 1.25)

plt.tight_layout()
# plt.savefig("results/catboost_results/feature_importance_from_CB_model.svg")
plt.show()

# ===============================
# --- Plotting mean ROC curve ---
# ===============================
# mean_tpr = np.mean(tprs, axis=0)
# mean_tpr[-1] = 1.0 # S'assurer qu'elle finit bien à 1
# std_tpr = np.std(tprs, axis=0)

# fig, ax = plt.subplots(figsize=(8, 6))

# # Plot mean curve
# ax.plot(mean_fpr, mean_tpr, color='blue', lw=2,
#         label=f'Mean ROC (AUC = {np.mean(aucs_3):.2f} $\pm$ {np.std(aucs_3):.2f})')

# # Plot standard deviation
# tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3, label='$\pm$ 1 std')

# # Plot Random line
# ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='red', alpha=0.8, label='Random')

# ax.set_xlim([-0.05, 1.05])
# ax.set_ylim([-0.05, 1.05])
# ax.set_xlabel('False Positive Rate (FPR)')
# ax.set_ylabel('True Positive Rate (TPR)')
# ax.set_title('ROC curve for 100 iterations')
# ax.legend(loc="lower right")
# plt.tight_layout()
# plt.show()
