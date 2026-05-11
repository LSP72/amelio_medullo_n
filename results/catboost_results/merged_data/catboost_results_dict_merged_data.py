import pickle as pkl
import numpy as np
import pandas as pd
import shap
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc
from amelio_medullo import DataCleaning

from pathlib import Path

def save_texte(name, dict, accuracy_mean, accuracy_std, auc_mean=False, auc_std=False):
    out_dir = Path("results/catboost_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    if auc_mean and auc_std:
        results_text = f"""
        Results for: {name}
        Number of iterations: {len(dict.items())}

        Accuracy:
        Mean = {accuracy_mean:.3f}
        Std  = {accuracy_std:.3f}

        AUC:
        Mean = {auc_mean:.3f}
        Std  = {auc_std:.3f}
        """
    else:
        results_text = f"""
        Results for: {name}
        Number of iterations: {len(dict.items())}

        Accuracy:
        Mean = {accuracy_mean:.3f}
        Std  = {accuracy_std:.3f}
        """

    with open(out_dir / f"{name}_metrics.txt", "w", encoding="utf-8") as f:
        f.write(results_text)

## ---------- Avec SHAP et sur x itérations ----------
data_path = 
data = pd.read_excel(data_path)
# data["Neurol_cond"] = data["Neurol_cond"].replace(["BM", "AVC", "Autre"], [1, 2, 3])
# data["Sex"] = data["Sex"].replace(["M", "F"], [1, 2])
# data.apply(DataCleaning.lesion_level_to_num, axis=1)

dict_path_3 = 
name = Path(dict_path_3).stem
print('- '*10)
print(f"Name of tries: {name}")
print('- '*10)
with open(dict_path_3, "rb") as file:
    dict_3 = pkl.load(file)
first_key = next(iter(dict_3))
cols_to_keep = dict_3[first_key]["model"].feature_names_
X = data[cols_to_keep]

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
    y_proba = res["proba_predictions"]
    y_true = res["true_values"]
    roc_auc = res["auc_test"]
    shap_vals = res["shap_values"]

    acc_3.append(accuracy_score(y_true, y_pred))
    aucs_3.append(roc_auc)

    # b. --- Calculatin & interpolating for ROC curve ---
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)

    # c. --- Saving for SHAP ---
    df_shap = pd.DataFrame(shap_vals, index=test_idx, columns=cols_to_keep)
    shap_records.append(df_shap)

    # d. --- Saving for Feature Importance ---
    feature_imp_df = res["model_fts_imp"]
    feature_imp_df = feature_imp_df.set_index("Feature Id")
    feature_imp_df = feature_imp_df.rename(columns={"Importances": rdm_state})
    feature_imp_records.append(feature_imp_df)

# --- Printing metrics ---
print(f"Results for selected features ({len(dict_3.items())} it.):")
print(f"Accuracy: {np.mean(acc_3):.3f} ± {np.std(acc_3):.3f}")
print(f"AUC:      {np.mean(aucs_3):.3f} ± {np.std(aucs_3):.3f}")
save_texte(name, dict_3, accuracy_mean=np.mean(acc_3), accuracy_std=np.std(acc_3), auc_mean=np.mean(aucs_3), auc_std=np.std(aucs_3))

# Concaténer toutes les itérations
all_shap = pd.concat(shap_records)
all_feature_imp = pd.concat(feature_imp_records, axis=1)

# Moyenne des valeurs SHAP pour chaque patient (sur toutes les fois où il était dans X_test)
mean_shap_per_patient = all_shap.groupby(all_shap.index).mean()
mean_shap_per_patient = all_shap.groupby(all_shap.index).mean()
std_shap_per_patient = all_shap.groupby(all_shap.index).std()
mean_feature_imp = all_feature_imp.mean(axis=1).sort_values(ascending=False)
std_feature_imp = all_feature_imp.std(axis=1)

# Moyenne globale sur tous les patients
abs_mean_shap_global = mean_shap_per_patient.abs().mean(axis=0).sort_values(key=abs, ascending=False)
mean_shap_global = mean_shap_per_patient.mean(axis=0).sort_values(key=abs, ascending=False)
std_shap_global = std_shap_per_patient.mean(axis=0).loc[mean_shap_global.index]
print(f"Global mean shap values:\n{pd.concat([abs_mean_shap_global, std_shap_global], keys=["Abs means", "StD"], axis=1).to_markdown()}")

print(f"Global feature importance values from CB:\n{pd.concat([mean_feature_imp, std_feature_imp], keys=["Means", "StD"], axis=1).to_markdown()}")

# Récupérer X correspondant (moyenne des X_test aussi par patient)
X_test_mean = X.loc[mean_shap_per_patient.index]

# ====================
# SHAP visualisations
# ====================
# Collecting feature names
feature_data = pd.read_excel("/Volumes/SP UFD U2/PhD/Stage Nantes/data/datasets/final/feature_names.xlsx")
feature_name_dict = dict(zip(feature_data["features"], feature_data["features_names"]))
features_names = [feature_name_dict.get(feature, feature) for feature in X_test_mean.columns.to_list()]

# Plotting SHAP plots
# --- Bar plot ---
plt.close("all")
shap.summary_plot(
    mean_shap_per_patient.values,
    X_test_mean,
    feature_names=features_names,
    plot_type="bar",
    title=f"Importance SHAP moyenne ({len(dict_3.items())} itérations)",
    show=False,
)
plt.tight_layout()
plt.gcf().savefig(
    f"results/catboost_results/{name}_shap_bar.svg",
    format="svg",
    bbox_inches="tight"
)
plt.close()

# --- Summary plot ---
plt.close("all")
shap.summary_plot(
    mean_shap_per_patient.values,
    X_test_mean,
    feature_names=features_names,
    title=f"Importance SHAP moyenne ({len(dict_3.items())} itérations)",
    show=False
)
plt.tight_layout()
plt.gcf().savefig(
    f"results/catboost_results/{name}_shap_summary.svg",
    format="svg",
    bbox_inches="tight",
)
plt.close()


# ==================
# SHAP by categories
# ==================

if "Neurol_cond" in X_test_mean.columns.to_list():
    cond_groups = X_test_mean["Neurol_cond"].map({1: "SCI", 2: "Stroke", 3: "Others"}).to_numpy()

    # Safety check
    assert len(cond_groups) == mean_shap_per_patient.shape[0]

    X_test_mean_copy = X_test_mean.drop(columns=["Neurol_cond"], axis=1)
    mean_shap_per_patient_copy = mean_shap_per_patient.drop(columns=["Neurol_cond"], axis=1)
    feature_name_dict = dict(zip(feature_data["features"], feature_data["features_names"]))
    features_names_copy = [feature_name_dict.get(feature, feature) for feature in X_test_mean_copy.columns.to_list()]

    # # Create SHAP Explanation object from your averaged SHAP values
    shap_exp = shap.Explanation(
        values=mean_shap_per_patient_copy.values, data=X_test_mean_copy.values, feature_names=features_names_copy
    )
    plt.close("all")
    shap.plots.bar(shap_exp.cohorts(cond_groups).abs.mean(0), max_display=len(features_names_copy), show=False)
    plt.tight_layout()
    plt.gcf().savefig(
        f"results/catboost_results/{name}_shap_summary_grouped_by_cond.svg",
        format="svg",
        bbox_inches="tight"
    )
    plt.close()

# ====================================================
# Feature importance from Catboost model visualisation
# ====================================================
original_features = mean_feature_imp.index.to_list()

# Noms jolis dans le même ordre
pretty_feature_names = [
    feature_name_dict.get(feature, feature)
    for feature in original_features
]

y_pos = np.arange(len(original_features))

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

ax.set_yticks(y_pos)
ax.set_yticklabels(pretty_feature_names)

ax.invert_yaxis()
ax.set_xlabel("Mean Importance")
ax.set_title(f"Feature Importance CatBoost (mean on {len(dict_3.items())} iterations)")

# Adjusting X-axis
xmax = max(mean_feature_imp.values + std_feature_imp[mean_feature_imp.index])
ax.set_xlim(0, xmax * 1.25)

plt.tight_layout()
plt.savefig(f"results/catboost_results/{name}_feature_importance_from_CB_model.svg")
plt.show()
plt.close()

# ===============================
# --- Plotting mean ROC curve ---
# ===============================
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0  # S'assurer qu'elle finit bien à 1
std_tpr = np.std(tprs, axis=0)

fig, ax = plt.subplots(figsize=(8, 6))

# # Plot mean curve
ax.plot(
    mean_fpr, mean_tpr, color="blue", lw=2, label=f"Mean ROC (AUC = {np.mean(aucs_3):.2f} $\pm$ {np.std(aucs_3):.2f})"
)

# # Plot standard deviation
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.3, label="$\pm$ 1 std")

# # Plot Random line
ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="red", alpha=0.8, label="Random")

ax.set_xlim([-0.05, 1.05])
ax.set_ylim([-0.05, 1.05])
ax.set_xlabel("False Positive Rate (FPR)")
ax.set_ylabel("True Positive Rate (TPR)")
ax.set_title("ROC curve for 100 iterations")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f'results/catboost_results/{name}_ROC_AUC.svg')
plt.show()
plt.close()
