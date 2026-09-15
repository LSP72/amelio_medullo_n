import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import time
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_auc_score, f1_score, precision_score
from amelio_medullo import Calculus, DataCleaning
import pickle as pkl


def train_and_test_catboost(X, y, rdm_state):
    cat_features = [col for col in X.columns if X[col].dtype == "object"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rdm_state, stratify=y)
    X_train[cat_features] = X_train[cat_features].fillna("missing").astype(str)
    X_test[cat_features] = X_test[cat_features].fillna("missing").astype(str)

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.03,
        depth=3,
        cat_features=cat_features,
        random_seed=42,
        verbose=150,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred, average="binary")
    prec = precision_score(y_test, y_pred, average="binary")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    return {
        "random_state": rdm_state,
        "model": model,
        "index_train": X_train.index,
        "index_test": y_test.index,
        "predictions": y_pred,
        "proba_predictions": y_pred_proba,
        "auc_test": auc,
        "true_values": y_test,
        "shap_values": shap_values,
        "model_fts_imp": model.get_feature_importance(prettified=True),
        "f1_score": f1,
        "precision": prec,
        "classif_report": classification_report(y_test, y_pred),
    }


def summarize_results(results_dict):
    """Means and standard deviations of metrics across all Monte Carlo iterations."""
    aucs = [v["auc_test"] for v in results_dict.values()]
    f1s = [v["f1_score"] for v in results_dict.values()]
    precs = [v["precision"] for v in results_dict.values()]

    summary = (
        pd.DataFrame(
            {
                "Metrics": ["AUC", "F1", "Precision"],
                "Mean": [np.mean(aucs), np.mean(f1s), np.mean(precs)],
                "Std Dev": [np.std(aucs), np.std(f1s), np.std(precs)],
                "Min": [np.min(aucs), np.min(f1s), np.min(precs)],
                "Max": [np.max(aucs), np.max(f1s), np.max(precs)],
            }
        )
        .set_index("Metrics")
        .round(4)
    )

    print("\n=== Results Monte Carlo ===")
    print(summary.to_string())

    # Distributions
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, values, label in zip(axes, [aucs, f1s, precs], ["AUC", "F1", "Precision"]):
        ax.hist(values, bins=20, edgecolor="black")
        ax.axvline(np.mean(values), color="red", linestyle="--", label=f"Mean = {np.mean(values):.3f}")
        ax.set_title(label)
        ax.legend()
    plt.suptitle("Distribution of metrics — Monte Carlo")
    plt.tight_layout()
    plt.show()

    return summary


def save_dict(results_dict, output_path, separated_sessions=True):
    pickle_file_name = (
        output_path
        + "/monte_carlo/catboost_results_separated_sessions_is_"
        + str(separated_sessions)
        + "_selected_features_monte_carlo.pkl"
    )
    with open(pickle_file_name, "wb") as file:
        pkl.dump(results_dict, file)


def main(data_path, cols_to_keep, random_state_list, output_path, num=True):
    data = pd.read_excel(data_path)
    if num:
        data["Neurol_cond"] = data["Neurol_cond"].replace(["BM", "AVC", "Autre"], [1, 2, 3])
        data["Sex"] = data["Sex"].replace(["M", "F"], [1, 2])
    data = data.apply(DataCleaning.lesion_level_to_num, axis=1)
    if "speed" in data.columns.to_list():
        data["speed"].replace([np.inf, -np.inf], np.nan, inplace=True)

    X = data[cols_to_keep].copy()
    if "10MWT_pas_pre" in X.columns.to_list():
        X[["10MWT_pas_pre", "10MWT_sec_pre"]].replace([np.nan], ["missing"], inplace=True)
    y = Calculus.calculate_MCID_2(data, default_threshold=45)
    y = y["MCID_classes"]

    results_dict = {}
    start = time.time()
    for rdm_state in random_state_list:
        results_dict[rdm_state] = train_and_test_catboost(X, y, rdm_state)
        results_dict[rdm_state]["list_of_features"] = cols_to_keep
    end = time.time()
    print(f"\nTotal time {end - start:.2f} seconds.")

    summarize_results(results_dict)  # ← moyennes affichées ici
    # save_dict(results_dict, output_path)


if __name__ == "__main__":
    data_path = "/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/CHUNantes collaboration/donnees/data_from_dpi/final_data_matrix_sessions_separated.xlsx"
    # cols_to_keep = ["Neurol_cond", "Lesion", "Sex",	"Age",	"Height",	"Weight",	"6MWT_m_pre",	"10MWT_pas_pre",	"10MWT_sec_pre",	"delay_injury",	"delay_loko",
    # "functional_level",	"Artic_hip_flex",	"Artic_hip_ext",	"Artic_hip_add",	"Artic_hip_abd",	"Artic_hip_rot_ext",	"Artic_hip_rot_int",	"Knee_flex",
    # "Knee_ext",	"Ank_flex_90",	"Ank_flex_180",	"Ank_ext",	"H_Flex_ass",	"H_Ext_PP",	"H_abd",	"H_add",	"H_rot_int",	"K_Flex",	"K_Ext",	"A_Dorsiflex_GT",	"A_Plantarflex"]
    # cols_to_keep = ['Neurol_cond', 'Lesion_num', 'Nb sessions', 'Sex', 'Age', 'Height', 'Weight', '6MWT_m_pre', '10MWT_pas_pre', '10MWT_sec_pre', 'delay_injury', 'delay_loko', 'functional_level']
    cols_to_keep = [
        "Neurol_cond",
        "Lesion_num",
        "Nb sessions",
        "Sex",
        "Age",
        "BMI",
        "6MWT_m_pre",
        # "10MWT_pas_pre",
        # "10MWT_sec_pre",
        "delay_injury",
        "delay_loko",
        "functional_level",
        "speed",
    ]
    # cols_to_keep = ['Neurol_cond', 'Sex', 'Age', 'BMI', '6MWT_m_pre', '10MWT_pas_pre', '10MWT_sec_pre', 'delay_injury', 'delay_loko', 'functional_level', 'Artic_hip_flex', 'Artic_hip_abd', 'Ank_flex_90', 'Ank_flex_180', 'H_abd', 'Lesion_num']
    # random_state_list = [42, 72]
    random_state_list = np.arange(1, 101)
    # random_state_list = np.random.randint(0, 100, size=30)
    output_path = "results/catboost_results/profile_data"
    main(data_path, cols_to_keep, random_state_list, num=False, output_path=output_path)
