import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_auc_score
from amelio_medullo import Calculus, DataCleaning
import pickle as pkl


def train_and_test_catboost(X, y, rdm_state):
    # Automatically detect categorical column
    cat_features = [col for col in X.columns if X[col].dtype == "object"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rdm_state, stratify=y)
    X_train = X_train.copy()
    # Convert NaN into string "missing'
    X_train[cat_features] = X_train[cat_features].fillna("missing").astype(str)
    X_test[cat_features] = X_test[cat_features].fillna("missing").astype(str)

    # ── 2. Training the CatBoost model ────────────────────────────────────────────────────
    model = CatBoostClassifier(
        iterations=500,
        # learning_rate=0.06,
        depth=6,
        eval_metric="AUC",
        cat_features=cat_features,
        random_seed=42,
        verbose=100,
    )

    model.fit(X_train, y_train)  # stopping if no improvement
                                # eval_set=(X_test, y_test) Fuite de données possible ici
    feature_imp_df = model.get_feature_importance(prettified=True)

    # ── 3. Validation ────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_test = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC Score sur le test set : {auc_test:.4f}")  # Pour l'afficher dans la console

    print(classification_report(y_test, y_pred))
    # print(f"True labels: {y_test}")
    # print(f"Predicted labels: {y_pred}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    plt.title("Matrice de confusion")
    plt.tight_layout()
    # plt.show()

    #     # ── 4. SHAP values ───────────────────────────────────────────────────────────
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    return {
        "random_state": rdm_state,
        "model": model,
        "index_train": X_train.index,
        "index_test": y_test.index,
        "predictions": y_pred,
        "proba_predictions": y_pred_proba,
        "auc_test": auc_test,
        "true_values": y_test,
        "shap_values": shap_values,
        "model_fts_imp": feature_imp_df,
        "f1_score":f1_score(y_test, y_pred),
        "classif_report": classification_report(y_test, y_pred)
    }


def save_dict(results_dict, output_path, num):
    pickle_file_name = (f"{output_path}/catboost_results_merged_data_selected_features_with_no_fuite.pkl"
    )
    with open(pickle_file_name, "wb") as file:
        pkl.dump(results_dict, file)


def shap_plot(shap_values, X_test):
    shap.summary_plot(shap_values, X_test, plot_type="bar", title="Importance globale (SHAP)")
    # shap.summary_plot(
    #         shap_values,
    #         X_test,
    #         plot_size=(8, 10),
    #         show=True,
    #     )


def main(data_path, cols_to_keep, random_state_list, output_path, num=True):
    data = pd.read_excel(data_path)
    if num == True:
        data["Neurol_cond"] = data["Neurol_cond"].replace(["BM", "AVC", "Autre"], [1, 2, 3])
        data["Sex"] = data["Sex"].replace(["M", "F"], [1, 2])
    data = data.apply(DataCleaning.lesion_level_to_num, axis=1)
    X = data[cols_to_keep]
    y = data["MCID_classes"]
    results_dict = {}
    for rdm_state in random_state_list:
        print(f"Split number: {rdm_state}")
        results_dict[rdm_state] = train_and_test_catboost(X, y, rdm_state)
        results_dict[rdm_state]["list_of_features"] = cols_to_keep

    save_dict(results_dict, output_path, num)


if __name__ == "__main__":
    data_path = "/Volumes/SP UFD U2/PhD/Stage Nantes/data/datasets/final/merged_data_final.xlsx"
    # cols_to_keep_0 = [
    #     "nb_sessions",	"duration",	"Durée_min", "Vitesse_kmh_MOY", "BWS_%_MOY", "cadence", "step_length",
    #     "Guidage_%_MOY", "sessions_per_week", "6MWT_m_pre", "Neurol_cond", "Sex", "Age", "Nb sessions",
    #     "delay_injury", "delay_loko", "functional_level", "Lesion_num", "BMI"
    #     ]
    # Removed all pre info, only let personal info
    cols_to_keep_1 = [
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
    ]
    # Removed highly correlated features (i.e., cadence (vitesse), )
    cols_to_keep_2 = [
        "nb_sessions",
        # "duration",
        "Durée_min",
        "Vitesse_kmh_MOY",
        "BWS_%_MOY",
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
    ]
    # features for loko intervention only after lasso
    cols_to_keep_3 = [
        "nb_sessions",
        "BWS_%_MOY",
        "Vitesse_kmh_MOY", # ajoutée manuellement
        "Guidage_%_MOY",
        "6MWT_m_pre",
        "cadence"
    ]
    # Selected features after LASSO stabilised
    cols_to_keep_4 = ["Nb sessions", "Neurol_cond", "Sex",
                      "BWS_%_MOY", "sessions_per_week", "Guidage_%_MOY",
                      "Durée_min"]
    # Selected features after profile analyses
    cols_to_keep_5 = [
        "nb_sessions",
        "duration",
        "Durée_min",
        "Vitesse_kmh_MOY",
        "BWS_%_MOY",
        "step_length",
        "Guidage_%_MOY",
        "sessions_per_week",
        "Neurol_cond",
        # "Sex",
        # "Nb sessions",
        "BMI",
        # "cadence"
    ]
    cols_to_keep_5_b = [
        # "nb_sessions",
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
        # "cadence"
    ]
    # random_state_list = [42, 72]
    random_state_list = np.arange(1, 101)
    output_path = "results/catboost_results/merged_data"
    main(data_path, cols_to_keep_5_b, random_state_list, output_path, num=False)
