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
    # Convert NaN into string "missing'
    X_train[cat_features] = X_train[cat_features].fillna("missing").astype(str)
    X_test[cat_features] = X_test[cat_features].fillna("missing").astype(str)

    # ── 2. Training the CatBoost model ────────────────────────────────────────────────────
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        eval_metric="AUC",
        cat_features=cat_features,
        random_seed=42,
        verbose=100,
    )

    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)  # stopping if no improvement

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
    }


def save_dict(results_dict, output_path, separated_sessions=True):
    pickle_file_name = (
        output_path
        + "/catboost_results_"
        + str(separated_sessions)
        + "all_features_with_patients_with_ankle_missing_removed.pkl"
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


def main(data_path, cols_to_keep, random_state_list, output_path, num=True, drop=False):
    data = pd.read_excel(data_path)
    if num == True:
        data["Neurol_cond"] = data["Neurol_cond"].replace(["BM", "AVC", "Autre"], [1, 2, 3])
        data["Sex"] = data["Sex"].replace(["M", "F"], [1, 2])
        data.apply(DataCleaning.lesion_level_to_num, axis=1)
    y = Calculus.calculate_MCID(data["6MWT_m_pre"], data["6MWT_m_post"], threshold=45)
    if drop:
        data["MCID"] = y
        data.dropna(subset =["Ank_flex_90",	"Ank_flex_180"], axis=0, inplace=True)
        y = data[["MCID"]]
        data.drop(columns=["MCID"], axis=1, inplace=True)
        print(f"Number of included patients: {len(data)}")
    X = data[cols_to_keep].copy()
    X[["10MWT_pas_pre", "10MWT_sec_pre"]].replace([0], ["missing"], inplace=True)
    results_dict = {}
    for rdm_state in random_state_list:
        results_dict[rdm_state] = train_and_test_catboost(X, y, rdm_state)

    save_dict(results_dict, output_path)


if __name__ == "__main__":
    data_path = 
    # cols_to_keep = ["Neurol_cond", "Lesion", "Sex",	"Age",	"Height",	"Weight",	"6MWT_m_pre",	"10MWT_pas_pre",	"10MWT_sec_pre",	"delay_injury",	"delay_loko",
    # "functional_level",	"Artic_hip_flex",	"Artic_hip_ext",	"Artic_hip_add",	"Artic_hip_abd",	"Artic_hip_rot_ext",	"Artic_hip_rot_int",	"Knee_flex",
    # "Knee_ext",	"Ank_flex_90",	"Ank_flex_180",	"Ank_ext",	"H_Flex_ass",	"H_Ext_PP",	"H_abd",	"H_add",	"H_rot_int",	"K_Flex",	"K_Ext",	"A_Dorsiflex_GT",	"A_Plantarflex"]
    cols_to_keep = ["Neurol_cond", "Lesion_num", "Sex",	"Age",	"BMI",	"6MWT_m_pre",	"10MWT_pas_pre",	"10MWT_sec_pre", "delay_injury",	"delay_loko",
    "functional_level",	"Ank_flex_90",	"Ank_flex_180"]
    # cols_to_keep = ['Neurol_cond', 'Lesion_num', 'Nb sessions', 'Sex', 'Age', 'Height', 'Weight', '6MWT_m_pre', '10MWT_pas_pre', '10MWT_sec_pre', 'delay_injury', 'delay_loko', 'functional_level']
    # cols_to_keep = [
    #     "Neurol_cond",
    #     "Lesion_num",
    #     "Nb sessions",
    #     "Sex",
    #     "Age",
    #     "BMI",
    #     "6MWT_m_pre",
    #     "10MWT_pas_pre",
    #     "10MWT_sec_pre",
    #     "delay_injury",
    #     "delay_loko",
    #     "functional_level",
    # ]
    # cols_to_keep = ['Neurol_cond', 'Sex', 'Age', 'BMI', '6MWT_m_pre', '10MWT_pas_pre', '10MWT_sec_pre', 'delay_injury', 'delay_loko', 'functional_level', 'Artic_hip_flex', 'Artic_hip_abd', 'Ank_flex_90', 'Ank_flex_180', 'H_abd', 'Lesion_num']
    # random_state_list = [42, 72]
    random_state_list = np.arange(1, 101)
    # random_state_list = np.random.randint(0, 100, size=30)
    output_path = "results/catboost_results"
    main(data_path, cols_to_keep, random_state_list, output_path, drop=True)
