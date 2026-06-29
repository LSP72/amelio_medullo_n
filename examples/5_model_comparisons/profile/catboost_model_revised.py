import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    roc_auc_score,
    f1_score,
    accuracy_score,
)
from amelio_medullo import Calculus, DataCleaning
import pickle as pkl


def train_and_test_catboost(X, y, rdm_state):
    """Train + evaluate one CatBoost on a single stratified split.

    Returns the result dict, or None if this split is degenerate (e.g. the
    test set ends up single-class, which makes AUC undefined). Returning None
    lets the caller skip the split instead of crashing the whole run — relevant
    only for the small ambulant subgroup.
    """
    cat_features = [col for col in X.columns if X[col].dtype == "object"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rdm_state, stratify=y
    )
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[cat_features] = X_train[cat_features].fillna("missing").astype(str)
    X_test[cat_features] = X_test[cat_features].fillna("missing").astype(str)

    # Guard: AUC is undefined if the test set has a single class. With ~4 test
    # patients in the ambulant run this happens; skip rather than crash.
    if y_test.nunique() < 2:
        print(f"  rs={rdm_state}: test set is single-class, skipped.")
        return None

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.03,
        depth=3,
        cat_features=cat_features,
        random_seed=42,
        verbose=0,  # silenced — 100 iterations with verbose=100 floods stdout
    )
    model.fit(X_train, y_train)
    feature_imp_df = model.get_feature_importance(prettified=True)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_test = roc_auc_score(y_test, y_pred_proba)
    acc_test = accuracy_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred)

    print(f"  rs={rdm_state:>3} | AUC={auc_test:.4f} | Acc={acc_test:.4f} | F1={f1_test:.4f}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    plt.title("Matrice de confusion")
    plt.tight_layout()
    plt.close(fig)  # close every iteration's figure — prevents 100 open figures

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
        "accuracy_test": acc_test,
        "f1_score": f1_test,
        "true_values": y_test,
        "shap_values": shap_values,
        "model_fts_imp": feature_imp_df,
        "classif_report": classification_report(y_test, y_pred),
    }


def save_dict(results_dict, output_path, run_label):
    out_dir = Path(output_path) / "monte_carlo"
    out_dir.mkdir(parents=True, exist_ok=True)  # create dir before writing
    pickle_file = out_dir / f"catboost_results_{run_label}_by_combi.pkl"
    with open(pickle_file, "wb") as f:
        pkl.dump(results_dict, f)
    print(f"Saved [{run_label}] -> {pickle_file}")


def prepare_data(data_path, cols_to_keep, num=False, mode="all"):
    """Build X, y for one of the three analysis modes.

    The LABEL is identical across modes (always derived from 6MWT_m_pre and
    6MWT_m_post); only the FEATURE representation / population changes:

      "all"           : all patients, 6MWT_m_pre kept as a continuous feature,
                        no is_ambulant. (Run 1 — the original analysis.)
      "is_ambulant"   : all patients, 6MWT_m_pre REMOVED and REPLACED by a
                        binary is_ambulant (= 6MWT_m_pre > 0). (Run 2.)
      "ambulant_only" : only patients with 6MWT_m_pre > 0, 6MWT_m_pre kept as a
                        continuous feature. (Run 3 — n is small, exploratory.)
    """
    data = pd.read_excel(data_path)
    if num:
        data["Neurol_cond"] = data["Neurol_cond"].replace(["BM", "AVC", "Autre"], [1, 2, 3])
        data["Sex"] = data["Sex"].replace(["M", "F"], [1, 2])
    data = data.apply(DataCleaning.lesion_level_to_num, axis=1)

    cols = list(cols_to_keep)  # never mutate the caller's list

    if mode == "all":
        pass  # 6MWT_m_pre stays, all rows, no flag

    elif mode == "is_ambulant":
        data["is_ambulant"] = (data["6MWT_m_pre"] != 0).astype(int)
        if "6MWT_m_pre" in cols:
            cols.remove("6MWT_m_pre")   # replaced, not kept alongside
        if "is_ambulant" not in cols:
            cols.append("is_ambulant")

    elif mode == "ambulant_only":
        data = data[data["6MWT_m_pre"] != 0].copy()  # subset rows, keep 6MWT_m_pre

    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    X = data[cols].copy()
    y = Calculus.calculate_MCID_2(data, default_threshold=45)  
    return X, y["MCID_classes"]


def run_analysis(data_path, cols_to_keep, random_state_list, output_path,
                 num=False, mode="all"):
    X, y = prepare_data(data_path, cols_to_keep, num=num, mode=mode)

    class_counts = y.value_counts().to_dict()
    print("=" * 60)
    print(f"RUN [{mode}] | N={len(X)} | class balance={class_counts}")
    print(f"features used: {list(X.columns)}")
    print("=" * 60)

    # Upfront viability check: stratified splitting needs >= 2 members per class.
    if y.nunique() < 2 or y.value_counts().min() < 2:
        print(f"RUN [{mode}] SKIPPED — a class has < 2 members; "
              f"stratified split is impossible. Treat this subgroup descriptively, "
              f"not as a classifier.")
        return None

    results_dict = {}
    for rdm_state in random_state_list:
        res = train_and_test_catboost(X, y, rdm_state)
        if res is None:
            continue  # degenerate split skipped
        res["list_of_features"] = list(X.columns)  # actual columns, not the input list
        res["mode"] = mode
        results_dict[rdm_state] = res

    n_ok = len(results_dict)
    print(f"RUN [{mode}] finished: {n_ok}/{len(random_state_list)} iterations succeeded.")
    if n_ok == 0:
        print(f"RUN [{mode}] produced no usable iterations — nothing saved.")
        return None

    save_dict(results_dict, output_path, run_label=mode)
    return results_dict


if __name__ == "__main__":
    data_path = (
        "/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/"
        "Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/"
        "CHUNantes collaboration/donnees/data_from_dpi/final_data_matrix_sessions_separated.xlsx"
    )

    # Base feature set — includes 6MWT_m_pre. The three modes transform this
    # list/population as documented in prepare_data; don't add is_ambulant here.
    cols_to_keep = [
        "Lesion_num",
        "Nb sessions",
        "Sex",
        "BMI",
        "6MWT_m_pre",
        "delay_loko",
        "functional_level",
        "speed",
    ]

    random_state_list = np.arange(1, 101)
    output_path = "results/catboost_results/profile_data"

    # Run 1 — original analysis: all patients, continuous 6MWT_m_pre
    run_analysis(data_path, cols_to_keep, random_state_list, output_path,
                 num=False, mode="all")

    # Run 2 — 6MWT_m_pre replaced by binary is_ambulant, all patients
    run_analysis(data_path, cols_to_keep, random_state_list, output_path,
                 num=False, mode="is_ambulant")

    # Run 3 — ambulant patients only, continuous 6MWT_m_pre (small N, exploratory)
    run_analysis(data_path, cols_to_keep, random_state_list, output_path,
                 num=False, mode="ambulant_only")