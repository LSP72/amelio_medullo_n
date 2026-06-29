
from sklearn.metrics import roc_auc_score, accuracy_score
from amelio_medullo import Calculus, DataCleaning
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon


def prepare_xgb_data(X, cat_features):
    """XGBoost veut des catégorielles en dtype 'category'."""
    X = X.copy()
    for col in cat_features:
        # NaN reste NaN, mais le type doit être category
        X[col] = X[col].astype("category")
    return X


def compare_catboost_vs_xgboost(X, y, n_splits=5, n_repeats=20, random_state=42):
    """
    Comparaison appariée CatBoost vs XGBoost.
    Mêmes folds, gestion native des manquants ET catégorielles des deux côtés.
    Hyperparamètres équivalents.
    """
    cat_features = [c for c in X.columns if X[c].dtype == "object"]

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    acc_cat, acc_xgb = [], []
    auc_cat, auc_xgb = [], []

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_tr, X_te = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        # ---- CatBoost : catégorielles en string + NaN -> "missing" ----
        X_tr_cb, X_te_cb = X_tr.copy(), X_te.copy()
        for df in (X_tr_cb, X_te_cb):
            df[cat_features] = df[cat_features].fillna("missing").astype(str)

        cb = CatBoostClassifier(
            iterations=500, depth=4, l2_leaf_reg=5,
            eval_metric="AUC", cat_features=cat_features,
            random_seed=42, verbose=0,
        )
        cb.fit(X_tr_cb, y_tr)
        acc_cat.append(accuracy_score(y_te, cb.predict(X_te_cb)))
        auc_cat.append(roc_auc_score(y_te, cb.predict_proba(X_te_cb)[:, 1]))

        # ---- XGBoost : catégorielles en dtype 'category', NaN gérés nativement ----
        X_tr_xgb = prepare_xgb_data(X_tr, cat_features)
        X_te_xgb = prepare_xgb_data(X_te, cat_features)

        xgb = XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            reg_lambda=5, tree_method="hist",
            enable_categorical=True, eval_metric="auc",
            random_state=42, verbosity=0,
        )
        xgb.fit(X_tr_xgb, y_tr)
        acc_xgb.append(accuracy_score(y_te, xgb.predict(X_te_xgb)))
        auc_xgb.append(roc_auc_score(y_te, xgb.predict_proba(X_te_xgb)[:, 1]))

        if (fold_i + 1) % 20 == 0:
            print(f"  Fold {fold_i+1}/{n_splits*n_repeats} fait")

    return np.array(auc_cat), np.array(auc_xgb), np.array(acc_cat), np.array(acc_xgb)


def report_cb_vs_xgb(auc_cat, auc_xgb, acc_cat, acc_xgb):
    diff_auc = auc_cat - auc_xgb
    diff_acc = acc_cat - acc_xgb
    print(f"AUC CatBoost : {auc_cat.mean():.4f} ± {auc_cat.std():.4f}")
    print(f"AUC XGBoost  : {auc_xgb.mean():.4f} ± {auc_xgb.std():.4f}")
    print(f"Différence   : {diff_auc.mean():+.4f} (CatBoost - XGBoost)")
    print(f"CatBoost gagne : {(diff_auc > 0).mean()*100:.0f}% des folds")

    stat, p = wilcoxon(auc_cat, auc_xgb)
    print(f"Wilcoxon signé : p = {p:.4f}")
    if p > 0.05:
        print("=> Pas de différence significative entre CatBoost et XGBoost.")
    else:
        gagnant = "CatBoost" if diff_auc.mean() > 0 else "XGBoost"
        print(f"=> Différence significative en faveur de {gagnant}.")


def main(data_path, cols_to_keep, num=True):
    data = pd.read_excel(data_path)
    if num == True:
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
    auc_cat, auc_xgb, acc_cat, acc_xgb = compare_catboost_vs_xgboost(X, y, n_splits=5, n_repeats=20, random_state=42)
    report_cb_vs_xgb(auc_cat, auc_xgb, acc_cat, acc_xgb)


if __name__ == "__main__":
    data_path = "/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/CHUNantes collaboration/donnees/data_from_dpi/final_data_matrix_sessions_separated.xlsx"
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
        "speed"
    ]
    
    main(data_path, cols_to_keep, num=False)
