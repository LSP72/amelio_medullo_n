import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from scipy.stats import wilcoxon
from amelio_medullo import DataCleaning


def compare_imputation_strategies(X, y, n_splits=5, n_repeats=20, random_state=42):
    """
    Compare CatBoost natif vs CatBoost + IterativeImputer.
    SEULE différence : traitement des manquants NUMÉRIQUES.
    Catégoriel identique (NaN -> 'missing') dans les deux conditions.
    Imputer fit sur train uniquement -> pas de fuite.
    """
    cat_features = [c for c in X.columns if X[c].dtype == "object"]
    num_features = [c for c in X.columns if c not in cat_features]

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    cb_params = dict(
        iterations=500, depth=4, l2_leaf_reg=5,
        eval_metric="AUC", random_seed=42, verbose=0,
    )

    auc_native, auc_imputed = [], []

    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        # Traitement catégoriel COMMUN aux deux conditions
        for df in (X_tr, X_te):
            df[cat_features] = df[cat_features].fillna("missing").astype(str)

        # ---- Condition A : numérique natif (CatBoost gère les NaN) ----
        mA = CatBoostClassifier(cat_features=cat_features, **cb_params)
        mA.fit(X_tr, y_tr)
        auc_native.append(roc_auc_score(y_te, mA.predict_proba(X_te)[:, 1]))

        # ---- Condition B : numérique imputé (fit sur TRAIN seulement) ----
        X_tr_b, X_te_b = X_tr.copy(), X_te.copy()
        imp = IterativeImputer(
            n_nearest_features=5, random_state=42, imputation_order="ascending"
        )
        X_tr_b[num_features] = imp.fit_transform(X_tr_b[num_features])
        X_te_b[num_features] = imp.transform(X_te_b[num_features])

        mB = CatBoostClassifier(cat_features=cat_features, **cb_params)
        mB.fit(X_tr_b, y_tr)
        auc_imputed.append(roc_auc_score(y_te, mB.predict_proba(X_te_b)[:, 1]))

    return np.array(auc_native), np.array(auc_imputed)

def report_comparison(auc_native, auc_imputed):
    diff = auc_native - auc_imputed
    print(f"AUC natif    : {auc_native.mean():.4f} ± {auc_native.std():.4f}")
    print(f"AUC imputé   : {auc_imputed.mean():.4f} ± {auc_imputed.std():.4f}")
    print(f"Différence   : {diff.mean():+.4f} (natif - imputé)")
    print(f"Natif gagne  : {(diff > 0).mean()*100:.0f}% des folds")

    stat, p = wilcoxon(auc_native, auc_imputed)
    print(f"Wilcoxon signé : p = {p:.4f}")
    if p > 0.05:
        print("=> Pas de différence significative : "
              "le natif est défendable par parcimonie.")
    else:
        gagnant = "natif" if diff.mean() > 0 else "imputé"
        print(f"=> Différence significative en faveur du {gagnant}.")

def main(data_path, cols_to_keep, num=True):
    data = pd.read_excel(data_path)
    if num == True:
        data["Neurol_cond"] = data["Neurol_cond"].replace(["BM", "AVC", "Autre"], [1, 2, 3])
        data["Sex"] = data["Sex"].replace(["M", "F"], [1, 2])
    data = data.apply(DataCleaning.lesion_level_to_num, axis=1)
    X = data[cols_to_keep]
    y = data["MCID_classes"]
    auc_native, auc_imputed = compare_imputation_strategies(X, y)
    report_comparison(auc_native, auc_imputed)

if __name__ == "__main__":
    data_path = "/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/CHUNantes collaboration/donnees/data_from_dpi/merged_data_final.xlsx"
    # All features
    cols_to_keep_all = [
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
    output_path = "results/catboost_results/merged_data"
    main(data_path, cols_to_keep_all, num=False)
