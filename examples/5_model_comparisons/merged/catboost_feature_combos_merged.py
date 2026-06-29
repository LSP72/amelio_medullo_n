import os
from itertools import combinations
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from amelio_medullo import DataCleaning
import pickle as pkl


def train_single_split(X, y, rdm_state):
    cat_features = [col for col in X.columns if X[col].dtype == "object"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rdm_state, stratify=y
    )
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[cat_features] = X_train[cat_features].fillna("missing").astype(str)
    X_test[cat_features] = X_test[cat_features].fillna("missing").astype(str)

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.03,
        depth=3,
        eval_metric="AUC",
        cat_features=cat_features,
        random_seed=42,
        verbose=0,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    return (
        roc_auc_score(y_test, y_pred_proba),
        f1_score(y_test, y_pred),
        accuracy_score(y_test, y_pred),
    )


def evaluate_feature_set(X, y, feature_combo, random_state_list):
    X_subset = X[list(feature_combo)]
    aucs, f1s, accs = [], [], []
    for rdm_state in random_state_list:
        auc, f1, acc = train_single_split(X_subset, y, rdm_state)
        aucs.append(auc)
        f1s.append(f1)
        accs.append(acc)
    return np.mean(aucs), np.std(aucs), np.mean(f1s), np.std(f1s), np.mean(accs), np.std(accs)


def search_all_feature_combinations(data_path, all_cols, output_path,
                                     random_state_list,
                                     min_features=2, max_features=6,
                                     num=False):
    data = pd.read_excel(data_path)
    if num:
        data["Neurol_cond"] = data["Neurol_cond"].replace(["BM", "AVC", "Autre"], [1, 2, 3])
        data["Sex"] = data["Sex"].replace(["M", "F"], [1, 2])
    data = data.apply(DataCleaning.lesion_level_to_num, axis=1)

    X = data[all_cols]
    y = data["MCID_classes"]

    results = []
    total = sum(len(list(combinations(all_cols, s))) for s in range(min_features, max_features + 1))
    print(f"Total combinations to test: {total}")

    for size in range(min_features, max_features + 1):
        for i, combo in enumerate(combinations(all_cols, size)):
            if i % 10 == 0:
                print(f"* Testing combination number {i}. *")
            mean_auc, std_auc, mean_f1, std_f1, mean_acc, std_acc = evaluate_feature_set(X, y, combo, random_state_list)
            results.append({
                "features": list(combo),
                "n_features": size,
                "mean_auc": mean_auc,
                "std_auc": std_auc,
                "mean_f1": mean_f1,
                "std_f1": std_f1,
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc
            })
            print(f"[{size} features] {list(combo)} → AUC={mean_auc:.4f} ± {std_auc:.4f} | F1={mean_f1:.4f} ± {std_f1:.4f} | Acc={mean_acc:.4f} ± {std_acc:.4f}")

    results_df = pd.DataFrame(results).sort_values("mean_auc", ascending=False)

    os.makedirs(output_path, exist_ok=True)
    results_df.to_csv(f"{output_path}/feature_combination_search.csv", index=False)
    print("\nTop 10 feature sets:")
    print(results_df.head(10).to_string(index=False))
    return results_df


if __name__ == "__main__":
    data_path = "/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/CHUNantes collaboration/donnees/data_from_dpi/merged_data_final.xlsx"

    all_cols = [
        "nb_sessions", "duration", "Durée_min", "Vitesse_kmh_MOY",
        "BWS_%_MOY", "cadence", "step_length", "Guidage_%_MOY",
        "sessions_per_week", "Neurol_cond", "Sex", "Age",
        "Nb sessions", "functional_level", "Lesion_num", "BMI",
    ]

    random_state_list = np.arange(1, 21)  # start with 20 splits, not 100 — much faster
    output_path = "results/catboost_results/merged_data/feature_combinations"

    search_all_feature_combinations(
        data_path=data_path,
        all_cols=all_cols,
        output_path=output_path,
        random_state_list=random_state_list,
        min_features=5,
        max_features=len(all_cols),  # increase once you've validated the approach
        num=False,
    )