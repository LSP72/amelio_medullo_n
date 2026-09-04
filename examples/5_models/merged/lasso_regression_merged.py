import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from collections import Counter


def lasso_pipeline_one_split(X, y, random_state=42, cv=5, test_size=0.2):
    """
    Un seul split train/test avec pipeline complet :
    KNNImputer → StandardScaler → LogisticRegressionCV (L1)

    Returns:
        selected_features (list)
        coefs (pd.Series)
        test_accuracy (float)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y  # important avec dataset petit + binaire
    )

    pipeline = Pipeline(
        [
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler()),
            (
                "lasso",
                LogisticRegressionCV(
                    penalty="l1",
                    solver="liblinear",
                    Cs=np.logspace(-3, 1, 50),  # cherche le meilleur C
                    cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
                    max_iter=10000,
                    scoring="roc_auc",
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    coefs = pd.Series(pipeline.named_steps["lasso"].coef_[0], index=X.columns)
    selected_features = coefs[coefs != 0].sort_values(key=abs, ascending=False).index.tolist()
    test_acc = pipeline.score(X_test, y_test)

    return selected_features, coefs, test_acc


def lasso_stability_over_splits(X, y, n_iterations=100, cv=5, test_size=0.2):
    """
    Répète le pipeline sur n_iterations splits différents.
    Retourne la fréquence de sélection de chaque feature.
    """
    selection_counts = Counter()
    all_coefs = []
    all_accuracies = []

    for rs in range(1, n_iterations + 1):
        selected, coefs, acc = lasso_pipeline_one_split(X, y, random_state=rs, cv=cv, test_size=test_size)
        selection_counts.update(selected)
        all_coefs.append(coefs)
        all_accuracies.append(acc)

    # Fréquence de sélection (sur 100 splits)
    freq_df = pd.DataFrame.from_dict(selection_counts, orient="index", columns=["count"])
    freq_df["frequency_%"] = (freq_df["count"] / n_iterations * 100).round(1)
    freq_df = freq_df.sort_values("frequency_%", ascending=False)

    # Moyenne et SD des coefficients sur les 100 splits
    coefs_df = pd.DataFrame(all_coefs)
    stability_df = pd.DataFrame(
        {"mean_coef": coefs_df.mean(), "std_coef": coefs_df.std(), "selection_freq_%": freq_df["frequency_%"]}
    ).sort_values("selection_freq_%", ascending=False)

    print(
        f"\nMean test accuracy over {n_iterations} splits: "
        f"{np.mean(all_accuracies):.3f} ± {np.std(all_accuracies):.3f}"
    )
    print("\nFeature selection stability:")
    print(stability_df.to_string())

    return stability_df, all_accuracies


def main(data_path, cols_to_keep, classif=True, n_iterations=100):
    data = pd.read_excel(data_path)
    data["Neurol_cond"] = data["Neurol_cond"].replace(["BM", "AVC", "Autre"], [1, 2, 3])
    data["Sex"] = data["Sex"].replace(["M", "F"], [1, 2])

    X = data[cols_to_keep]
    y = data["MCID_classes"] if classif else data["6MWT_m_post"]

    stability_df, accuracies = lasso_stability_over_splits(X, y, n_iterations=n_iterations)

    return stability_df


if __name__ == "__main__":
    data_path = "/Volumes/SP UFD U2/PhD/Stage Nantes/data/datasets/final/merged_data_final.xlsx"
    ## SELECTED FEATURES for loko
    cols_to_keep = [
        "nb_sessions",
        "duration",
        "Durée_min",
        "Vitesse_kmh_MOY",
        "BWS_%_MOY",
        "step_length",
        "Guidage_%_MOY",
        "sessions_per_week",
        "6MWT_m_pre",
        "functional_level",
        "cadence",
    ]
    ## ALL FEATURES for loko
    # cols_to_keep = ["nb_sessions",	"duration",	"Distance_m",	"Distance_pas",	"Durée_min",	"Vitesse_kmh_MIN",	"Vitesse_kmh_MAX",	"Vitesse_kmh_MOY",	"BWS_%_MIN",	"BWS_%_MAX",
    #                      "BWS_%_MOY",	"BWS_kg_MIN",	"BWS_kg_MAX",	"BWS_kg_MOY",	"Guidage_G_%_MIN",	"Guidage_G_%_MAX",	"Guidage_G_%_MOY",	"Guidage_D_%_MIN",	"Guidage_D_%_MAX",
    #                      "Guidage_D_%_MOY",	"sessions_per_week",	"6MWT_m_pre", "functional_level", "cadence"]
    # features for merged data
    cols_to_keep = [
        "nb_sessions",
        "duration",
        "Durée_min",
        "Vitesse_kmh_MOY",
        "BWS_%_MOY",
        "step_length",
        "Guidage_%_MOY",
        "sessions_per_week",
        "Neurol_cond",
        "Sex",
        # "Age", # 1 missing
        "Nb sessions",
        "functional_level",
        # "Lesion_num", # 5 missing
        # "BMI", # 7 missing
        "cadence",
    ]
    random_state_list = [42]
    # random_state_list = np.arange(1, 101)
    output_path = "results/loko_results"

    main(data_path, cols_to_keep, classif=True)
