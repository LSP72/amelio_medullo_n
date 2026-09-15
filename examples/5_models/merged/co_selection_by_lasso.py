import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from itertools import combinations
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from collections import Counter
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


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


def co_selection_analysis(X, y, n_iterations=100, threshold=0.5):
    """
    Analyse la co-sélection des features sur n_iterations splits.

    Returns:
        co_matrix : fréquence de co-sélection (en %) pour chaque paire
        selection_per_split : liste des sets de features sélectionnées par split
        freq_df : fréquence de sélection individuelle
    """
    co_counts = Counter()
    selection_per_split = []
    selection_counts = Counter()

    for rs in range(1, n_iterations + 1):
        selected, _, _ = lasso_pipeline_one_split(X, y, random_state=rs)
        selection_per_split.append(set(selected))
        selection_counts.update(selected)
        for pair in combinations(sorted(selected), 2):
            co_counts[pair] += 1

    # Fréquence individuelle
    freq_df = pd.DataFrame.from_dict(selection_counts, orient="index", columns=["count"])
    freq_df["frequency_%"] = (freq_df["count"] / n_iterations * 100).round(1)
    freq_df = freq_df.sort_values("frequency_%", ascending=False)

    # Matrice de co-sélection (en %)
    features = X.columns.tolist()
    co_matrix = pd.DataFrame(0.0, index=features, columns=features)
    for (f1, f2), count in co_counts.items():
        co_matrix.loc[f1, f2] = round(count / n_iterations * 100, 1)
        co_matrix.loc[f2, f1] = round(count / n_iterations * 100, 1)

    # Garder seulement les features qui apparaissent au moins une fois
    active_features = freq_df.index.tolist()
    co_matrix = co_matrix.loc[active_features, active_features]

    return co_matrix, selection_per_split, freq_df


def plot_co_selection(co_matrix, freq_df, threshold=50):
    """
    Visualise la matrice de co-sélection + fréquences individuelles.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1 : Heatmap de co-sélection ---
    mask = np.eye(len(co_matrix), dtype=bool)  # masque diagonale
    sns.heatmap(
        co_matrix,
        ax=axes[0],
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        mask=mask,
        vmin=0,
        vmax=100,
        linewidths=0.5,
        cbar_kws={"label": "Co-sélection (%)"},
    )
    axes[0].set_title("Fréquence de co-sélection (%)\nsur 100 splits", fontsize=13)
    axes[0].tick_params(axis="x", rotation=90)
    axes[0].tick_params(axis="y", rotation=0)

    # --- Plot 2 : Barplot fréquences individuelles ---
    colors = ["#2ecc71" if f >= threshold else "#e74c3c" for f in freq_df["frequency_%"]]
    axes[1].barh(freq_df.index, freq_df["frequency_%"], color=colors)
    axes[1].axvline(x=threshold, color="black", linestyle="--", linewidth=1.5, label=f"Seuil {threshold}%")
    axes[1].set_xlabel("Fréquence de sélection (%)")
    axes[1].set_title("Stabilité individuelle\nde chaque feature", fontsize=13)
    axes[1].legend()
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig("co_selection_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()


def cadence_bws_conditional_analysis(selection_per_split, feature_a="cadence", feature_b="BWS_%_MOY"):
    """
    Analyse conditionnelle : dans quels splits cadence apparaît-elle ?
    Est-ce corrélé avec la présence/absence de BWS ?
    """
    cadence_with_bws = 0
    cadence_without_bws = 0
    neither = 0
    bws_without_cadence = 0

    for selected in selection_per_split:
        has_cadence = feature_a in selected
        has_bws = feature_b in selected

        if has_cadence and has_bws:
            cadence_with_bws += 1
        elif has_cadence and not has_bws:
            cadence_without_bws += 1
        elif has_bws and not has_cadence:
            bws_without_cadence += 1
        else:
            neither += 1

    total = len(selection_per_split)
    print(f"\n=== Analyse conditionnelle : {feature_a} vs {feature_b} ===")
    print(f"Cadence ET BWS sélectionnés      : {cadence_with_bws:3d}/100 ({cadence_with_bws}%)")
    print(f"Cadence SANS BWS                 : {cadence_without_bws:3d}/100 ({cadence_without_bws}%)")
    print(f"BWS SANS cadence                 : {bws_without_cadence:3d}/100 ({bws_without_cadence}%)")
    print(f"Ni l'un ni l'autre               : {neither:3d}/100 ({neither}%)")

    # Interprétation automatique
    if cadence_without_bws > cadence_with_bws:
        print("\n→ Cadence tend à remplacer BWS (redondance partielle)")
    elif cadence_with_bws > cadence_without_bws * 2:
        print("\n→ Cadence et BWS sont souvent co-sélectionnés (effets complémentaires)")
    else:
        print("\n→ Pas de pattern clair de substitution ou complémentarité")


# --- Main ---
def main(data_path, cols_to_keep):
    data = pd.read_excel(data_path)
    data["Neurol_cond"] = data["Neurol_cond"].replace(["BM", "AVC", "Autre"], [1, 2, 3])
    data["Sex"] = data["Sex"].replace(["M", "F"], [1, 2])

    X = data[cols_to_keep]
    y = data["MCID_classes"]

    print("=== Analyse de co-sélection sur 100 splits ===\n")
    co_matrix, selection_per_split, freq_df = co_selection_analysis(X, y, n_iterations=100)

    print("Fréquences individuelles :")
    print(freq_df.to_string())

    plot_co_selection(co_matrix, freq_df, threshold=70)

    # Analyse spécifique cadence vs BWS
    cadence_bws_conditional_analysis(selection_per_split, feature_a="cadence", feature_b="BWS_%_MOY")


if __name__ == "__main__":
    data_path = "/Volumes/SP UFD U2/PhD/Stage Nantes/data/datasets/final/merged_data_final.xlsx"
    ## SELECTED FEATURES for loko
    # cols_to_keep = [
    #     "nb_sessions",
    #     "duration",
    #     "Durée_min",
    #     "Vitesse_kmh_MOY",
    #     "BWS_%_MOY",
    #     "step_length",
    #     "Guidage_%_MOY",
    #     "sessions_per_week",
    #     "6MWT_m_pre",
    #     "functional_level",
    #     "cadence",
    # ]
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

    main(data_path, cols_to_keep)
