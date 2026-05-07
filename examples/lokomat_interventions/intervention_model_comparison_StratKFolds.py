import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import partial_dependence

# from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

# Code based on a paper by Kuo et al. (2021)

# ----- Functions to evaluate the models -----


def sensitivity(y_true, y_pred):
    """TP / (TP + FN)"""
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def specificity(y_true, y_pred):
    """TN / (TN + FP)"""
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


# ----- Creating model dict -----
def get_model():
    """
    Returns 5 models within a dict

    """
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": GradientBoostingClassifier(
            n_estimators=100,
            random_state=42,
        ),
    }

    models = {}
    for name, clf in classifiers.items():
        models[name] = Pipeline(
            [
                ("scaler", StandardScaler()),  # normalisation
                ("smote", SMOTE(sampling_strategy="auto", random_state=42)),
                ("clf", clf),
            ]
        )

    return models


# ----- Function to train and evaluate the models -----
def evaluate_models(X, y, models, param_grids, cv=10):
    """
    Optimise les hyperparamètres de chaque modèle par RandomizedSearchCV.
    Retourne un DataFrame de résultats et les meilleurs modèles.
    """
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = {}
    best_models = {}
    scoring = {
        "auc": "roc_auc",
        "accuracy": "accuracy",
        "sensitivity": make_scorer(sensitivity),
        "specificity": make_scorer(specificity),
    }

    print(f"\n Tuning des hyperparamètres (RandomizedSearchCV, {cv}-fold)...\n")

    for name, pipeline in models.items():
        if name not in param_grids:
            print(f"  ⚠️  Pas de grille pour {name} — ignoré.")
            continue

        search = RandomizedSearchCV(
            pipeline,
            param_grids[name],
            n_iter=50,  # 30 combinaisons aléatoires par modèle
            cv=cv_splitter,
            scoring="roc_auc",  # optimise sur l'AUC
            random_state=42,
            n_jobs=-1,
            refit=True,  # réentraîne sur tout X avec les meilleurs params
        )
        search.fit(X, y)

        # # Récupère les scores CV du meilleur modèle
        # best_idx = search.best_index_
        # cv_res   = search.cv_results_

        # Recalcule toutes les métriques sur le meilleur modèle trouvé
        scores = cross_validate(search.best_estimator_, X, y, cv=cv_splitter, scoring=scoring)

        results[name] = {
            "AUC": f"{scores['test_auc'].mean():.3f} +/- {scores['test_auc'].std():.3f}",
            "Accuracy": f"{scores['test_accuracy'].mean():.3f} +/- {scores['test_accuracy'].std():.3f}",
            "Sensitivity": f"{scores['test_sensitivity'].mean():.3f} +/- {scores['test_sensitivity'].std():.3f}",
            "Specificity": f"{scores['test_specificity'].mean():.3f} +/- {scores['test_specificity'].std():.3f}",
            "_auc_mean": scores["test_auc"].mean(),
        }
        best_models[name] = search.best_estimator_

        print(f"  ✅ {name}")
        print(f"     Meilleurs params : {search.best_params_}")
        print(f"     AUC après tuning : {scores['test_auc'].mean():.3f}\n")

    results_df = pd.DataFrame(results).T.sort_values("_auc_mean", ascending=False)
    results_df = results_df.drop(columns="_auc_mean")
    return results_df, best_models


# ----- Functions to plot feature importance from RF -----
def plot_feature_importance(X, y, feature_cols, top_n=20):
    """
    Entraîne un Random Forest sur tout le dataset et affiche
    l'importance des variables (Mean Decrease in Impurity).
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    importances.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Importance (Mean Decrease in Impurity)", fontsize=12)
    ax.set_title("Importance des Variables — Random Forest", fontsize=14, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.8)
    plt.tight_layout()
    # plt.savefig('/mnt/user-data/outputs/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n📊 Graphique d'importance des variables sauvegardé.")
    return rf, importances


# ----- Function to plot results table -----
def plot_results_table(results_df, output_path):
    """Sauvegarde le tableau de comparaison des modèles en PNG."""
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.axis("off")
    tbl = ax.table(
        cellText=results_df.values,
        rowLabels=results_df.index,
        colLabels=results_df.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 2)

    # Mettre en évidence la première ligne (meilleur modèle)
    for j in range(len(results_df.columns)):
        tbl[(1, j)].set_facecolor("#c8e6c9")

    ax.set_title("Comparaison des modèles ML — Validation croisée 10-fold", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(
        f"{output_path}/results_table_selected_feature_SMOTE_SVM_forced_at_lin.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("📊 Tableau des résultats sauvegardé.")


# %% ----- MAIN -----


def main(file_path: str, cols_to_keep: list, output_path: str):

    print("=" * 60)
    print("  PREDICTION OF 6MWT DISTANCE (LOKOMAT)")
    print("  Inspired by Kuo et al. (2021)")
    print("=" * 60)

    # 1. Loading and preparing the data
    data = pd.read_excel(file_path)
    data = data[cols_to_keep]
    X = data.drop(columns=["6MWT_m_post", "MCID_classes"])
    y = data["MCID_classes"]

    # 2. Models to consider
    models = get_model()

    # 3. Get models' parameters to optimise
    param_grids = {
        "Logistic Regression": {
            "clf__C": [0.001, 0.01, 0.1, 1, 10, 100],
            "clf__penalty": ["l1", "l2"],
            "clf__solver": ["liblinear", "saga"],
        },
        "Decision Tree": {
            "clf__max_depth": [2, 3, 4, 5, None],
            "clf__min_samples_leaf": [2, 3, 5, 8, 10],
            "clf__criterion": ["gini", "entropy"],
        },
        "SVM": {
            "clf__C": [0.01, 0.1, 1, 10, 100],
            "clf__kernel": ["linear"],
            "clf__gamma": ["scale", "auto", 0.001, 0.01],
        },
        "Random Forest": {
            "clf__n_estimators": [50, 100, 200, 300],
            "clf__max_depth": [2, 3, 4, 5, None],
            "clf__min_samples_leaf": [2, 3, 5, 8, 10],
            "clf__max_features": ["sqrt", "log2", 0.5],
            "clf__class_weight": ["balanced", "balanced_subsample"],
        },
        "XGBoost": {
            "clf__n_estimators": [50, 100, 200],
            "clf__max_depth": [2, 3, 4, 5],
            "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "clf__subsample": [0.6, 0.8, 1.0],
            "clf__max_features": ["sqrt", "log2", 0.5],  # remplace colsample_bytree
            "clf__min_samples_leaf": [2, 3, 5],  # remplace reg_lambda
        },
    }

    # 4. Train and evaluate
    results_df, best_models = evaluate_models(X, y, models, param_grids)

    print("\n" + "=" * 60)
    print("  RESULTS  ")
    print("=" * 60)
    print(results_df.to_string())

    # 5. Plots
    plot_results_table(results_df, output_path)
    rf_model, importances = plot_feature_importance(X, y, list(X.columns))

    print("\n✅ Pipeline terminé. Fichiers générés dans /mnt/user-data/outputs/")
    print("   - feature_importance.png")
    print("   - partial_dependence_plots.png")
    print("   - results_table.png")

    return results_df, rf_model


if __name__ == "__main__":
    file_path = 
    # cols_to_keep = ["nb_sessions",	"duration",	"Distance_m",	"Distance_pas",	"Durée_min",	"Vitesse_kmh_MIN",	"Vitesse_kmh_MAX",	"Vitesse_kmh_MOY",	"BWS_%_MIN",	"BWS_%_MAX",
    #                      "BWS_%_MOY",	"BWS_kg_MIN",	"BWS_kg_MAX",	"BWS_kg_MOY",	"Guidage_G_%_MIN",	"Guidage_G_%_MAX",	"Guidage_G_%_MOY",	"Guidage_D_%_MIN",	"Guidage_D_%_MAX",
    #                      "Guidage_D_%_MOY",	"sessions_per_week",	"6MWT_m_pre", "6MWT_m_post", "MCID_classes",	"functional_level"]
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
        "6MWT_m_post",
        "MCID_classes",
        "functional_level",
    ]
    output_path = "results/loko_results/Kuo_approach"
    main(file_path, cols_to_keep, output_path)
