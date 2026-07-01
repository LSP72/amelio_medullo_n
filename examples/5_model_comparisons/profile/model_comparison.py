# =====================================================================
# Comparaison de modèles ML — version corrigée
# ---------------------------------------------------------------------
# Structure en blocs "# %%" : dans VS Code ou Jupyter, chaque bloc se
# lance indépendamment (Shift+Enter). Exécute-les dans l'ordre.
#
# Correctif principal : tout le prétraitement (imputation, scaling,
# one-hot) est DANS un Pipeline passé à cross_validate. Il est donc
# ré-ajusté sur le train de chaque pli -> plus de fuite de données.
# =====================================================================


# =====================================================================
# BLOC 0 — Imports et configuration
# =====================================================================
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (active IterativeImputer)
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import (
    StratifiedKFold,
    LeaveOneOut,
    cross_validate,
    train_test_split,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC

import shap
from amelio_medullo import Calculus

# ---- Config (remplace les input()) ----------------------------------
DATA_PATH = "/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/CHUNantes collaboration/donnees/data_from_dpi/final_data_matrix_sessions_separated.xlsx"
MCID_THRESHOLD = 45
RANDOM_STATE = 42

USE_LOO = False  # True -> LeaveOneOut ; False -> StratifiedKFold
N_SPLITS = 10  # utilisé seulement si USE_LOO = False (sera plafonné selon la classe la plus rare)

# Colonnes à garder : identifiants + variables post-traitement (fuite vers la cible)
COLS_TO_KEEP = [
    "Neurol_cond",
    "Lesion_num",
    "Nb sessions",
    "Sex",
    "Age",
    "BMI",
    "6MWT_m_pre",
    "10MWT_pas_pre",
    "10MWT_sec_pre",
    "delay_injury",
    "delay_loko",
    "functional_level",
]

# Colonnes catégorielles NOMINALES -> one-hot (surtout pas 1/2/3)
CATEGORICAL_COLS = ["Trouble neuro"]


# =====================================================================
# BLOC 1 — Chargement des données et construction de la cible
# =====================================================================
# %%
data = pd.read_excel(DATA_PATH)

# La cible est calculée AVANT le drop (6MWT_m_post sert au calcul du MCID)
y = pd.Series(Calculus.calculate_MCID_2(data, default_threshold=MCID_THRESHOLD)["MCID_classes"]).reset_index(drop=True)

X = data[COLS_TO_KEEP].copy()

# On garde "Trouble neuro" en texte : le one-hot se fera dans le pipeline.
feature_names = X.columns.tolist()

# ---- Vérifications rapides (à lire, ce sont elles qui te diront si tout est sain) ----
assert len(X) == len(y), "X et y n'ont pas la même longueur — vérifie l'alignement des index."
print(f"n = {len(X)} patients, {X.shape[1]} variables")
print("\nType des colonnes (repère d'éventuelles colonnes texte oubliées) :")
print(X.dtypes)
print("\nRépartition de la cible :")
print(y.value_counts(dropna=False))
print(f"Classe majoritaire = {y.value_counts(normalize=True).max():.1%}  " f"(un modèle naïf ferait déjà ce score)")


# =====================================================================
# BLOC 2 — Prétraitement (DÉFINI, pas encore entraîné)
# =====================================================================
# %%
# Le ColumnTransformer sera ré-ajusté sur chaque pli d'entraînement
# par cross_validate -> aucune information du test ne fuit.
numeric_cols = [c for c in X.columns if c not in CATEGORICAL_COLS]

numeric_pipe = Pipeline(
    [
        # NB : avec un très petit n, SimpleImputer(strategy="median") est plus
        # robuste qu'IterativeImputer. À tester.
        (
            "imputer",
            IterativeImputer(
                n_nearest_features=5,
                imputation_order="ascending",
                random_state=RANDOM_STATE,
            ),
        ),
        ("scaler", StandardScaler()),  # inutile pour les arbres, mais sans effet néfaste
    ]
)

categorical_pipe = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocess = ColumnTransformer(
    [
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, CATEGORICAL_COLS),
    ]
)


# =====================================================================
# BLOC 3 — Modèles à comparer
# =====================================================================
# %%
models = {
    "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    "SVC": SVC(random_state=RANDOM_STATE, probability=True),  # probability pour AUC/SHAP
    "LinearSVC": LinearSVC(random_state=RANDOM_STATE),  # AUC via decision_function
}


# =====================================================================
# BLOC 4 — Validation croisée (métriques multiples, stratifiée par y)
# =====================================================================
# %%
is_binary = y.nunique() == 2

if USE_LOO:
    cv = LeaveOneOut()
    # En LOO, AUC/F1 par pli sont indéfinis et l'écart-type n'a pas de sens.
    scoring = ["accuracy"]
else:
    # On plafonne n_splits par la taille de la classe la plus rare.
    min_class = int(y.value_counts().min())
    n_splits = min(N_SPLITS, min_class)
    if n_splits < 2:
        raise ValueError(f"Classe trop rare ({min_class} cas) pour une validation croisée.")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "balanced_accuracy"]
    if is_binary:
        scoring += ["roc_auc", "f1"]
    print(f"StratifiedKFold avec n_splits = {n_splits}")

rows = []
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    res = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1, error_score="raise")
    row = {"model": name}
    for m in scoring:
        s = res[f"test_{m}"]
        if USE_LOO:
            row[m] = s.mean()
        else:
            row[f"{m}_mean"] = s.mean()
            row[f"{m}_std"] = s.std()
    rows.append(row)

results = pd.DataFrame(rows).set_index("model")
sort_key = "accuracy" if USE_LOO else "balanced_accuracy_mean"
results = results.sort_values(sort_key, ascending=False)
print(results.round(3))


# =====================================================================
# BLOC 5 — SHAP pour un modèle choisi
# =====================================================================
# %%
def run_shap(name, test_size=0.25, n_background=50):
    """Explique un modèle via SHAP en évitant la fuite : le préprocesseur
    n'est ajusté que sur le train. On explique la PROBABILITÉ (pas le label)."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)

    prep = preprocess.fit(X_tr, y_tr)  # ajusté sur le train uniquement
    X_tr_t = prep.transform(X_tr)
    X_te_t = prep.transform(X_te)
    feat = prep.get_feature_names_out()  # noms après one-hot

    model = models[name]
    model.fit(X_tr_t, y_tr)

    tree_models = {"DecisionTree", "RandomForest", "HistGradientBoosting"}
    if name in tree_models:
        # Exact et bien plus rapide pour les arbres.
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_te_t)
        if isinstance(shap_values, list):  # binaire -> [classe0, classe1]
            shap_values = shap_values[1]
    else:
        # Probabilité de la classe positive si dispo, sinon score de décision.
        if hasattr(model, "predict_proba"):
            f = lambda d: model.predict_proba(d)[:, 1]
        else:
            f = model.decision_function
        background = shap.sample(X_tr_t, min(n_background, len(X_tr_t)), random_state=RANDOM_STATE)
        explainer = shap.KernelExplainer(f, background)
        shap_values = explainer.shap_values(X_te_t)

    plt.figure(figsize=(8, 10))
    shap.summary_plot(shap_values, X_te_t, feature_names=feat, max_display=len(feat), show=True)


# Exemple :
# run_shap("RandomForest")
