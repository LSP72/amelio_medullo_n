# =====================================================================
# Régression logistique régularisée — analyse pré-spécifiée et bien validée
# ---------------------------------------------------------------------

# =====================================================================
# BLOC 0 — Imports et configuration
# =====================================================================
# %%
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import (
    StratifiedKFold,
    RepeatedStratifiedKFold,
    GridSearchCV,
    cross_validate,
)

from amelio_medullo import Calculus

# ---- DATA SET TO ESTABLISH ---------------------------------------------------------
DATA_SET = "merged"

# ---- Config ---------------------------------------------------------
if DATA_SET == 'profile':
    DATA_PATH = "/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/CHUNantes collaboration/donnees/data_from_dpi/final_data_matrix_sessions_separated.xlsx"
    COLS_TO_KEEP = [
        "Lesion_num",
        "Nb sessions",
        "Sex",
        "BMI",
        "6MWT_m_pre",
        "delay_loko",
        "functional_level",
        "speed",
    ]
    CATEGORICAL_COLS = ["Sex"]
    
elif DATA_SET == 'merged':
    DATA_PATH = "/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/CHUNantes collaboration/donnees/data_from_dpi/merged_data_final.xlsx"
    COLS_TO_KEEP = [
        "nb_sessions",
        # "duration",
        "Durée_min",
        "Vitesse_kmh_MOY",
        "BWS_%_MOY",
        "step_length",
        "Guidage_%_MOY",
        "sessions_per_week",
        # "Neurol_cond",
        # "Sex",
        # "Age",
        # "Nb sessions",
        # "functional_level",
        # "Lesion_num",
        # "BMI",
    ]
    CATEGORICAL_COLS = [] # ["Sex", "Neurol_cond"]

MCID_THRESHOLD = 45
RANDOM_STATE = 42

# Régularisation
PENALTY = "l2"                 # "l2" = coefficients stables ; "l1"/"elasticnet" = sélection
                               #   mais instable sur petit n avec variables corrélées.
C_GRID = np.logspace(-3, 2, 12)
CLASS_WEIGHT = "balanced"      # "balanced" recommandé si classes déséquilibrées ; sinon None

# Validation
K = 5                          # nb de plis (interne ET externe) ; plafonné selon la classe rare
N_REPEATS = 1                  # mets 10-20 pour stabiliser l'estimation sur petit n (CV répétée)

data = pd.read_excel(DATA_PATH)
y = pd.Series(Calculus.calculate_MCID_2(data, default_threshold=MCID_THRESHOLD)["MCID_classes"]).reset_index(drop=True)
X = data[COLS_TO_KEEP].copy()
X = X.replace([np.inf, -np.inf], 0) 

is_binary = (y.nunique() == 2)
n_events = int(y.value_counts().min())
print(f"n = {len(X)} | événements classe rare = {n_events} | "
      f"classe majoritaire = {y.value_counts(normalize=True).max():.1%}")


# =====================================================================
# BLOC 1 — Prétraitement + modèle (dans un seul pipeline)
# =====================================================================
# %%
numeric_cols = [c for c in X.columns if c not in CATEGORICAL_COLS]

preprocess = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),   # robuste sur petit n
                      ("sc", StandardScaler())]), numeric_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_COLS),
])

# Plis : l'interne (réglage de C) tourne sur un train plus petit -> plus petit que l'externe.
# Si ça plante encore sur "n_splits ... members in each class", passe inner_cv à 2.
outer_k = max(2, min(5, n_events))
inner_cv = max(2, min(3, n_events - 1))

clf = LogisticRegressionCV(
    Cs=10, cv=inner_cv,                       # choisit C par CV interne, tout seul
    penalty="l2", solver="liblinear",
    scoring="roc_auc" if is_binary else "balanced_accuracy",
    class_weight="balanced",                  # mets None si classes équilibrées
    max_iter=5000, random_state=RANDOM_STATE,
)
pipe = Pipeline([("prep", preprocess), ("clf", clf)])


# =====================================================================
# BLOC 2 — Performance honnête (nichée) + odds ratios
# =====================================================================
# %%
outer_cv = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=RANDOM_STATE)
scoring = ["balanced_accuracy", "accuracy"] + (["roc_auc"] if is_binary else [])

res = cross_validate(pipe, X, y, cv=outer_cv, scoring=scoring, n_jobs=-1, error_score="raise")
print("=== Performance (validation nichée) ===")
for m in scoring:
    s = res[f"test_{m}"]
    print(f"{m:18s}: {s.mean():.3f} ± {s.std():.3f}  [{s.min():.3f}, {s.max():.3f}]")

# Modèle final sur toutes les données -> uniquement pour les coefficients à interpréter.
pipe.fit(X, y)
feat = pipe.named_steps["prep"].get_feature_names_out()
coefs = pipe.named_steps["clf"].coef_.ravel()
print("\nC retenu :", float(pipe.named_steps["clf"].C_[0]))
print((pd.DataFrame({"feature": feat, "coef": coefs, "odds_ratio": np.exp(coefs)})
       .sort_values("coef", key=lambda s: s.abs(), ascending=False)
       .to_string(index=False, float_format=lambda v: f"{v:.3f}")))
# Numériques : standardisées -> OR par +1 écart-type. One-hot : OR de la catégorie.

# =====================================================================
# BLOC 3 — Rapport Excel
# =====================================================================
# %%
from datetime import datetime

# 1) Performance : une ligne par métrique
perf = pd.DataFrame({m: {"mean": res[f"test_{m}"].mean(), "std": res[f"test_{m}"].std(),
                         "min": res[f"test_{m}"].min(), "max": res[f"test_{m}"].max()}
                     for m in scoring}).T
perf.index.name = "metric"
perf = perf.reset_index()

# 2) Odds ratios
or_table = (pd.DataFrame({"feature": feat, "coef": coefs, "odds_ratio": np.exp(coefs)})
            .sort_values("coef", key=lambda s: s.abs(), ascending=False).reset_index(drop=True))

# 3) Infos + garde-fous d'interprétation (ils voyagent avec le fichier)
meta = pd.DataFrame({"champ": [
    "date", "n_patients", "n_evenements_classe_rare", "taux_classe_majoritaire",
    "C_retenu", "penalty", "class_weight", "plis_externes", "plis_internes",
    "NOTE_1", "NOTE_2", "NOTE_3"],
    "valeur": [
    datetime.now().strftime("%Y-%m-%d %H:%M"), len(X), n_events,
    f"{y.value_counts(normalize=True).max():.1%}",
    round(float(pipe.named_steps['clf'].C_[0]), 4), "l2", "balanced", outer_k, inner_cv,
    "Lire la performance AVANT les OR ; comparer accuracy au taux_classe_majoritaire, balanced_acc/AUC a 0.5",
    "OR des variables numeriques = effet par +1 ecart-type (donnees standardisees), pas par unite brute",
    "Aucun intervalle de confiance ici + OR biaises vers 1 par la regularisation : pas de 'significativite'"]})

stamp = datetime.now().strftime("%Y%m%d_%H%M")          # horodatage -> pas d'ecrasement entre runs
out_path = f"results/rapport_logreg_{DATA_SET}_{stamp}.xlsx"
with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
    meta.to_excel(xl, sheet_name="infos", index=False)
    perf.to_excel(xl, sheet_name="performance", index=False)
    or_table.to_excel(xl, sheet_name="odds_ratios", index=False)
print(f"Report saved in: {out_path}")