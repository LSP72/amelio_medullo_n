import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from amelio_medullo import Calculus

# ── 1. Load and process data ──────────────────────────────────────────────────
data = pd.read_excel(
    "/Volumes/SP UFD U2/PhD/Stage Nantes/data/datasets/final/final_data_matrix_sessions_separated.xlsx"
)

X = data[
    [
        "Neurol_cond",
        "Lesion_num",
        "Nb sessions",
        "Sex",
        "Age",
        "Height",
        "Weight",
        "6MWT_m_pre",
        "10MWT_pas_pre",
        "10MWT_sec_pre",
        "delay_injury",
        "delay_loko",
        "functional_level",
    ]
]

# X = data[["Neurol_cond", "Lesion", "Sex",	"Age",	"Height",	"Weight",	"6MWT_m_pre",	"10MWT_pas_pre",	"10MWT_sec_pre",	"delay_injury",	"delay_loko",
#     "functional_level",	"Artic_hip_flex",	"Artic_hip_ext",	"Artic_hip_add",	"Artic_hip_abd",	"Artic_hip_rot_ext",	"Artic_hip_rot_int",	"Knee_flex",
#     "Knee_ext",	"Ank_flex_90",	"Ank_flex_180",	"Ank_ext",	"H_Flex_ass",	"H_Ext_PP",	"H_abd",	"H_add",	"H_rot_int",	"K_Flex",	"K_Ext",	"A_Dorsiflex_GT",	"A_Plantarflex"]]

y = Calculus.calculate_MCID_2(data, default_threshold=45)
y = y["MCID_classes"]

# Automatically detect categorical column
cat_features = [col for col in X.columns if X[col].dtype == "object"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=72, stratify=y)
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

# ── 3. Validation ────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
# print(f"True labels: {y_test}")
# print(f"Predicted labels: {y_pred}")

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
plt.title("Matrice de confusion")
plt.tight_layout()
plt.show()

# ── 4. SHAP values ───────────────────────────────────────────────────────────
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)


shap.summary_plot(shap_values, X_test, plot_type="bar", title="Importance globale (SHAP)")
# shap.summary_plot(
#         shap_values,
#         X_test,
#         plot_size=(8, 10),
#         show=True,
#     )
