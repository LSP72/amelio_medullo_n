import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from amelio_medullo import Calculus

# ── 1. Load and process data ──────────────────────────────────────────────────
data = pd.read_excel("/Users/mathildetardif/Documents/final_data_matrix.xlsx")

X = data[
    [
        "Neurol_cond",
        "Lesion",
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

y = Calculus.calculate_MCID_2(data, default_threshold=45)
y = y["MCID_classes"]

# Automatically detect categorical column
cat_features = [col for col in X.columns if X[col].dtype == "object"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Convert NaN into string "missing'
X_train[cat_features] = X_train[cat_features].fillna("NaN").astype(str)
X_test[cat_features] = X_test[cat_features].fillna("NaN").astype(str)

# ── 2. Training the CatBoost model ────────────────────────────────────────────────────
model = GradientBoostingClassifier()

model.fit(
    X_train,
    y_train,
)

# ── 3. Validation ────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
plt.title("Matrice de confusion")
plt.tight_layout()
plt.show()

# # ── 4. SHAP values ───────────────────────────────────────────────────────────
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X_test)


# shap.summary_plot(shap_values, X_test, plot_type="bar", title="Importance globale (SHAP)")
# shap.summary_plot(
#         shap_values,
#         X_test,
#         plot_size=(8, 10),
#         show=True,
#     )
