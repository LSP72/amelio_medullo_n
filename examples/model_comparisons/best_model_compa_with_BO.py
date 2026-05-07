import time
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score, accuracy_score
from amelio_medullo import Calculus

from bayes_opt import BayesianOptimization

# --- Data Loading and Preprocessing ---
print(time.time())

data = pd.read_excel("/Users/mathildetardif/Documents/Documents/PhD/Nantes/data_drafts/prelim_table_2.xlsx")
MCID = Calculus.calculate_MCID(data["6MWT_m_pre"], data["6MWT_m_post"], threshold=45)
data.drop(
    columns=[
        "IPP",
        "6MWT_m_post",
        "Height",
        "Weight",
        "Nb sessions",
        "ASIA_mot_D_pre",
        "ASIA_mot_G_pre",
        "ASIA_mot_pre",
        "ASIA_LL_D",
        "ASIA_LL_G",
        "ASIA_LL",
        "10MWT_pas_pre",
    ],
    inplace=True,
)
X, y = data, MCID
data["Trouble neuro"] = data["Trouble neuro"].replace(["AVC", "BM", "Autre"], [1, 2, 3])
feature_names_file = pd.read_excel(
    "/Users/mathildetardif/Documents/Documents/PhD/Nantes/datasets/final/feature_names.xlsx"
)
names_dict = dict(zip(feature_names_file["features"], feature_names_file["features_names"]))
features_names = [names_dict.get(name, name) for name in X.columns.to_list()]

# Imputation
imp_mean = IterativeImputer(missing_values=np.nan, n_nearest_features=5, random_state=42, imputation_order="ascending")
X = imp_mean.fit_transform(X)

# --- Define CV and Strats (Added back for completeness) ---
strats = [data["Trouble neuro"], None]  # Testing both stratified and non-stratified
tune_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Adjust to LeaveOneOut() if preferred


# --- SHAP Analysis Function ---
def shap_analysis(model, x_train, x_test, features_names, model_name, strat):
    explainer = shap.KernelExplainer(model.predict, x_train)
    shap_values = explainer.shap_values(x_test, silent=True)

    plt.figure(figsize=(8, 10))
    shap.summary_plot(
        shap_values,
        x_test,
        feature_names=features_names,
        max_display=len(features_names),
        plot_size=(8, 10),
        show=False,
    )
    save_path = f"/Users/mathildetardif/Documents/Python/Biomarkers/amelio_medullo_n/results/model_tests/shaps/3_models_comp_BO+AUC/{model_name}_{strat}_shap_plot_2.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


# --- Bayesian Optimization Configuration ---
models_config = {
    "log_reg": {
        "model_class": LogisticRegression,
        "base_params": {"max_iter": 2000, "random_state": 42},
        "pbounds": {"C": (0.001, 100)},
        "formatter": lambda p: {"C": float(p["C"]), "penalty": "l2"},
    },
    "rf": {
        "model_class": RandomForestClassifier,
        "base_params": {"random_state": 42},
        "pbounds": {"n_estimators": (50, 200), "max_depth": (2, 20), "min_samples_split": (2, 10)},
        "formatter": lambda p: {
            "n_estimators": int(p["n_estimators"]),
            "max_depth": int(p["max_depth"]),
            "min_samples_split": int(p["min_samples_split"]),
        },
    },
    "linear_svc": {
        "model_class": LinearSVC,
        "base_params": {"random_state": 42, "dual": False},
        "pbounds": {"C": (0.001, 100)},
        "formatter": lambda p: {"C": float(p["C"])},
    },
}


# --- Generic Bayesian Optimizer Wrapper ---
def optimize_and_train(model_name, config, x_train, y_train, cv):
    model_class = config["model_class"]
    base_params = config["base_params"]
    formatter = config["formatter"]

    def function_to_maximize(**params):
        formatted_params = formatter(params)
        model = model_class(**base_params, **formatted_params)
        # CHANGED: Scoring is now roc_auc instead of accuracy
        scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    optimizer = BayesianOptimization(f=function_to_maximize, pbounds=config["pbounds"], random_state=42, verbose=0)
    optimizer.maximize(init_points=5, n_iter=10)

    best_params_raw = optimizer.max["params"]
    best_params_formatted = formatter(best_params_raw)

    best_model = model_class(**base_params, **best_params_formatted)
    best_model.fit(x_train, y_train)

    return best_model, best_params_formatted


# --- Main Execution ---
print(f"\n\n===== Processing =====\n")

for strat in strats:
    strat_name = "strat" if strat is not None else "no_strat"
    print(f"\n--- Running: {strat_name.upper()} ---\n")

    x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=strat, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    for model_name, config in models_config.items():
        print(f"Running Bayesian Optimization for {model_name}...")

        best_model, best_params = optimize_and_train(
            model_name=model_name, config=config, x_train=x_train_scaled, y_train=y_train, cv=tune_cv
        )

        # CHANGED: Evaluate out-of-sample Accuracy AND ROC AUC
        test_acc = accuracy_score(y_test, best_model.predict(x_test_scaled))

        # For ROC AUC, we need probabilities. LinearSVC doesn't have predict_proba,
        # so we use decision_function instead.
        if hasattr(best_model, "predict_proba"):
            y_pred_scores = best_model.predict_proba(x_test_scaled)[:, 1]
        else:
            y_pred_scores = best_model.decision_function(x_test_scaled)

        test_auc = roc_auc_score(y_test, y_pred_scores)

        print(f"  Best Params: {best_params}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Test ROC AUC:  {test_auc:.4f}\n")

        # Generate SHAP map
        shap_analysis(best_model, x_train_scaled, x_test_scaled, features_names, model_name, strat_name)

print("All models optimized, trained, and SHAP plots generated.")
