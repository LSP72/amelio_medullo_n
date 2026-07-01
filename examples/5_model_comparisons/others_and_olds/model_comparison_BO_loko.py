import time
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from amelio_medullo import Calculus

from bayes_opt import BayesianOptimization

# --- Data Loading and Preprocessing ---
print(time.time())

data = pd.read_excel("/Volumes/SP UFD U2/PhD/Stage Nantes/LOKOMAT/loko_final_table_sessions_separated.xlsx")
data = data[
    [
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
]
X = data.drop(columns=["6MWT_m_post", "MCID_classes"])
y = data["MCID_classes"]
features_names = X.columns.to_list()


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
    # save_path = f"/Users/mathildetardif/Documents/Python/Biomarkers/amelio_medullo_n/results/model_tests/shaps/{model_name}_{strat}_shap_plot.png"
    # plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


# --- Configuration ---
stratify_input = input("Use of Stratification? (True/False): ")
loo_input = input("Use Leave-One-Out Cross-Validation for tuning? (True/False): ")

tune_cv = LeaveOneOut() if loo_input == "True" else StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Bayesian Optimization Configuration ---
# Note: Categorical index bounds use .999 (e.g., 0 to 1.999) so int() truncation safely maps to 0 or 1 without out-of-bounds errors.
models_config = {
    "log_reg": {
        "model_class": LogisticRegression,
        "base_params": {"max_iter": 2000, "random_state": 42},
        "pbounds": {"C": (0.001, 100)},
        "formatter": lambda p: {"C": float(p["C"]), "penalty": "l2"},
    },
    "knn": {
        "model_class": KNeighborsClassifier,
        "base_params": {},
        "pbounds": {"n_neighbors": (3, 15), "weights": (0, 1.999)},
        "formatter": lambda p: {
            "n_neighbors": int(p["n_neighbors"]),
            "weights": ["uniform", "distance"][int(p["weights"])],
        },
    },
    "dt": {
        "model_class": DecisionTreeClassifier,
        "base_params": {"random_state": 42},
        "pbounds": {"max_depth": (2, 20), "min_samples_split": (2, 10)},
        "formatter": lambda p: {"max_depth": int(p["max_depth"]), "min_samples_split": int(p["min_samples_split"])},
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
    "hgb": {
        "model_class": HistGradientBoostingClassifier,
        "base_params": {"random_state": 42},
        "pbounds": {"learning_rate": (0.01, 0.3), "max_iter": (50, 200), "max_depth": (3, 15)},
        "formatter": lambda p: {
            "learning_rate": float(p["learning_rate"]),
            "max_iter": int(p["max_iter"]),
            "max_depth": int(p["max_depth"]),
        },
    },
    "svc": {
        "model_class": SVC,
        "base_params": {"random_state": 42, "probability": True},
        "pbounds": {"C": (0.01, 100), "gamma": (0.001, 1), "kernel": (0, 1.999)},
        "formatter": lambda p: {
            "C": float(p["C"]),
            "gamma": float(p["gamma"]),
            "kernel": ["linear", "rbf"][int(p["kernel"])],
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

    # The function BO will try to maximize
    def function_to_maximize(**params):
        formatted_params = formatter(params)
        model = model_class(**base_params, **formatted_params)
        scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        return scores.mean()

    # Initialize and run Bayesian Optimization
    optimizer = BayesianOptimization(
        f=function_to_maximize,
        pbounds=config["pbounds"],
        random_state=42,
        verbose=0,  # Set to 1 or 2 if you want to see step-by-step terminal outputs
    )
    optimizer.maximize(init_points=5, n_iter=10)

    # Extract best parameters and format them properly
    best_params_raw = optimizer.max["params"]
    best_params_formatted = formatter(best_params_raw)

    # Re-instantiate the model with the absolute best parameters and train it
    best_model = model_class(**base_params, **best_params_formatted)
    best_model.fit(x_train, y_train)

    return best_model, best_params_formatted


# --- Main Execution ---
print(f"\n\n===== Processing =====\n")

if stratify_input == "True":
    strats = [y.to_list()]  # [data["Neurol_cond"]]
else:
    strats = [None]

for strat in strats:
    strat_name = "strat" if strat is not None else "no_strat"
    print(f"\n--- Running: {strat_name.upper()} ---\n")

    # Split and Scale safely
    x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=strat, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Train, optimize, and evaluate each model
    for model_name, config in models_config.items():
        print(f"Running Bayesian Optimization for {model_name}...")

        # Call the optimization wrapper
        best_model, best_params = optimize_and_train(
            model_name=model_name, config=config, x_train=x_train_scaled, y_train=y_train, cv=tune_cv
        )

        # Evaluate out-of-sample accuracy
        test_score = best_model.score(x_test_scaled, y_test)

        print(f"  Best Params: {best_params}")
        print(f"  Test Accuracy: {test_score:.4f}\n")

        # Generate SHAP map
        # shap_analysis(best_model, x_train_scaled, x_test_scaled, features_names, model_name, strat_name)

print("All models optimized, trained, and SHAP plots generated.")
