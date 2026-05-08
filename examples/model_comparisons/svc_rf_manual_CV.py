import time
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization


# --- SHAP Analysis Function ---
def shap_analysis(model, x_train, x_test, features_names):
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

    return shap_values


# --- Bayesian Optimization Configuration ---

models_config = {
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


def train_and_validate(X, y, rdm_state, features_names):
    # --- Main Execution ---
    print(f"\n\n===== Processing {rdm_state} =====\n")

    # Split and Scale safely
    x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=rdm_state)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    results_dict_i = {
        "index_train": x_train.index,
        "index_test": y_test.index,
        "true_values": y_test,
    }

    # Train, optimize, and evaluate each model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for model_name, config in models_config.items():
        print(f"Running Bayesian Optimization for {model_name}...")

        # Call the optimization wrapper
        best_model, best_params = optimize_and_train(
            model_name=model_name, config=config, x_train=x_train_scaled, y_train=y_train, cv=cv
        )

        y_pred = best_model.predict(x_test_scaled)
        # Evaluate out-of-sample accuracy
        test_score = best_model.score(x_test_scaled, y_test)
        # print(f"  Best Params: {best_params}")
        # print(f"  Test Accuracy: {test_score:.4f}\n")

        # Generate SHAP map
        shap_values = shap_analysis(best_model, x_train_scaled, x_test_scaled, features_names)

        results_dict_i[model_name] = {"accuracy": test_score, "predictions": y_pred, "shap_values": shap_values}

    return results_dict_i


def save_dict(results_dict, output_path, separated_sessions=True):
    pickle_file_name = output_path + "/loko_results_all_features_100it_with_shap_with_cadence.pkl"
    with open(pickle_file_name, "wb") as file:
        pkl.dump(results_dict, file)


def main(data_path, cols_to_keep, rdm_state_list, output_path):
    # --- Data Loading and Preprocessing ---
    print(time.time())
    data = pd.read_excel(data_path)
    data = data[cols_to_keep]
    X = data.drop(columns=["6MWT_m_post", "MCID_classes"])
    y = data["MCID_classes"]
    features_names = X.columns.to_list()
    results_dict = {}
    for rdm_state in rdm_state_list:
        results_dict[rdm_state] = train_and_validate(X, y, rdm_state, features_names)

    save_dict(results_dict, output_path)


if __name__ == "__main__":
    data_path = 
    ## SELECTED FEATURES
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
        "cadence",
    ]
    ## ALL FEATURES
    # cols_to_keep = ["nb_sessions",	"duration",	"Distance_m",	"Distance_pas",	"Durée_min",	"Vitesse_kmh_MIN",	"Vitesse_kmh_MAX",	"Vitesse_kmh_MOY",	"BWS_%_MIN",	"BWS_%_MAX",
    #                      "BWS_%_MOY",	"BWS_kg_MIN",	"BWS_kg_MAX",	"BWS_kg_MOY",	"Guidage_G_%_MIN",	"Guidage_G_%_MAX",	"Guidage_G_%_MOY",	"Guidage_D_%_MIN",	"Guidage_D_%_MAX",
    #                      "Guidage_D_%_MOY",	"sessions_per_week",	"6MWT_m_pre", "6MWT_m_post", "MCID_classes",	"functional_level", "cadence"]
    # random_state_list = [42]
    random_state_list = np.arange(1, 101)
    output_path = "results/loko_results"
    main(data_path, cols_to_keep, random_state_list, output_path)
