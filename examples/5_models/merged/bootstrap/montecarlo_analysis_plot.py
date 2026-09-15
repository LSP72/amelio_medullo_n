import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt


def analyze_shap_stability_across_splits(pickle_path, data):
    with open(pickle_path, "rb") as f:
        results = pkl.load(f)

    # Collecter mean(|SHAP|) par feature pour chaque split
    feature_names = results[list(results.keys())[0]]["list_of_features"]
    shap_importance_per_split = []

    for rdm_state, res in results.items():
        # shap_values shape: (n_test, n_features)
        mean_abs_shap = np.abs(res["shap_values"]).mean(axis=0)
        shap_importance_per_split.append(mean_abs_shap)

    shap_array = np.array(shap_importance_per_split)  # (100, n_features)

    shap_mean = shap_array.mean(axis=0)
    shap_lower = np.percentile(shap_array, 2.5, axis=0)
    shap_upper = np.percentile(shap_array, 97.5, axis=0)

    # Bonus : rank stability
    ranks = np.argsort(np.argsort(-shap_array, axis=1), axis=1)
    rank_std = ranks.std(axis=0)

    sorted_idx = np.argsort(shap_mean)[::-1]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        range(len(feature_names)),
        shap_mean[sorted_idx],
        xerr=[shap_mean[sorted_idx] - shap_lower[sorted_idx], shap_upper[sorted_idx] - shap_mean[sorted_idx]],
        align="center",
        alpha=0.8,
        color="steelblue",
        ecolor="black",
        capsize=4,
    )
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([f"{feature_names[i]} (rank σ={rank_std[i]:.1f})" for i in sorted_idx])
    ax.set_xlabel("Mean |SHAP value| ± 95% CI (across 100 splits)")
    ax.set_title("Stabilité SHAP à travers 100 splits stratifiés")
    plt.tight_layout()
    plt.savefig(f"results/catboost_results/{data}/monte_carlo/shap_stability_for_{data}.png", dpi=150)
    plt.show()

    return shap_mean, shap_lower, shap_upper, rank_std


if __name__ == "__main__":
    data = input("Which dataset do you want to analyze? (merged_data or profile): ")
    if data == "merged_data":
        pickle_path = "results/catboost_results/merged_data/monte_carlo/catboost_results_merged_data_selected_features_monte_carlo.pkl"
    elif data == "profile":
        pickle_path = "/Users/mathildetardif/Documents/Python/Biomarkers/amelio_medullo_n/results/catboost_results/profile_data/monte_carlo/catboost_results_separated_sessions_is_True_selected_features_monte_carlo.pkl"
    else:
        raise ValueError("Invalid dataset choice. Please choose 'merged_data' or 'profile'.")
    analyze_shap_stability_across_splits(pickle_path, data)
