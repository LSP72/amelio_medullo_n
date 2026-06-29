"""
Plotting / reporting pipeline for CatBoost Monte-Carlo results.

Pipeline stages:
    1. load_data           - load + clean raw Excel data
    2. load_results        - load pickled per-random-state results
    3. aggregate_metrics   - accuracy / AUC / ECE / dispersion across iterations
    4. aggregate_shap      - per-patient mean SHAP + global feature importance
    5. plot_*              - all figure-producing functions, saved as SVG

Run as a script: edit the CONFIG block at the bottom and run.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import shap
from sklearn.metrics import accuracy_score, roc_curve
from amelio_medullo import DataCleaning, ResultsCalculus


# ============================================================
# 1. Data loading
# ============================================================

def load_data(data_path):
    data = pd.read_excel(data_path)
    data["Neurol_cond"] = data["Neurol_cond"].replace(["BM", "AVC", "Autre"], [1, 2, 3])
    data["Sex"] = data["Sex"].replace(["M", "F"], [1, 2])
    data = data.apply(DataCleaning.lesion_level_to_num, axis=1)
    return data


def load_results(dict_path):
    with open(dict_path, "rb") as f:
        results_dict = pkl.load(f)
    first_key = next(iter(results_dict))
    cols_to_keep = results_dict[first_key]["model"].feature_names_
    return results_dict, cols_to_keep


def load_feature_name_map(feature_names_path):
    feature_data = pd.read_excel(feature_names_path)
    return dict(zip(feature_data["features"], feature_data["features_names"])), feature_data


# ============================================================
# 2. Metric aggregation across Monte-Carlo iterations
# ============================================================

def aggregate_metrics(results_dict, mean_fpr=None):
    """Collect accuracy, AUC, ECE, probability dispersion, and ROC curves
    across all iterations. Returns a dict of arrays/lists plus interpolated TPRs.
    """
    if mean_fpr is None:
        mean_fpr = np.linspace(0, 1, 100)

    acc, aucs, ece, proba_dispersion, tprs = [], [], [], [], []

    for res in results_dict.values():
        y_pred = res["predictions"]
        y_proba = res["proba_predictions"]
        y_true = res["true_values"]

        acc.append(accuracy_score(y_true, y_pred))
        aucs.append(res["auc_test"])
        ece.append(
            ResultsCalculus.expected_calibration_error(
                y_true, y_proba, n_bins=5, strategy="quantile"
            )
        )
        # NOTE: this is mean absolute deviation of predicted probabilities
        # around their own mean — a confidence-dispersion metric, NOT an
        # error metric on ECE itself. Renamed from "mad_ece" to avoid
        # implying it measures calibration error.
        proba_dispersion.append(np.mean(np.abs(y_proba - np.mean(y_proba))))

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    return {
        "accuracy": np.array(acc),
        "auc": np.array(aucs),
        "ece": np.array(ece),
        "proba_dispersion": np.array(proba_dispersion),
        "tprs": np.array(tprs),
        "mean_fpr": mean_fpr,
    }


def print_metrics_summary(name, n_iterations, metrics):
    print(f"Results for: {name} ({n_iterations} it.)")
    print(f"Accuracy: {metrics['accuracy'].mean():.3f} ± {metrics['accuracy'].std():.3f}")
    print(f"AUC:      {metrics['auc'].mean():.3f} ± {metrics['auc'].std():.3f}")
    print(f"ECE:      {metrics['ece'].mean():.3f} ± {metrics['ece'].std():.3f}")
    print(
        f"Proba dispersion (mean abs deviation from mean proba): "
        f"{metrics['proba_dispersion'].mean():.3f} ± {metrics['proba_dispersion'].std():.3f}"
    )


def save_metrics_text(name, n_iterations, metrics, out_dir="results/catboost_results"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text = (
        f"Results for: {name}\n"
        f"Number of iterations: {n_iterations}\n\n"
        f"Accuracy:\nMean = {metrics['accuracy'].mean():.3f}\nStd  = {metrics['accuracy'].std():.3f}\n\n"
        f"AUC:\nMean = {metrics['auc'].mean():.3f}\nStd  = {metrics['auc'].std():.3f}\n\n"
        f"Expected Calibration Error:\nMean = {metrics['ece'].mean():.3f}\nStd  = {metrics['ece'].std():.3f}\n\n"
        f"Probability dispersion (mean abs deviation from mean proba):\n"
        f"Mean = {metrics['proba_dispersion'].mean():.3f}\nStd  = {metrics['proba_dispersion'].std():.3f}\n"
    )
    (out_dir / f"{name}_metrics.txt").write_text(text, encoding="utf-8")


# ============================================================
# 3. SHAP aggregation
# ============================================================

def aggregate_shap(results_dict, cols_to_keep):
    shap_records, feature_imp_records = [], []

    for rdm_state, res in results_dict.items():
        df_shap = pd.DataFrame(res["shap_values"], index=res["index_test"], columns=cols_to_keep)
        shap_records.append(df_shap)

        feature_imp_df = (
            res["model_fts_imp"]
            .set_index("Feature Id")
            .rename(columns={"Importances": rdm_state})
        )
        feature_imp_records.append(feature_imp_df)

    all_shap = pd.concat(shap_records)
    all_feature_imp = pd.concat(feature_imp_records, axis=1)

    mean_shap_per_patient = all_shap.groupby(all_shap.index).mean()
    std_shap_per_patient = all_shap.groupby(all_shap.index).std()

    mean_feature_imp = all_feature_imp.mean(axis=1).sort_values(ascending=False)
    std_feature_imp = all_feature_imp.std(axis=1)

    abs_mean_shap_global = mean_shap_per_patient.abs().mean(axis=0).sort_values(ascending=False)
    mean_shap_global = mean_shap_per_patient.mean(axis=0).reindex(abs_mean_shap_global.index)
    std_shap_global = std_shap_per_patient.mean(axis=0).reindex(abs_mean_shap_global.index)

    return {
        "mean_shap_per_patient": mean_shap_per_patient,
        "std_shap_per_patient": std_shap_per_patient,
        "mean_feature_imp": mean_feature_imp,
        "std_feature_imp": std_feature_imp,
        "abs_mean_shap_global": abs_mean_shap_global,
        "mean_shap_global": mean_shap_global,
        "std_shap_global": std_shap_global,
    }


def print_shap_summary(shap_agg):
    summary_df = pd.concat(
        [shap_agg["abs_mean_shap_global"], shap_agg["std_shap_global"]],
        keys=["Abs means", "StD"],
        axis=1,
    )
    print(f"Global mean SHAP values:\n{summary_df.to_markdown()}")

    imp_df = pd.concat(
        [shap_agg["mean_feature_imp"], shap_agg["std_feature_imp"]],
        keys=["Means", "StD"],
        axis=1,
    )
    print(f"Global feature importance (CatBoost):\n{imp_df.to_markdown()}")


# ============================================================
# 4. Plotting helpers
# ============================================================

def _savefig(fig_or_none, path):
    plt.tight_layout()
    plt.gcf().savefig(path, format="svg", bbox_inches="tight")
    plt.close("all")


def plot_shap_bar(mean_shap_per_patient, X_test_mean, feature_names, name, n_iterations, out_dir):
    shap.summary_plot(
        mean_shap_per_patient.values,
        X_test_mean,
        feature_names=feature_names,
        plot_type="bar",
        title=f"Importance SHAP moyenne ({n_iterations} itérations)",
        show=False,
    )
    _savefig(None, f"{out_dir}/{name}_shap_bar.svg")


def plot_shap_summary(mean_shap_per_patient, X_test_mean, feature_names, name, n_iterations, out_dir):
    shap.summary_plot(
        mean_shap_per_patient.values,
        X_test_mean,
        feature_names=feature_names,
        title=f"Importance SHAP moyenne ({n_iterations} itérations)",
        show=False,
    )
    _savefig(None, f"{out_dir}/{name}_shap_summary.svg")


def plot_shap_by_cohort(
    X_test_mean,
    mean_shap_per_patient,
    group_col,
    group_label_map,
    feature_name_dict,
    name,
    out_dir,
    beeswarm_per_group=False,
):
    """
    Generic replacement for the repeated "SHAP by <category>" blocks.

    group_col: column in X_test_mean to group by (e.g. "Sex", "Neurol_cond")
    group_label_map: dict or callable mapping raw values -> display labels.
        Pass a dict for discrete categories, or a callable for continuous
        binning (e.g. a function wrapping pd.cut for delay_injury).
    """
    if group_col not in X_test_mean.columns:
        return None

    if callable(group_label_map):
        groups = group_label_map(X_test_mean[group_col]).astype(str).to_numpy()
    else:
        groups = X_test_mean[group_col].map(group_label_map).to_numpy()

    assert len(groups) == mean_shap_per_patient.shape[0], (
        f"Group length mismatch for '{group_col}': {len(groups)} vs {mean_shap_per_patient.shape[0]}"
    )

    X_sub = X_test_mean.drop(columns=[group_col])
    shap_sub = mean_shap_per_patient.drop(columns=[group_col])
    feature_names = [feature_name_dict.get(f, f) for f in X_sub.columns]

    shap_exp = shap.Explanation(
        values=shap_sub.values, data=X_sub.values, feature_names=feature_names
    )

    plt.close("all")
    shap.plots.bar(shap_exp.cohorts(groups).abs.mean(0), max_display=len(feature_names), show=False)
    _savefig(None, f"{out_dir}/{name}_shap_summary_grouped_by_{group_col}.svg")

    if beeswarm_per_group:
        for label in sorted(set(groups)):
            mask = groups == label
            plt.close("all")
            shap.plots.beeswarm(shap_exp[mask], max_display=len(feature_names), show=False)
            plt.title(f"SHAP beeswarm - {group_col} = {label}")
            _savefig(None, f"{out_dir}/{name}_shap_beeswarm_{group_col}_{label}.svg")

    return shap_exp


def plot_feature_importance_bar(mean_feature_imp, std_feature_imp, feature_name_dict, name, n_iterations, out_dir):
    pretty_names = [feature_name_dict.get(f, f) for f in mean_feature_imp.index]
    y_pos = np.arange(len(mean_feature_imp))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        mean_feature_imp.index,
        mean_feature_imp.values,
        xerr=std_feature_imp[mean_feature_imp.index],
        color="steelblue",
        ecolor="gray",
        capsize=3,
    )

    for i, (feature, mean_val) in enumerate(mean_feature_imp.items()):
        std_val = std_feature_imp[feature]
        margin = 0.02 * mean_feature_imp.max()
        ax.text(mean_val + std_val + margin, i, f"{mean_val:.2f} ± {std_val:.2f}",
                va="center", ha="left", fontsize=9, color="black")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(pretty_names)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Importance")
    ax.set_title(f"Feature Importance CatBoost (mean on {n_iterations} iterations)")

    xmax = (mean_feature_imp.values + std_feature_imp[mean_feature_imp.index].values).max()
    ax.set_xlim(0, xmax * 1.25)

    _savefig(fig, f"{out_dir}/{name}_feature_importance_from_CB_model.svg")


def plot_mean_roc(metrics, name, out_dir):
    mean_fpr = metrics["mean_fpr"]
    mean_tpr = metrics["tprs"].mean(axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = metrics["tprs"].std(axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(mean_fpr, mean_tpr, color="blue", lw=2,
            label=f"Mean ROC (AUC = {metrics['auc'].mean():.2f} ± {metrics['auc'].std():.2f})")

    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.3, label="± 1 std")
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="red", alpha=0.8, label="Random")

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title(f"ROC curve ({len(metrics['auc'])} iterations)")
    ax.legend(loc="lower right")

    _savefig(fig, f"{out_dir}/{name}_ROC_AUC.svg")


# ============================================================
# 5. Orchestration
# ============================================================

def run_pipeline(data_path, dict_path, feature_names_path, out_dir="results/catboost_results"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(dict_path).stem

    print("- " * 10)
    print(f"Name of run: {name}")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("- " * 10)

    data = load_data(data_path)
    data["is_ambulant"] = (data["6MWT_m_pre"] != 0).astype(int)
    results_dict, cols_to_keep = load_results(dict_path)
    feature_name_dict, _ = load_feature_name_map(feature_names_path)
    X = data[cols_to_keep]

    # --- metrics ---
    metrics = aggregate_metrics(results_dict)
    print_metrics_summary(name, len(results_dict), metrics)
    save_metrics_text(name, len(results_dict), metrics, out_dir=out_dir)

    # --- SHAP aggregation ---
    shap_agg = aggregate_shap(results_dict, cols_to_keep)
    print_shap_summary(shap_agg)

    X_test_mean = X.loc[shap_agg["mean_shap_per_patient"].index]
    feature_names = [feature_name_dict.get(f, f) for f in X_test_mean.columns]

    # --- global SHAP plots ---
    plot_shap_bar(shap_agg["mean_shap_per_patient"], X_test_mean, feature_names, name, len(results_dict), out_dir)
    plot_shap_summary(shap_agg["mean_shap_per_patient"], X_test_mean, feature_names, name, len(results_dict), out_dir)

    # --- SHAP by cohort (replaces 5 copy-pasted blocks) ---
    cohort_specs = [
        ("Neurol_cond", {1: "SCI", 2: "Stroke", 3: "Others"}, False),
        ("Sex", {1: "M", 2: "F"}, True),
        ("functional_level", {i: str(i) for i in range(6)}, False),
        (
            "delay_injury",
            lambda s: pd.cut(s, bins=[0, 7, 180, np.inf], labels=["acute", "sub-acute", "chronic"]),
            False,
        ),
    ]
    for group_col, label_map, do_beeswarm in cohort_specs:
        plot_shap_by_cohort(
            X_test_mean, shap_agg["mean_shap_per_patient"], group_col, label_map,
            feature_name_dict, name, out_dir, beeswarm_per_group=do_beeswarm,
        )

    # --- feature importance + ROC ---
    plot_feature_importance_bar(
        shap_agg["mean_feature_imp"], shap_agg["std_feature_imp"], feature_name_dict,
        name, len(results_dict), out_dir,
    )
    plot_mean_roc(metrics, name, out_dir)

    print(f"All figures saved to {out_dir}/")
    return {"metrics": metrics, "shap_agg": shap_agg, "name": name}


# ============================================================
# CONFIG — edit and run
# ============================================================

if __name__ == "__main__":
    DATA_PATH = (
        "/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/"
        "Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/"
        "CHUNantes collaboration/donnees/data_from_dpi/final_data_matrix_sessions_separated.xlsx"
    )
    DICT_PATH = "results/catboost_results/profile_data/with_no_fuite_selected_features_wout_0.7_correlated/catboost_results_separated_sessions_Neurol_cond_Lesion_num_Nb_sessions_Sex_BMI_6MWT_m_pre_delay_injury_functional_level_speed_selected_features_by_combi.pkl"
    FEATURE_NAMES_PATH = "/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/CHUNantes collaboration/donnees/others/feature_names.xlsx"

    run_pipeline(DATA_PATH, DICT_PATH, FEATURE_NAMES_PATH)