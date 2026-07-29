"""
Rebuild shap.Explanation objects from a pickled list/dict of CatBoost runs and
plot heatmaps.

Assumes each run is the dict returned by train_and_test_catboost:
    {"random_state", "model", "index_train", "index_test", "shap_values", ...}

The original feature frame X must be available and UNCHANGED since training,
because X_test is reconstructed as X.loc[run["index_test"]].
"""

import pickle

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from catboost import Pool
from amelio_medullo import DataCleaning, ResultsCalculus


# ── helpers ───────────────────────────────────────────────────────────────────
def _prep_cats(df, cat_features):
    """Reapply exactly the preprocessing done inside train_and_test_catboost."""
    df = df.copy()
    if cat_features:
        df[cat_features] = df[cat_features].fillna("missing").astype(str)
    return df
 
 
def _expected_value(model, X_test, cat_features):
    """
    shap.TreeExplainer(catboost).expected_value is None until shap_values() has
    actually been called: shap doesn't compute the base value itself, it reads it
    off the bias column CatBoost returns. So ask CatBoost. One row suffices --
    the bias is the same for every row.
    """
    pool = Pool(X_test.iloc[:1], cat_features=cat_features)
    arr = np.asarray(model.get_feature_importance(type="ShapValues", data=pool))
    if arr.ndim == 3:                      # multiclass: (n, n_classes, p+1)
        arr = arr[:, 1, :]
    return float(arr[0, -1])               # last column is the bias / E[f(x)]
 
 
def _normalise_shap(sv, X_test, model, cat_features):
    """
    Return (values [n, p], base [n]).
 
    CatBoost binary through shap.TreeExplainer usually gives (n, p). Depending on
    shap/catboost version you can also see (n, p+1) with the bias in the last
    column, or (n, p, 2) with one slice per class. Handle all three rather than
    trusting one.
    """
    sv = np.asarray(sv)
    n_features = X_test.shape[1]
 
    if sv.ndim == 3:                       # (n, p, n_classes) -> positive class
        sv = sv[:, :, 1]
 
    if sv.shape[1] == n_features + 1:      # bias column tacked on the end
        base = sv[:, -1]
        values = sv[:, :-1]
    elif sv.shape[1] == n_features:
        base = np.full(sv.shape[0], _expected_value(model, X_test, cat_features))
        values = sv
    else:
        raise ValueError(
            f"shap array has {sv.shape[1]} columns but X has {n_features} features"
        )
 
    return values, base
 
 
def _check_additivity(run, values, base, tol=1e-3):
    """
    TreeSHAP is additive: base + sum(shap_i) == raw margin == logit(proba_i).
 
    This is worth running once. It doesn't just validate the base value -- it also
    catches the failure mode this whole script is exposed to, namely X.loc[idx]
    coming back in a different row order than the stored SHAP array, or X having
    been mutated since training. A misalignment shows up here as a large residual.
    """
    p = np.clip(np.asarray(run["proba_predictions"], dtype=float), 1e-12, 1 - 1e-12)
    margin = np.log(p / (1 - p))
    resid = np.abs(margin - (base + values.sum(axis=1)))
    worst = resid.max()
    if worst > tol:
        raise AssertionError(
            f"additivity violated (max residual {worst:.4g}). Either the base value "
            f"is wrong, or X.loc[index_test] no longer aligns row-for-row with the "
            f"stored shap_values."
        )
    return worst
 
 
def run_to_explanation(run, X, check=True):
    """One run -> shap.Explanation aligned with its own test set."""
    cat_features = [c for c in X.columns if X[c].dtype == "object"]
    X_test = _prep_cats(X.loc[run["index_test"]], cat_features)
 
    values, base = _normalise_shap(run["shap_values"], X_test, run["model"], cat_features)
 
    if check and "proba_predictions" in run:
        _check_additivity(run, values, base)
 
    return shap.Explanation(
        values=values,
        base_values=base,
        data=X_test.values,
        feature_names=list(X_test.columns),
    )
 
 
def pooled_explanation(runs, X):
    """
    Average each sample's SHAP values over every run in which it was in the test
    set. Samples never tested are dropped.
 
    Caveat worth stating in any caption: this mixes SHAP values from different
    models fitted on different training splits. The resulting f(x) is a mean
    prediction across models, not the output of any single model.
    """
    cat_features = [c for c in X.columns if X[c].dtype == "object"]
    p = X.shape[1]
 
    sums = pd.DataFrame(0.0, index=X.index, columns=X.columns)
    base_sums = pd.Series(0.0, index=X.index)
    counts = pd.Series(0, index=X.index)
 
    for run in runs:
        idx = pd.Index(run["index_test"])
        X_test = _prep_cats(X.loc[idx], cat_features)
        values, base = _normalise_shap(run["shap_values"], X_test, run["model"], cat_features)
        sums.loc[idx] += pd.DataFrame(values, index=idx, columns=X.columns)
        base_sums.loc[idx] += base
        counts.loc[idx] += 1
 
    keep = counts[counts > 0].index
    n_runs_per_sample = counts.loc[keep]
    print(
        f"pooled over {len(runs)} runs | samples kept: {len(keep)}/{len(X)} | "
        f"appearances per sample: min={n_runs_per_sample.min()} "
        f"median={int(n_runs_per_sample.median())} max={n_runs_per_sample.max()}"
    )
 
    mean_vals = sums.loc[keep].div(n_runs_per_sample, axis=0)
    mean_base = base_sums.loc[keep] / n_runs_per_sample
    X_kept = _prep_cats(X.loc[keep], cat_features)
 
    return shap.Explanation(
        values=mean_vals.values,
        base_values=mean_base.values,
        data=X_kept.values,
        feature_names=list(X.columns),
    )
 
 
def plot_heatmap(expl, max_display=15, title=None, outfile=None):
    shap.plots.heatmap(expl, max_display=max_display,  show=False)
    if title:
        plt.title(title)
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.show()


# ── usage ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data_path = "/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/CHUNantes collaboration/donnees/data_from_dpi/final_data_matrix_sessions_separated.xlsx"
    dict_path = "results/catboost_results/profile_data//monte_carlo/catboost_results_separated_sessions_w_four_best.pkl_selected_features_by_combi.pkl"
    with open(dict_path, "rb") as f:
        results = pickle.load(f)

    runs = list(results.values()) if isinstance(results, dict) else results
    data = pd.read_excel(data_path)
    data["Neurol_cond"] = data["Neurol_cond"].replace(["BM", "AVC", "Autre"], [1, 2, 3])
    data["Sex"] = data["Sex"].replace(["M", "F"], [1, 2])
    data = data.apply(DataCleaning.lesion_level_to_num, axis=1)

    first_key = next(iter(results))
    cols_to_keep = results[first_key]["model"].feature_names_
    X = data[cols_to_keep]

    # (a) single run — the defensible, standard plot
    best = max(runs, key=lambda r: r["auc_test"])
    expl = run_to_explanation(best, X)
    plot_heatmap(
        expl,
        title=f"SHAP heatmap — seed {best['random_state']} (AUC {best['auc_test']:.3f})",
        outfile="shap_heatmap_single.png",
    )

    # (b) pooled across runs — read the caveat in pooled_explanation()
    expl_pooled = pooled_explanation(runs, X)
    plot_heatmap(
        expl_pooled,
        title="SHAP heatmap — mean over out-of-fold test appearances"
    )