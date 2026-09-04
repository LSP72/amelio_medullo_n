"""
shap_rank_stability.py
======================
Quantify how much SHAP-based feature *rankings* vary across repeated trials.

Input
-----
A pickle (.pck / .pkl) holding the results of N iterations. The container may be
a list, tuple, or dict of per-iteration dictionaries. Each per-iteration dict is
assumed to follow the schema:

    {
        "random_state":     int,
        "model":            fitted estimator (sklearn / xgboost / lightgbm ...),
        "index_train":      X_train.index,
        "index_test":       y_test.index,
        "predictions":      y_pred,
        "proba_predictions":y_pred_proba,
        "auc_test":         float,
        "true_values":      y_test,
        "shap_values":      shap values for the test set,
        "model_fts_imp":    feature_imp_df (native model importance),
        "f1_score":         float,
        "classif_report":   str,
    }

What it does
------------
1. For every iteration, turns feature importance into a *rank* (rank 1 = most
   important). Importance source is SHAP by default: mean(|SHAP|) per feature.
2. Builds a rank matrix (iterations x features).
3. Reports, per feature: mean / median / SD / min / max / IQR of rank, the modal
   rank and how often the feature sits there, number of distinct ranks it takes,
   a normalised rank-entropy (0 = perfectly stable, 1 = uniform), and P(feature
   in top-k).
4. Reports, globally: Kendall's W (coefficient of concordance) and mean pairwise
   Spearman rho across iterations, plus top-k set stability (mean Jaccard).
5. Saves a per-feature CSV, a rank-frequency CSV, and two figures: a rank
   boxplot (features ordered by median rank) and a rank-frequency heatmap.

The boxplot is the figure that carries the argument: stable (usually top)
features show tight boxes; unstable (usually lower) features show wide boxes.

Usage
-----
    python shap_rank_stability.py results.pck
    python shap_rank_stability.py results.pck --rank-source model --top-k 15

"""

from __future__ import annotations

import argparse
import itertools
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_iterations(path: str) -> list[dict]:
    """Load the pickle and return a flat list of per-iteration dicts."""
    with open(path, "rb") as fh:
        obj = pickle.load(fh)

    if isinstance(obj, dict):
        # Could be a single iteration, or a dict keyed by e.g. random_state.
        if "shap_values" in obj or "model_fts_imp" in obj:
            iters = [obj]                       # a single iteration dict
        else:
            iters = list(obj.values())          # dict of iterations
    elif isinstance(obj, (list, tuple)):
        iters = list(obj)
    else:
        raise TypeError(f"Unsupported pickle top-level type: {type(obj)!r}")

    iters = [it for it in iters if isinstance(it, dict)]
    if not iters:
        raise ValueError("No per-iteration dictionaries found in the pickle.")
    return iters


# --------------------------------------------------------------------------- #
# Feature names (in model/column order, so they align with SHAP columns)
# --------------------------------------------------------------------------- #
def get_feature_names(iterations: list[dict], n_features: int,
                      override: list[str] | None = None) -> list[str]:
    """Resolve feature names in the SAME order as the SHAP value columns.

    Priority: explicit override -> shap.Explanation.feature_names ->
    fitted-model attributes -> column order of a feature_imp_df that is NOT
    pre-sorted -> generic f0..f{n-1}.
    """
    if override is not None:
        if len(override) != n_features:
            raise ValueError(
                f"override feature_names has {len(override)} entries, "
                f"but SHAP has {n_features} features.")
        return list(override)

    for it in iterations:
        sv = it.get("shap_values", None)
        names = getattr(sv, "feature_names", None)  # shap.Explanation
        if names is not None and len(names) == n_features:
            return list(names)

    for it in iterations:
        model = it.get("model", None)
        if model is None:
            continue
        for attr in ("feature_names_in_", "feature_name_", "feature_names_"):
            names = getattr(model, attr, None)
            if callable(names):
                try:
                    names = names()
                except Exception:
                    names = None
            if names is not None and len(list(names)) == n_features:
                return list(names)
        # xgboost booster
        try:
            booster = model.get_booster()
            names = booster.feature_names
            if names is not None and len(names) == n_features:
                return list(names)
        except Exception:
            pass

    # feature_imp_df fallback -- only trustworthy if it is in column order.
    for it in iterations:
        fi = it.get("model_fts_imp", None)
        if isinstance(fi, pd.DataFrame) and len(fi) == n_features:
            names = _feature_names_from_imp_df(fi)
            if names is not None:
                warnings.warn(
                    "Feature names taken from model_fts_imp. If that dataframe "
                    "was sorted by importance, SHAP columns may be mislabelled. "
                    "Pass --feature-names or store model.feature_names_in_ to be "
                    "safe.", stacklevel=2)
                return names

    warnings.warn("Could not resolve feature names; using generic f0..fN.",
                  stacklevel=2)
    return [f"f{i}" for i in range(n_features)]


def _feature_names_from_imp_df(fi: pd.DataFrame) -> list[str] | None:
    name_cols = ("feature", "features", "name", "col", "column", "variable")
    for c in fi.columns:
        if str(c).lower() in name_cols:
            return fi[c].astype(str).tolist()
    # otherwise assume the index holds names
    if fi.index.dtype == object:
        return fi.index.astype(str).tolist()
    return None


# --------------------------------------------------------------------------- #
# Importance per iteration
# --------------------------------------------------------------------------- #
def normalize_shap(shap_values, class_index: int = 1) -> np.ndarray:
    """Return a 2D (n_samples, n_features) array of SHAP values for one output."""
    # shap.Explanation -> take .values
    values = getattr(shap_values, "values", shap_values)
    values = np.asarray(values)

    if values.ndim == 2:                       # (n_samples, n_features)
        return values
    if values.ndim == 3:
        # Either (n_classes, n_samples, n_features) or
        # (n_samples, n_features, n_classes). Pick the class axis heuristically.
        if values.shape[0] <= values.shape[2]:  # class axis is first
            idx = min(class_index, values.shape[0] - 1)
            return values[idx]
        idx = min(class_index, values.shape[2] - 1)  # class axis is last
        return values[:, :, idx]
    raise ValueError(f"Unexpected SHAP ndim={values.ndim} (shape={values.shape}).")


def shap_importance(shap_values, feature_names: list[str],
                    class_index: int = 1) -> pd.Series:
    """Global SHAP importance = mean(|SHAP|) over samples, per feature."""
    if isinstance(shap_values, list):
        # old TreeExplainer: list per class
        idx = min(class_index, len(shap_values) - 1)
        arr = np.asarray(shap_values[idx])
    else:
        arr = normalize_shap(shap_values, class_index)
    imp = np.abs(arr).mean(axis=0)
    if imp.shape[0] != len(feature_names):
        raise ValueError(
            f"SHAP has {imp.shape[0]} features but {len(feature_names)} names.")
    return pd.Series(imp, index=feature_names)


def model_importance(fi: pd.DataFrame, feature_names: list[str]) -> pd.Series:
    """Importance from the stored feature_imp_df, aligned to feature_names."""
    if not isinstance(fi, pd.DataFrame):
        raise TypeError("model_fts_imp is not a DataFrame.")
    imp_col = None
    for c in fi.columns:
        if str(c).lower() in ("importance", "imp", "gain", "weight",
                              "value", "score", "mean_abs_shap", "shap"):
            imp_col = c
            break
    if imp_col is None:
        imp_col = fi.select_dtypes("number").columns[-1]

    names = _feature_names_from_imp_df(fi)
    if names is not None and len(names) == len(fi):
        s = pd.Series(fi[imp_col].to_numpy(), index=names)
    else:
        s = pd.Series(fi[imp_col].to_numpy(), index=feature_names)
    return s.reindex(feature_names)


def iteration_importance(it: dict, feature_names: list[str], rank_source: str,
                         class_index: int) -> pd.Series:
    if rank_source == "shap":
        return shap_importance(it["shap_values"], feature_names, class_index)
    if rank_source == "model":
        return model_importance(it["model_fts_imp"], feature_names)
    raise ValueError("rank_source must be 'shap' or 'model'.")


# --------------------------------------------------------------------------- #
# Rank matrix + statistics
# --------------------------------------------------------------------------- #
def build_rank_matrix(iterations, feature_names, rank_source, class_index):
    """Rows = iterations, cols = features, values = rank (1 = most important)."""
    rows = []
    for it in iterations:
        imp = iteration_importance(it, feature_names, rank_source, class_index)
        rows.append(imp.reindex(feature_names))
    imp_df = pd.DataFrame(rows).reset_index(drop=True)
    # higher importance -> smaller (better) rank
    rank_df = imp_df.rank(axis=1, ascending=False, method="average")
    return rank_df, imp_df


def per_feature_stats(rank_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    n_feats = rank_df.shape[1]
    out = {}
    for feat in rank_df.columns:
        r = rank_df[feat]
        counts = r.round().astype(int).value_counts(normalize=True)
        p = counts.to_numpy()
        entropy = -(p * np.log(p)).sum()
        max_entropy = np.log(len(counts)) if len(counts) > 1 else 1.0
        out[feat] = {
            "mean_rank": r.mean(),
            "median_rank": r.median(),
            "sd_rank": r.std(ddof=1),
            "min_rank": r.min(),
            "max_rank": r.max(),
            "iqr_rank": r.quantile(0.75) - r.quantile(0.25),
            "modal_rank": int(counts.idxmax()),
            "p_at_modal_rank": float(counts.max()),
            "n_distinct_ranks": int(r.round().nunique()),
            "rank_entropy_norm": float(entropy / max_entropy) if max_entropy else 0.0,
            f"p_in_top_{top_k}": float((r <= top_k).mean()),
        }
    stats = pd.DataFrame(out).T
    stats = stats.sort_values("median_rank")
    stats.index.name = "feature"
    return stats


def kendalls_w(rank_df: pd.DataFrame) -> float:
    """Coefficient of concordance across iterations (0 = none, 1 = perfect)."""
    m, n = rank_df.shape                       # m judges (iters), n items
    R = rank_df.sum(axis=0).to_numpy()         # summed rank per feature
    S = ((R - R.mean()) ** 2).sum()
    denom = (m ** 2) * (n ** 3 - n)
    return float(12 * S / denom) if denom else float("nan")


def mean_pairwise_spearman(rank_df: pd.DataFrame, max_pairs: int = 5000) -> float:
    idx = list(range(len(rank_df)))
    pairs = list(itertools.combinations(idx, 2))
    if len(pairs) > max_pairs:                 # subsample for very large N
        rng = np.random.default_rng(0)
        pairs = [pairs[i] for i in rng.choice(len(pairs), max_pairs, replace=False)]
    vals = [spearmanr(rank_df.iloc[a], rank_df.iloc[b]).statistic for a, b in pairs]
    return float(np.nanmean(vals))


def topk_set_stability(rank_df: pd.DataFrame, k: int) -> dict:
    """How stable the *identity* of the top-k set is across iterations."""
    topsets = [set(rank_df.iloc[i].nsmallest(k).index) for i in range(len(rank_df))]
    # frequency each feature appears in the top-k
    freq = pd.Series(0, index=rank_df.columns, dtype=float)
    for s in topsets:
        for f in s:
            freq[f] += 1
    freq /= len(topsets)
    # mean pairwise Jaccard of the top-k sets
    pairs = list(itertools.combinations(range(len(topsets)), 2))
    if len(pairs) > 5000:
        rng = np.random.default_rng(0)
        pairs = [pairs[i] for i in rng.choice(len(pairs), 5000, replace=False)]
    jac = [len(topsets[a] & topsets[b]) / len(topsets[a] | topsets[b])
           for a, b in pairs]
    return {
        "k": k,
        "mean_jaccard": float(np.mean(jac)),
        "n_features_ever_in_topk": int((freq > 0).sum()),
        "n_features_always_in_topk": int((freq == 1).sum()),
        "membership_frequency": freq.sort_values(ascending=False),
    }


def rank_frequency_matrix(rank_df: pd.DataFrame) -> pd.DataFrame:
    """features x rank-position, entries = fraction of iterations."""
    n = rank_df.shape[1]
    positions = list(range(1, n + 1))
    freq = pd.DataFrame(0.0, index=rank_df.columns, columns=positions)
    r_int = rank_df.round().astype(int)
    for feat in rank_df.columns:
        vc = r_int[feat].value_counts(normalize=True)
        for pos, frac in vc.items():
            if pos in freq.columns:
                freq.loc[feat, pos] = frac
    order = rank_df.mean(axis=0).sort_values().index
    return freq.loc[order]


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_rank_boxplot(rank_df: pd.DataFrame, path: str, top_n: int | None = None):
    order = rank_df.median(axis=0).sort_values().index.tolist()
    if top_n:
        order = order[:top_n]
    data = [rank_df[f].to_numpy() for f in order]
    height = max(4, 0.32 * len(order))
    fig, ax = plt.subplots(figsize=(9, height))
    ax.boxplot(data, vert=False, showfliers=False,
               medianprops=dict(color="#c0392b"))
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels(order, fontsize=8)
    ax.invert_yaxis()                          # most important feature on top
    ax.set_xlabel("Rank across iterations (1 = most important)")
    ax.set_title("SHAP feature-rank variability")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rank_heatmap(freq: pd.DataFrame, path: str, top_n: int | None = None):
    f = freq.iloc[:top_n] if top_n else freq
    fig, ax = plt.subplots(figsize=(min(12, 0.5 * f.shape[1] + 3),
                                    max(4, 0.32 * f.shape[0])))
    im = ax.imshow(f.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(f.shape[1]))
    ax.set_xticklabels(f.columns, fontsize=7)
    ax.set_yticks(range(f.shape[0]))
    ax.set_yticklabels(f.index, fontsize=8)
    ax.set_xlabel("Rank position")
    ax.set_title("Fraction of iterations at each rank position")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def analyse(path: str, rank_source: str = "shap", class_index: int = 1,
            top_k: int = 10, feature_names: list[str] | None = None,
            out_dir: str = "rank_stability_out") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    iterations = load_iterations(path)
    print(f"Loaded {len(iterations)} iterations from {path}")

    # infer n_features from a SHAP array
    probe = normalize_shap(iterations[0]["shap_values"], class_index) \
        if not isinstance(iterations[0]["shap_values"], list) \
        else np.asarray(iterations[0]["shap_values"][min(class_index,
              len(iterations[0]["shap_values"]) - 1)])
    n_features = probe.shape[1]
    names = get_feature_names(iterations, n_features, feature_names)

    rank_df, imp_df = build_rank_matrix(iterations, names, rank_source, class_index)
    stats = per_feature_stats(rank_df, top_k)
    W = kendalls_w(rank_df)
    rho = mean_pairwise_spearman(rank_df)
    topk = topk_set_stability(rank_df, top_k)
    freq = rank_frequency_matrix(rank_df)

    # save
    stats.to_csv(os.path.join(out_dir, "per_feature_rank_stats.csv"))
    freq.to_csv(os.path.join(out_dir, "rank_frequency.csv"))
    rank_df.to_csv(os.path.join(out_dir, "rank_matrix.csv"), index_label="iteration")
    plot_rank_boxplot(rank_df, os.path.join(out_dir, "rank_boxplot.png"))
    plot_rank_heatmap(freq, os.path.join(out_dir, "rank_heatmap.png"))

    print(f"\nGlobal ranking stability across {len(iterations)} iterations:")
    print(f"  Kendall's W            : {W:.3f}  (0 = no agreement, 1 = perfect)")
    print(f"  mean pairwise Spearman : {rho:.3f}")
    print(f"  top-{top_k} set mean Jaccard : {topk['mean_jaccard']:.3f}")
    print(f"  features always in top-{top_k}: {topk['n_features_always_in_topk']}"
          f" / ever in top-{top_k}: {topk['n_features_ever_in_topk']}")
    print(f"\nMost stable features (low SD of rank):")
    print(stats[["median_rank", "sd_rank", "p_at_modal_rank",
                 f"p_in_top_{top_k}"]].head(min(10, len(stats))).round(3)
          .to_string())
    print(f"\nWritten to: {out_dir}/")

    return {"rank_df": rank_df, "importance_df": imp_df, "stats": stats,
            "kendalls_w": W, "mean_spearman": rho, "topk": topk,
            "rank_frequency": freq, "feature_names": names}


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="SHAP feature-rank stability analysis.")
    p.add_argument("pickle", help="Path to the .pck results file.")
    p.add_argument("--rank-source", choices=["shap", "model"], default="shap",
                   help="Rank by mean|SHAP| (default) or by model_fts_imp.")
    p.add_argument("--class-index", type=int, default=1,
                   help="Class to use if SHAP is per-class (binary: 1).")
    p.add_argument("--top-k", type=int, default=10,
                   help="k for top-k stability summaries.")
    p.add_argument("--out-dir", default="rank_stability_out")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    analyse(args.pickle, rank_source=args.rank_source,
            class_index=args.class_index, top_k=args.top_k,
            out_dir=args.out_dir)