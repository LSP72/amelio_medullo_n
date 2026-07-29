"""
Récupération des métriques de classification à partir du .pkl de résultats CatBoost.

Chaque dictionnaire sauvegardé (un par random_state) contient notamment :
    - "true_values"       : y_test  (pandas Series, valeurs réelles)
    - "predictions"       : y_pred  (classes prédites)
    - "proba_predictions" : probabilités de la classe positive
    - "auc_test"          : AUC déjà calculée (sert de sanity check)

À partir de true_values + predictions on reconstruit la matrice de confusion,
donc TP / FP / TN / FN, puis toutes les métriques dérivées, split par split,
puis on agrège sur l'ensemble des splits.

Dépendance : catboost doit être installé dans l'environnement, car l'objet
modèle CatBoostClassifier est stocké dans le pkl et doit être désérialisé.
"""

import pickle as pkl

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score


# ── Classe "positive" ─────────────────────────────────────────────────────────
# 1 = répondeur MCID (hypothèse tirée de ton f1_score par défaut). À CONFIRMER.
# Si tu inverses, sensibilité et spécificité s'échangent.
POS_LABEL = 1


def metrics_from_split(y_true, y_pred, y_proba=None, pos_label=POS_LABEL):
    """Reconstruit la matrice de confusion et les métriques pour un split."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()  # gère un éventuel (n, 1) renvoyé par CatBoost

    # Aligner le type des prédictions sur celui des vraies valeurs
    try:
        y_pred = y_pred.astype(y_true.dtype)
    except (ValueError, TypeError):
        y_true = y_true.astype(str)
        y_pred = y_pred.astype(str)
        pos_label = str(pos_label)

    labels = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
    if len(labels) != 2:
        raise ValueError(
            f"Attendu 2 classes, trouvé {labels}. Ce script gère le cas binaire."
        )
    if pos_label not in labels:
        raise ValueError(f"pos_label={pos_label!r} absent des classes {labels}.")

    neg_label = next(l for l in labels if l != pos_label)

    # confusion_matrix avec labels=[neg, pos] -> ravel donne tn, fp, fn, tp
    cm = confusion_matrix(y_true, y_pred, labels=[neg_label, pos_label])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())

    n_pos = tp + fn  # positifs réels
    n_neg = tn + fp  # négatifs réels

    def safe_div(a, b):
        return a / b if b else np.nan

    sensitivity = safe_div(tp, n_pos)          # rappel / TPR
    specificity = safe_div(tn, n_neg)          # TNR
    precision = safe_div(tp, tp + fp)          # VPP
    npv = safe_div(tn, tn + fn)                # VPN

    out = {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "n_pos": n_pos, "n_neg": n_neg, "n_total": n_pos + n_neg,
        "sensitivity_recall_TPR": sensitivity,
        "specificity_TNR": specificity,
        "precision_PPV": precision,
        "NPV": npv,
        "FPR": safe_div(fp, n_neg),
        "FNR": safe_div(fn, n_pos),
        "accuracy": safe_div(tp + tn, n_pos + n_neg),
        "balanced_accuracy": np.nanmean([sensitivity, specificity]),
    }

    if np.isnan(precision) or np.isnan(sensitivity) or (precision + sensitivity) == 0:
        out["f1"] = np.nan
    else:
        out["f1"] = 2 * precision * sensitivity / (precision + sensitivity)

    # AUC recalculée depuis les probas (doit coller à auc_test si pos_label=1)
    if y_proba is not None:
        y_proba = np.asarray(y_proba).ravel()
        try:
            out["AUC"] = roc_auc_score((y_true == pos_label).astype(int), y_proba)
        except ValueError:
            out["AUC"] = np.nan

    return out


def main(pkl_path, csv_out=None, pos_label=POS_LABEL):
    with open(pkl_path, "rb") as f:
        results = pkl.load(f)

    rows = []
    for rdm_state, res in results.items():
        m = metrics_from_split(
            res["true_values"],
            res["predictions"],
            res.get("proba_predictions"),
            pos_label=pos_label,
        )
        m["random_state"] = rdm_state
        if "auc_test" in res:
            m["AUC_stored"] = res["auc_test"]  # sanity check vs AUC recalculée
        rows.append(m)

    df = pd.DataFrame(rows).set_index("random_state").sort_index()

    # ── Agrégation sur l'ensemble des splits ─────────────────────────────────
    metric_cols = [
        "sensitivity_recall_TPR", "specificity_TNR", "precision_PPV", "NPV",
        "accuracy", "balanced_accuracy", "f1", "AUC",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]
    summary = pd.DataFrame({
        "mean":   df[metric_cols].mean(),
        "std":    df[metric_cols].std(),
        "median": df[metric_cols].median(),
        "p2.5":   df[metric_cols].quantile(0.025),
        "p97.5":  df[metric_cols].quantile(0.975),
    })

    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")

    print(f"\n{len(df)} splits chargés depuis :\n  {pkl_path}\n")
    print("── Metrics per split (5 firsts) ─────────────────────────────────")
    print(df.head())
    print("\n── Summary on all splits ───────────────────────────────────────")
    print(summary)

    # Sum of confusion matrix cases (useful for a global report)
    print("\n── Total of confusion matrix cases on all splits ─────────────────")
    print(df[["TP", "FP", "TN", "FN"]].sum())

    if csv_out:
        df.to_csv(csv_out)
        summary_path = csv_out.replace(".csv", "_summary.csv")
        summary.to_csv(summary_path)
        print(f"\nÉcrit :\n  {csv_out}\n  {summary_path}")

    return df, summary


if __name__ == "__main__":
    PKL_PATH = (
        "results/catboost_results/merged_data/selected_features_with_no_fuite/catboost_results_merged_data_selected_features_with_no_fuite.pkl"
    )
    csv_out = "/results/catboost_results/catboost_metrics_per_split.csv"
    main(PKL_PATH, pos_label=1)