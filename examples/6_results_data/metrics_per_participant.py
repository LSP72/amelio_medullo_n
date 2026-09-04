"""
Analyse au niveau du PARTICIPANT (et non plus au niveau du split).

Deux sorties distinctes, qui répondent à deux questions différentes :

1. `per_participant` : pour chaque participant, sur combien de splits il a été
   dans le test set, combien de fois il a été bien classé, sa proba moyenne.
   -> répond à "l'erreur est-elle diffuse ou concentrée sur quelques sujets ?"

2. `pooled_metrics` : on agrège d'abord au niveau participant (vote majoritaire
   ou proba moyenne), PUIS on calcule une seule matrice de confusion sur les N
   participants. Chaque participant compte une fois, quel que soit le nombre de
   fois où il est tombé dans un test set.
   -> estimateur plus propre que la moyenne des métriques par split.

3. Bootstrap SUR LES PARTICIPANTS pour un IC honnête (les percentiles calculés
   sur les 100 splits ne sont PAS un IC : les splits partagent les mêmes sujets).

Dépendance : catboost doit être installé (le CatBoostClassifier est dans le pkl).
"""

import pickle as pkl

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

POS_LABEL = 1
RNG = np.random.default_rng(0)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Table longue : une ligne par (participant, split)
# ──────────────────────────────────────────────────────────────────────────────
def build_long_table(results):
    frames = []
    for rdm_state, res in results.items():
        y_true = pd.Series(res["true_values"])
        idx = y_true.index
        n = len(idx)

        y_pred = np.asarray(res["predictions"]).ravel()

        proba = res.get("proba_predictions")
        if proba is None:
            proba_pos = np.full(n, np.nan)
        else:
            proba = np.asarray(proba)
            if proba.ndim == 2 and proba.shape[1] == 2:
                # predict_proba brut -> on prend la colonne de la classe positive
                proba_pos = proba[:, 1]
            else:
                proba_pos = proba.ravel()
            if len(proba_pos) != n:
                raise ValueError(
                    f"split {rdm_state}: {len(proba_pos)} probas pour {n} sujets. "
                    "Vérifie la forme de proba_predictions."
                )

        # build a long table with one row per participant per split for all splits
        frames.append(
            pd.DataFrame(
                {
                    "participant": np.asarray(idx),
                    "random_state": rdm_state,
                    "y_true": y_true.to_numpy(),
                    "y_pred": y_pred,
                    "proba": proba_pos,
                }
            )
        )

    long = pd.concat(frames, ignore_index=True)
    long["correct"] = (long["y_pred"] == long["y_true"]).astype(int)
    return long


# ──────────────────────────────────────────────────────────────────────────────
# 2. Agrégation par participant
# ──────────────────────────────────────────────────────────────────────────────
def per_participant_table(long, pos_label=POS_LABEL):
    # Sanity check : does the participant have the same y_true across splits ? (should be yes)
    incoherent = long.groupby("participant")["y_true"].nunique()
    incoherent = incoherent[incoherent > 1]
    if len(incoherent):
        raise ValueError(
            f"{len(incoherent)} participants ont un y_true qui change selon le split "
            f"(index dupliqué ? fuite ?) : {list(incoherent.index[:5])}"
        )

    g = long.groupby("participant")
    tab = pd.DataFrame(
        {
            "y_true": g["y_true"].first(),
            "n_tested": g.size(),  # how many times the participant was in a test set
            "n_correct": g["correct"].sum(),  # how many times the participant was correctly classified
            "accuracy": g["correct"].mean(),  # accuracy per participant (mean of correct predictions)
            "mean_proba": g["proba"].mean(),
            "std_proba": g["proba"].std(),
            "vote_pos_rate": g["y_pred"].apply(lambda s: (s == pos_label).mean()),
        }
    )
    # Consensus : classe prédite majoritairement sur les splits où le sujet est testé
    tab["pred_vote"] = np.where(tab["vote_pos_rate"] > 0.5, pos_label, 1 - pos_label)
    # Consensus alternatif : seuil 0.5 sur la proba moyenne
    tab["pred_meanproba"] = np.where(tab["mean_proba"] > 0.5, pos_label, 1 - pos_label)
    return tab.sort_values("accuracy")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Métriques calculées UNE fois, au niveau participant
# ──────────────────────────────────────────────────────────────────────────────
def confusion_metrics(y_true, y_pred, y_score=None, pos_label=POS_LABEL):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    neg_label = [l for l in np.unique(y_true) if l != pos_label][0]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[neg_label, pos_label]).ravel()

    def sd(a, b):
        return a / b if b else np.nan

    sens = sd(tp, tp + fn)
    spec = sd(tn, tn + fp)
    prec = sd(tp, tp + fp)

    out = {
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "sensitivity_TPR": sens,
        "specificity_TNR": spec,
        "precision_PPV": prec,
        "NPV": sd(tn, tn + fn),
        "accuracy": sd(tp + tn, tp + tn + fp + fn),
        "balanced_accuracy": np.nanmean([sens, spec]),
        "f1": np.nan if not (prec and sens) else 2 * prec * sens / (prec + sens),
    }
    if y_score is not None and not np.all(np.isnan(y_score)):
        out["AUC"] = roc_auc_score((y_true == pos_label).astype(int), y_score)
    return out


def bootstrap_participants(tab, pred_col, n_boot=2000, pos_label=POS_LABEL):
    """IC en rééchantillonnant les PARTICIPANTS (la vraie unité d'incertitude)."""
    keys = ["sensitivity_TPR", "specificity_TNR", "precision_PPV", "NPV", "accuracy", "balanced_accuracy", "f1", "AUC"]
    boots = []
    n = len(tab)
    for _ in range(n_boot):
        s = tab.iloc[RNG.integers(0, n, n)]
        if s["y_true"].nunique() < 2:
            continue
        m = confusion_metrics(s["y_true"], s[pred_col], s["mean_proba"], pos_label)
        boots.append({k: m.get(k, np.nan) for k in keys})
    b = pd.DataFrame(boots)
    return pd.DataFrame(
        {
            "mean": b.mean(),
            "p2.5": b.quantile(0.025),
            "p97.5": b.quantile(0.975),
        }
    )


# ──────────────────────────────────────────────────────────────────────────────
def main(pkl_path, csv_out=None, pos_label=POS_LABEL, n_boot=2000):
    with open(pkl_path, "rb") as f:
        results = pkl.load(f)

    long = build_long_table(results)
    tab = per_participant_table(long, pos_label)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")

    print(f"\n{long['random_state'].nunique()} splits, {len(tab)} participants uniques.")
    print(
        "Nb de fois testé par participant : "
        f"min={tab.n_tested.min()}, median={tab.n_tested.median():.0f}, max={tab.n_tested.max()}"
    )

    print("\n── Distribution de l'accuracy par participant ────────────────────")
    bins = [-0.001, 0.2, 0.4, 0.6, 0.8, 1.0]
    print(pd.cut(tab["accuracy"], bins).value_counts().sort_index())

    print("\n── 15 participants les plus mal classés ──────────────────────────")
    print(tab.head(15)[["y_true", "n_tested", "n_correct", "accuracy", "mean_proba", "std_proba"]])

    hard = tab[tab["accuracy"] < 0.5]
    print(
        f"\n{len(hard)}/{len(tab)} participants ({100*len(hard)/len(tab):.1f} %) "
        "sont mal classés dans la MAJORITÉ des splits où ils sont testés."
    )
    if len(hard):
        print("  répartition par classe réelle :")
        print(
            pd.crosstab(tab["y_true"], tab["accuracy"] < 0.5, normalize="index").rename(
                columns={False: "ok", True: "hard"}
            )
        )

    print("\n── Métriques au niveau PARTICIPANT (consensus = vote majoritaire) ─")
    m_vote = confusion_metrics(tab["y_true"], tab["pred_vote"], tab["mean_proba"], pos_label)
    print(pd.Series(m_vote))

    print("\n── IC 95 % par bootstrap SUR LES PARTICIPANTS ────────────────────")
    print(bootstrap_participants(tab, "pred_vote", n_boot=n_boot, pos_label=pos_label))

    if csv_out:
        tab.to_csv(csv_out)
        long.to_csv(csv_out.replace(".csv", "_long.csv"), index=False)
        print(f"\nÉcrit :\n  {csv_out}\n  {csv_out.replace('.csv', '_long.csv')}")

    return long, tab


if __name__ == "__main__":
    PKL_PATH = (
        "results/catboost_results/merged_data/selected_features_with_no_fuite/"
        "catboost_results_merged_data_selected_features_with_no_fuite.pkl"
    )
    CSV_OUT = "results/catboost_results/per_participant_accuracy.csv"
    main(PKL_PATH, csv_out=CSV_OUT, pos_label=1)
