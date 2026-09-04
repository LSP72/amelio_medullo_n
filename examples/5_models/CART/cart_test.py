import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score

"""
This script uses CART according to the PREP2 algorithm, proposed by Stinear et al. (2017)

"""


def load_data(data_path, features, cat_cols):
    all_data = pd.read_excel(data_path)
    ct = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough",
    )
    all_data.dropna(subset=["Sex"], inplace=True, axis=0)
    all_data.replace({"Sex": {"M": 0, "F": 1}}, inplace=True)
    X = ct.fit_transform(all_data[features])
    y = all_data["MCID_classes"].to_numpy()
    features_names = ct.get_feature_names_out()
    class_names = np.unique(y)

    return X, y, features_names, class_names


# 1) Node information extraction
def node_table(clf, feature_names, class_names):
    t = clf.tree_
    lignes = []
    for i in range(t.node_count):
        feuille = (
            t.children_left[i] == -1
        )  # Check if the left child of the root node is a leaf node (i.e., it has no children). If it is a leaf node, this will return True; otherwise, it will return False.
        prop = t.value[i][0]  # proportions from total number of each class at the node i
        n_i = t.n_node_samples[i]  # number of samples at node i
        counts = np.round(prop * n_i).astype(int)  # -> Counts of each class at node i

        ligne = {
            "node": i,
            "type": "leaf" if feuille else "internal",
            "n": n_i,  # number of samples used in node i
            "sample_pct": round(
                100 * n_i / t.n_node_samples[0], 1
            ),  # percentage of samples at node i relative to the total number of samples in the dataset (root node = node 0).
            "gini": round(t.impurity[i], 4),
            "rule": (
                "-" if feuille else f"{feature_names[t.feature[i]]} <= {t.threshold[i]:.2f}"
            ),  # only created for internal nodes
            # t.feature[0] = index of the feature used for splitting at node 0
            "left": t.children_left[i],  # index of the left child node of node i
            "right": t.children_right[i],  # index of the left child node of node i
            "prediction": class_names[int(np.argmax(prop))],
        }
        for k, cl in enumerate(class_names):
            ligne[f"n_{cl}"] = counts[k]
            ligne[f"%_{cl}"] = round(100 * prop[k], 1)
        lignes.append(ligne)
    return pd.DataFrame(lignes)


# 2) "Improvement" information extraction in each split
# delta_G = G(parent) - [ w_left * G(left) + w_right * G(right) ]
#                           w/ w = count in child / count in parent.


def improvements(clf, feature_names):
    t = clf.tree_
    out = []
    for i in range(t.node_count):
        l, r = t.children_left[i], t.children_right[i]
        if l == -1:
            continue  #
        w = (
            t.weighted_n_node_samples
        )  # take into account any weight attributed to the samples, if any (in case of class imbalance, ponderation, etc.)
        delta_G = t.impurity[i] - (
            w[l] / w[i] * t.impurity[l] + w[r] / w[i] * t.impurity[r]
        )  # Calculation of deltaGini
        out.append(
            {
                "node": i,
                "split": f"{feature_names[t.feature[i]]} <= {t.threshold[i]:.2f}",
                "gini_parent": round(t.impurity[i], 4),
                "gini_left": round(t.impurity[l], 4),
                "gini_right": round(t.impurity[r], 4),
                "delta_gini": round(delta_G, 4),
                # version balanced by the total count = base for feature_importances_ => ensures that a very pur small child doesn't count like a big child
                "delta_balanced": round(
                    delta_G * w[i] / w[0], 4
                ),  # avg impurity of children => the delta is what have been gained
            }
        )
    return pd.DataFrame(out)


# 3) Cost-complexity pruning
# Collecting pruning track, evaluating each alpha by CV,
# retaining simplest tree with score ≤ 1*SE of the best score


def pruning_1se(X, y, base_params, cv=10, random_state=42):
    """
    Finds the smallest tree whose performance is indiscernible from the best

    Parameters
    ----------
    X : _type_
        _description_
    y : _type_
        _description_
    base_params : _type_
        _description_
    cv : int, optional
        _description_, by default 10
    random_state : int, optional
        _description_, by default 42

    Returns
    -------
    _type_
        _description_
    """
    track = DecisionTreeClassifier(**base_params, random_state=random_state).cost_complexity_pruning_path(X, y)
    alphas = np.unique(track.ccp_alphas[track.ccp_alphas >= 0])

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    means, std_err = [], []
    for a in alphas:
        m = DecisionTreeClassifier(**base_params, ccp_alpha=a, random_state=random_state)
        s = cross_val_score(m, X, y, cv=skf, scoring="accuracy")
        means.append(s.mean())
        std_err.append(s.std(ddof=1) / np.sqrt(cv))

    means, std_err = np.array(means), np.array(std_err)
    i_best = int(np.argmax(means))
    threshold = means[i_best] - std_err[i_best]  # ≤ 1*SE
    # highest alpha (i.e., simplest tree, probably only the root) above the threshold
    candidates = np.where(means >= threshold)[0]
    i_1se = int(candidates[-1])

    return {
        "alpha_best": alphas[i_best],
        "acc_cv_best": means[i_best],
        "alpha_1se": alphas[i_1se],
        "acc_cv_1se": means[i_1se],
        "table": pd.DataFrame({"ccp_alpha": alphas, "acc_cv": means.round(4), "se": std_err.round(4)}),
    }


# 4) PERFORMANCE metrics: confusion matrix, PPV, NPV, CI 95 %
# =========================================================================
# PPV(k) = VP / (VP + FP)  -> among patients predicted k, how many are they
# NPV(k) = VN / (VN + FN)  -> among patients NON among predicted patients k, how many are they


def ic_wilson(success, total, z=1.96):
    """Wilson 95% CI: more robus on small sample size."""
    if total == 0:
        return (np.nan, np.nan)
    p = success / total
    d = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / d
    demi = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / d
    return (round(100 * (centre - demi), 1), round(100 * (centre + demi), 1))


def table_performance(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    lignes = []
    for k, cl in enumerate(class_names):
        VP = cm[k, k]
        FP = cm[:, k].sum() - VP
        FN = cm[k, :].sum() - VP
        VN = cm.sum() - VP - FP - FN
        lignes.append(
            {
                "categorie": cl,
                "n_true": int(cm[k, :].sum()),
                "n_predicted": int(cm[:, k].sum()),
                "PPV_%": round(100 * VP / (VP + FP), 1) if (VP + FP) else np.nan,
                "PPV_95CI": ic_wilson(VP, VP + FP),
                "NPV_%": round(100 * VN / (VN + FN), 1) if (VN + FN) else np.nan,
                "NPV_IC95": ic_wilson(VN, VN + FN),
            }
        )
    return pd.DataFrame(lignes), cm


# 5) Displaying information
def display_metrics(
    clf, clf_no_pruning, FEATURES, CLASSES, nodes, ameliorations, res_pruning, matrix, perf, acc_apparente, acc_cv
):
    pd.set_option("display.width", 200, "display.max_columns", 40)

    print("=" * 72)
    print("TREE (with rules)")
    print("=" * 72)
    print(export_text(clf, feature_names=FEATURES, class_names=CLASSES, decimals=1))

    print("=" * 72)
    print("NODE TABLE")
    print("=" * 72)
    print(nodes[["node", "type", "n", "sample_pct", "gini", "rule", "prediction"]].to_string(index=False))
    print()
    print(nodes[["node"] + [f"%_{c}" for c in CLASSES]].to_string(index=False))

    print()
    print("=" * 72)
    print("IMPROVEMENT PER SPLIT (Gini's reduction)")
    print("=" * 72)
    print(ameliorations.to_string(index=False))

    print()
    print("=" * 72)
    print("COST-COMPLEXITY PRUNING / 1-SE RULE")
    print("=" * 72)
    print(f"optimal alpha      : {res_pruning['alpha_best']:.5f} " f"(acc CV = {res_pruning['acc_cv_best']:.3f})")
    print(f"retained alpha (1-SE): {res_pruning['alpha_1se']:.5f} " f"(acc CV = {res_pruning['acc_cv_1se']:.3f})")
    print(f"leaves without / with pruning : " f"{clf_no_pruning.get_n_leaves()} / {clf.get_n_leaves()}")

    print()
    print("=" * 72)
    print("PERFORMANCE")
    print("=" * 72)
    print("Confusion MATRIX (lines = true, colunms = predict)")
    print(pd.DataFrame(matrix, index=CLASSES, columns=CLASSES))
    print()
    print(perf.to_string(index=False))
    print()
    print(f"Apparent accuracy (learning) : {acc_apparente:.1%}  <- OPTIMISTE")
    print(f"10-CV accuracy validation: {acc_cv:.1%}  <- a citer")

    print()
    print("Feature importance based on Gini:")
    for nom, imp in sorted(zip(FEATURES, clf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {nom:<6} {imp:.3f}")

    # Figure : decommenter pour sauvegarder l'arbre
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(18, 9))
    plot_tree(clf, feature_names=FEATURES, class_names=CLASSES, filled=True, rounded=True, proportion=False, ax=ax)
    # fig.savefig("arbre_cart.png", dpi=200, bbox_inches="tight")


def main(data_path, features, cat_cols, random_state=42):
    # load data
    X, y, features_names, class_names = load_data(data_path, features, cat_cols)

    params = dict(criterion="gini", max_depth=None, min_samples_leaf=9)
    res_pruning = pruning_1se(X, y, params)

    clf_no_pruning = DecisionTreeClassifier(**params, ccp_alpha=0, random_state=random_state)
    clf_no_pruning.fit(X, y)

    clf = DecisionTreeClassifier(
        criterion="gini",  # The function to measure the quality of a split. 'gini' refers to the Gini impurity, which is a measure of how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset.
        max_depth=None,  # Maximum depth of the tree to 3, i.e., the tree will have at most 3 levels of nodes.
        ## A JUSTIFIER !
        min_samples_leaf=9,  # Minimum number of samples required to be at a leaf node (i.e., no further splitting will occur if a node has fewer than 10 samples).
        ## A JUSTIFIER
        ccp_alpha=res_pruning[
            "alpha_1se"
        ],  # Complexity parameter used for Minimal Cost-Complexity Pruning. A value of 0.0 means no pruning will be performed; here, has been optimised.
        random_state=random_state,
    )
    clf.fit(X, y)

    nodes = node_table(clf, features_names, class_names)
    ameliorations = improvements(clf, features_names)

    # Performance
    y_pred = clf.predict(X)
    perf, matrix = table_performance(y, y_pred, class_names)

    # Accuracy apparente (sur les donnees d'apprentissage : OPTIMISTE)
    acc_apparente = accuracy_score(y, y_pred)

    # Accuracy honnete (validation croisee)
    acc_cv = cross_val_score(
        DecisionTreeClassifier(**params, ccp_alpha=res_pruning["alpha_1se"], random_state=random_state),
        X,
        y,
        cv=StratifiedKFold(10, shuffle=True, random_state=random_state),
    ).mean()

    display_metrics(
        clf,
        clf_no_pruning,
        features_names,
        class_names,
        nodes,
        ameliorations,
        res_pruning,
        matrix,
        perf,
        acc_apparente,
        acc_cv,
    )


if __name__ == "__main__":

    data_path = "/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/CHUNantes collaboration/donnees/data_from_dpi/merged_data_final.xlsx"

    features = [
        "duration",
        "Durée_min",
        "Vitesse_kmh_MOY",
        "BWS_%_MOY",
        "step_length",
        "Guidage_%_MOY",
        "sessions_per_week",
        "Neurol_cond",
        "Sex",
        "Nb sessions",
        "BMI",
        "Age",
    ]

    cat_cols = ["Neurol_cond"]

    main(data_path, features, cat_cols)
