import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score


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


def pruning(X, y, params, random_state):

    # pruning_path = DecisionTreeClassifier(**params,
    #                                       random_state=random_state
    #                                       ).cost_complexity_pruning_path(X,y)
    # clf = DecisionTreeClassifier(**params,
    #                              ccp_alpha=,
    #                              random_state=random_state)
    pass


def main(data_path, features, cat_cols, random_state=42):
    # load data
    X, y, features_names, class_names = load_data(data_path, features, cat_cols)
    print(features_names)

    params = dict(criterion="gini", max_depth=3, min_samples_leaf=9)
    # res_pruning = pruning_1se(X, y, params)

    clf = DecisionTreeClassifier(**params, ccp_alpha=0, random_state=random_state)
    clf.fit(X, y)

    plt.figure(figsize=(10, 8))
    plot_tree(clf, feature_names=features_names, fontsize=14)
    plt.show()


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
        "Nb sessions",
        "BMI",
        "Age",
    ]

    cat_cols = ["Neurol_cond"]

    main(data_path, features, cat_cols)
