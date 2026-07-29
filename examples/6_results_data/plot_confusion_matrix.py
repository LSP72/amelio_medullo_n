"""
Matrice de confusion moyenne (± écart-type) sur N itérations.

Hypothèse : `results` est une liste de dicts, un par itération, contenant
au minimum les clés "true_values" (y_test) et "predictions" (y_pred).
"""

import numpy as np
import pickle as pkl
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def matrices_confusion(results, labels=(0, 1), normalize=None):
    """
    Renvoie (mean_cm, std_cm), chacune de forme (n_classes, n_classes).

    normalize :
        None    -> comptages bruts (TN, FP, FN, TP)
        "true"  -> chaque matrice normalisée par ligne (taux par classe réelle)
                   avant moyennage. À privilégier si les splits ne sont PAS
                   stratifiés (composition de classes variable entre itérations).
    """
    cms = []
    for r in results:
        cm = confusion_matrix(results[r]["true_values"], results[r]["predictions"], labels=labels).astype(float)
        if normalize == "true":
            row_sums = cm.sum(axis=1, keepdims=True)
            # évite la division par 0 si une classe est absente d'un test set
            cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
        cms.append(cm)

    cms = np.array(cms)  # forme (n_iter, n_classes, n_classes)
    mean_cm = cms.mean(axis=0)
    std_cm = cms.std(axis=0, ddof=1)  # ddof=1 = écart-type d'échantillon
    return mean_cm, std_cm


def plot_matrice_confusion_moyenne(
    mean_cm,
    std_cm,
    class_names=("Non-Responders", "Responders"),
    normalize=None,
    cmap="viridis",
    title=None,
):
    """Affiche la matrice avec 'moyenne ± écart-type' dans chaque case."""
    n = mean_cm.shape[0]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mean_cm, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title is None:
        suffixe = " (rate)" if normalize == "true" else " (count)"
        title = f"Confusion Matrix (Mean ± Std Dev) {suffixe}"

    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted Value",
        ylabel="True Value",
        title=title,
    )
    ax.tick_params(labelsize=14)
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    ax.title.set_size(14)
    plt.setp(ax.get_yticklabels(), rotation=90, va="center")

    fmt = ".2f" if normalize == "true" else ".1f"
    seuil = mean_cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{mean_cm[i, j]:{fmt}} ± {std_cm[i, j]:{fmt}}",
                ha="center",
                va="center",
                color="black" if mean_cm[i, j] > seuil else "white",
                fontweight='bold',
                fontsize=15,
            )

    fig.tight_layout()
    return fig, ax


def main(results, output_path=None):
    # Comptages bruts (ce que tu as demandé) :
    mean_cm, std_cm = matrices_confusion(results, labels=(0, 1), normalize=None)
    fig, ax = plot_matrice_confusion_moyenne(mean_cm, std_cm, normalize=None)
    plt.show()

    if output_path:
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = output_path + f"mean_confusion_matrix_{date}.svg"
        fig.savefig(output_name, dpi=300, bbox_inches="tight")
        print(f"plot saved in {output_name}")

    # Variante en taux (si splits non stratifiés) :
    # mean_cm, std_cm = matrices_confusion(results, labels=(0, 1), normalize="true")
    # plot_matrice_confusion_moyenne(mean_cm, std_cm, normalize="true")
    # plt.show()


if __name__ == "__main__":
    # --- Exemple d'utilisation ---
    # data_path = "results/catboost_results/profile_data/selected_features_with_no_fuite/catboost_results_separated_sessions_is_True_selected_features_with_no_fuite.pkl"
    data_path = "results/catboost_results/merged_data/selected_features_with_no_fuite/catboost_results_merged_data_selected_features_with_no_fuite.pkl"
    with open(data_path, "rb") as file:
        results = pkl.load(file)
    
    output_path = "results/catboost_results/"

    main(results, output_path=output_path)

    # Rappel de l'ordre des cases (labels=[0,1]) :
    #   mean_cm[0,0] = TN | mean_cm[0,1] = FP
    #   mean_cm[1,0] = FN | mean_cm[1,1] = TP
