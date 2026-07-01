import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # On ajoute seaborn
from matplotlib.colors import ListedColormap  # Pour créer la palette de couleurs
from amelio_medullo import MuscleScore


def main(data_path, dict_mvt_BM):
    data = pd.read_excel(data_path)

    # Création de la palette : 0 = Blanc (Manquant), 1 = Bleu (BM), 2 = Rouge (AVC)
    ma_palette = ListedColormap(["#ffffff", "#3498db", "#e74c3c"])

    for side in ["right", "left"]:
        dict_mvt_BM_transformed = MuscleScore.transform_dict_to_side(dict_mvt_BM, side)

        for muscle_grp, muscles in dict_mvt_BM_transformed.items():
            features_to_map = [muscle_grp] + muscles

            # --- Étape 1 : Créer la matrice numérique ---
            # .notna().astype(int) transforme les NaN en 0 et les données présentes en 1
            matrice_couleurs = data[features_to_map].notna().astype(int)

            # --- Étape 2 : Appliquer la condition ---
            # On multiplie par 2 les lignes correspondant aux AVC (les 1 deviennent des 2)
            # Les 0 (données manquantes) restent à 0 car 0 * 2 = 0
            matrice_couleurs.loc[data["Neurol_cond"] == "AVC"] *= 2

            # (Les lignes BM qui sont à 1 restent à 1)

            # --- Étape 3 : Tracer le graphique avec Seaborn ---
            plt.figure(figsize=(5, 15))

            sns.heatmap(
                matrice_couleurs,
                cmap=ma_palette,
                cbar=False,  # Désactive la barre d'échelle sur le côté
                yticklabels=data["IPP"].to_list(),
                linewidths=0.5,  # Recrée le style "quadrillage" de missingno
                linecolor="lightgray",
            )

            # --- Étape 4 : Mise en forme et Sauvegarde ---
            plt.title(f"Données manquantes : {muscle_grp} ({side})", fontsize=12, pad=15)
            plt.yticks(fontsize=8)
            plt.xticks(fontsize=10, rotation=45, ha="right")

            # Utilisation de bbox_inches='tight' pour ne pas rogner le texte sur les bords
            plt.savefig(
                f"/Users/mathildetardif/Documents/Python/Biomarkers/amelio_medullo_n/results/data_exploration/missing_data/{muscle_grp}_{side}.png",
                bbox_inches="tight",
            )
            # TRÈS IMPORTANT : Fermer la figure courante pour libérer la mémoire.
            # Sinon, Python va superposer les 24 graphiques les uns sur les autres !
            plt.close()


if __name__ == "__main__":
    data_path = "/Users/mathildetardif/Documents/pre_post_data_final.xlsx"

    dict_mvt_BM = {
        "Artic_hip_flex": ["H_flex_ass", "H_flex_GT", "Sartorius", "Iliopsoas"],
        "Artic_hip_ext": ["H_ext_PP", "H_ext_GF", "Gmax"],
        "Artic_hip_abd": ["H_abd", "GM"],
        "Artic_hip_add": ["H_add", "Adductor"],
        "Artic_hip_rot_int": ["H_rot_int", "Gmin"],
        "Artic_hip_rot_ext": ["H_rot_ext"],
        "Knee_flex": ["K_flex", "SmTD", "Smbr", "Bic_Fem"],
        "Knee_ext": ["K_ext", "RF", "QF", "Gracilis"],
        "Ank_flex_90": ["A_dorsiflex_GF"],
        "Ank_flex_180": ["A_dorsiflex_GT", "TA"],
        "Ank_ext": ["A_plantarflex", "Gastroc", "Sol"],
        "A_eversion": ["Fibu_long"],
    }

    main(data_path, dict_mvt_BM)
