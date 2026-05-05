import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # On ajoute seaborn
from matplotlib.colors import ListedColormap  # Pour créer la palette de couleurs
from amelio_medullo import MuscleScore


def main(data_path, dict_mvt_BM):
    data = pd.read_excel(data_path)

    for side in ["right", "left"]:

        for features_to_map in dict_mvt_BM:
             
            for feature in features_to_map:
                if features_to_map in data.columns:
                    count_NaN = data[features_to_map].isna().sum()
            
            missing_percentages = (count_NaN / len(data)) * 100

            # 2. Combine into a clean DataFrame
            missing_df = pd.DataFrame({
                'Missing Count': count_NaN,
                'Percentage (%)': missing_percentages
            })

            # 3. Filter to only show columns that ACTUALLY have NaNs, and sort them
            missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)

            print(missing_df.head())

            plt.figure(figsize=(17, 8))

            # Plot the missing counts
            ax = sns.barplot(
                x=missing_df.index, 
                y='Missing Count', 
                data=missing_df, 
                palette='flare' # A nice warm color palette
            )

            # Rotate column names so they are readable
            plt.xticks(rotation=45, ha='right')
            plt.title('Count of Missing Values per Column', fontsize=14, pad=15)
            plt.ylabel('Number of NaNs', fontsize=12)
            plt.xlabel('Features', fontsize=12)

            # Optional: Add the exact numbers on top of each bar
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height()*100/89)}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='bottom', 
                            fontsize=10, color='black', xytext=(0, 3), 
                            textcoords='offset points')

            plt.tight_layout()
            plt.show()
            plt.save
        
            # TRÈS IMPORTANT : Fermer la figure courante pour libérer la mémoire.
            # Sinon, Python va superposer les 24 graphiques les uns sur les autres !
            plt.close()


if __name__ == "__main__":
    data_path = "/Users/mathildetardif/Documents/final_data_matrix.xlsx"

    dict_mvt_BM = [
        # Artic_hip_flex_D_pre, Artic_hip_flex_G_pre, Artic_hip_flex,
        ["H_Flex_ass_D_pre", "H_Flex_ass_G_pre", "H_Flex_ass"],
        # Artic_hip_ext_D_pre	Artic_hip_ext_G_pre, Artic_hip_ext
        ["H_Ext_PP_D_pre", "H_Ext_PP_G_pre", "H_Ext_PP"],
        # H_ext_GF_D_pre, H_ext_GF_G_pre
        # Knee_flex_D_pre	Knee_flex_G_pre,
        ["K_Flex_D_pre", "K_Flex_G_pre", "Knee_flex"],
        # Knee_ext_D_pre, Knee_ext_G_pre, 
        ["K_Ext_D_pre", "K_Ext_G_pre", "Knee_ext"],
        ["Ank_flex_180_D_pre"	"Ank_flex_180_G_pre", "Ank_flex_180"],
        # A_Dorsiflex_GT_D_pre,  A_Dorsiflex_GT_G_pre,  A_dorsiflex_GF_D_pre, A_dorsiflex_GF_G_pre,
        # A_Plantarflex_G_pre
        # H_abd_D_pre, H_abd_G_pre
        # H_add_D_pre, H_add_G_pre
        # H_rot_int_D_pre, H_rot_int_G_pre
        # A_Ever_G_pre
        # A_Inver_G_pre										
        # Artic_hip_add,	Artic_hip_abd
        # Artic_hip_rot_ext, Artic_hip_rot_int							
    ]
    {
        "Artic_hip_flex": ["H_flex_ass_D_pre", "H_flex_GT_D_pre", "H_flex_ass_G_pre", "H_flex_GT_G_pre"],
        "Artic_hip_ext": ["H_ext_PP_D_pre", "H_ext_GF_D_pre", "H_ext_PP_G_pre", "H_ext_GF_G_pre"],
        "Artic_hip_abd": ["H_abd_D_pre", "H_abd_G_pre"],
        "Artic_hip_add": ["H_add_D_pre", "H_add_G_pre"],
        "Artic_hip_rot_int": ["H_rot_int_D_pre", "H_rot_int_G_pre"],
        "Artic_hip_rot_ext": ["H_rot_ext_D_pre", "H_rot_ext_G_pre"],
        "Knee_flex": ["K_flex_D_pre", "K_flex_G_pre"],
        "Knee_ext": ["K_ext_D_pre", "K_ext_G_pre"],
        "Ank_flex_90": ["A_dorsiflex_GF_G_pre", "A_dorsiflex_GF_G_pre"],
        "Ank_flex_180": ["A_dorsiflex_GT_G_pre", "A_dorsiflex_GT_G_pre"],
        "Ank_ext": ["A_plantarflex_G_pre", "A_plantarflex_G_pre"],
    }

    main(data_path, dict_mvt_BM)
