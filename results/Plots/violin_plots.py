import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Vos données (une seule variable) ---
# Remplacer par votre vraie colonne de DataFrame, ex: df['Ma_Colonne']
data_path = "/Volumes/SP UFD U2/PhD/Stage Nantes/LOKOMAT/loko_final_table_sessions_separated.xlsx"
data = pd.read_excel(data_path)
cols_to_keep = ["nb_sessions",	"duration",	"Distance_m",	"Distance_pas",	"Durée_min",	"Vitesse_kmh_MIN",	"Vitesse_kmh_MAX",	"Vitesse_kmh_MOY",	"BWS_%_MIN",	"BWS_%_MAX",
                         "BWS_%_MOY",	"BWS_kg_MIN",	"BWS_kg_MAX",	"BWS_kg_MOY",	"Guidage_G_%_MIN",	"Guidage_G_%_MAX",	"Guidage_G_%_MOY",	"Guidage_D_%_MIN",	"Guidage_D_%_MAX",
                         "Guidage_D_%_MOY",	"sessions_per_week",	"6MWT_m_pre", "6MWT_m_post", "MCID_classes",	"functional_level"]
    
data = data[cols_to_keep]

# --- 2. Configuration esthétique ---
sns.set_theme(style="whitegrid")
plt.figure(figsize=(6, 8)) 


for col in cols_to_keep:
    sns.violinplot(
        y=data[col], 
        color="skyblue",     
        inner="quartile",    
        linewidth=1.5
    )

    sns.stripplot(
        y=data[col],
        color="darkblue",  
        size=5,            
        jitter=True,       
        alpha=0.7,         
        zorder=1           
    )

    plt.title(f"Distribution of {col}", fontsize=14)
    plt.ylabel("Value", fontsize=12)
    plt.savefig(f"results/Plots/violin_plot_{col}.png", dpi=300)
    plt.close()
    # --- 5. Affichage ---
    # plt.show()