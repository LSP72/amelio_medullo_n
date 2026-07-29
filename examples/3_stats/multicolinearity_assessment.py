"""
This script computes the Variance Inflation Factor (VIF) for a given dataset to assess multicollinearity among the features.
The VIF quantifies how much the variance of a regression coefficient is inflated due to multicollinearity with other features.
A VIF value greater than 10 is often considered indicative of high

Returns
-------
_type_
    _description_
"""

import pandas as pd
import numpy as np
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor


def compute_vif(X):
    # Only numeric columns are considered for VIF calculation. Non-numeric columns are ignored.
    X_num = X.select_dtypes(include='number').dropna()
    Xc = add_constant(X_num)  
    vif = pd.DataFrame({
        'feature': Xc.columns,
        'VIF': [variance_inflation_factor(Xc.values, i)
                for i in range(Xc.shape[1])]
    })
    return vif[vif['feature'] != 'const'].sort_values('VIF', ascending=False)

def main(data_path, cols_to_keep):
    # Load data
    df = pd.read_excel(data_path) 
    X = df[cols_to_keep].copy()
    X = X.replace([np.inf, -np.inf], np.nan)  # Replace inf values with NaN
    X = X.dropna()  # Drop any remaining NaN values

    # Calculer le VIF
    vif_df = compute_vif(X)
    print(vif_df)

if __name__ == "__main__":
    DATA_PATH = "/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/CHUNantes collaboration/donnees/data_from_dpi/final_data_matrix_sessions_separated.xlsx"
    # DATA_PATH = "/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/CHUNantes collaboration/donnees/data_from_dpi/merged_data_final.xlsx"

    FEATURES = [
        "Lesion_num", "Nb sessions", "Sex", "BMI",
        "6MWT_m_pre", "delay_loko", "functional_level", "speed",
    ]
    # FEATURES = [
    #         "duration", "Durée_min", "Vitesse_kmh_MOY", "BWS_%_MOY",
    #         "step_length", "Guidage_%_MOY", "sessions_per_week",
    #         "Neurol_cond", "Sex", "Nb sessions", "BMI"
    #     ]

    main(DATA_PATH, FEATURES)
