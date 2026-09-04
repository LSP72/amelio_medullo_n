import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

"""
    This script processes the Lokomat data to perform linear regression analyses on specified features
        over the first n sessions for each patient. It allows for splitting patient sessions into blocks
        and saves the results to an Excel file.

"""

#%% ===== Auxiliary functions =====
def loading_data(data_path):
    all_reports = pd.read_excel(data_path)
    return all_reports

def _split_into_blocks(group_sorted, block_sizes):
    """Split a patient's sorted sessions into consecutive blocks of the given sizes.

    Any leftover sessions past the last declared block size are kept as one
    final block, so a patient with 34 sessions and block_sizes=[20, 14] yields
    exactly two blocks, while block_sizes=[20] on the same patient yields a
    [20, 14] split too (the second block is inferred).
    """
    blocks = []
    start = 0

    for size in block_sizes:
        blocks.append(group_sorted.iloc[start : start + size])
        start += size
    if start < len(group_sorted):
        blocks.append(group_sorted.iloc[start:])
    return blocks

def _add_to_dict(ID, feature, block_results, results_dict):
    if ID not in results_dict:
        results_dict[ID] = {}
    results_dict[ID][feature] = block_results

def fit_stat(blocks):
    block_results = []

    for i, (sessions, values) in enumerate(blocks):
        X = np.array(sessions)
        y = np.array(values)
        n = len(y)

        fit = sm.OLS(y, sm.add_constant(X)).fit()
        slope, se_slope = fit.params[1], fit.bse[1]
        p_slope = fit.pvalues[1]
        ci_low, ci_high = fit.conf_int()[1]
        r2 = fit.rsquared
        intercept = fit.params[0]
        y_pred = fit.fittedvalues

        block_results.append({
            "block": i + 1, "n": n,
            "slope": slope, "slope_se": se_slope,
            "slope_ci_low": ci_low, "slope_ci_high": ci_high,
            "slope_p": p_slope,
            "intercept": intercept, "r2": r2,
            "y_pred": y_pred
        })

    return block_results
    
def save_results_to_excel(results_dict, output_path):
    rows = []
    for ID, features in results_dict.items():
        for feature, block_results in features.items():
            for block in block_results:
                rows.append(
                    {
                        "ID": ID,
                        "Feature": feature,
                        "Block": block["block"],
                        "Slope": block["slope"],
                        "Slope SE": block["slope_se"],
                        "Intercept": block["intercept"],
                        "R2": block["r2"],
                        "Nb of sessions": block["n"],
                        "Predictions": block["y_pred"]
                    }
                )

    results_df = pd.DataFrame(rows)
    results_df.to_excel(output_path, index=False)

#%% ===== Main function =====
def main(data_path, feature_list, output_dir, id_col='ID', patient_blocks=None, nb_sessions=None):
    all_reports = loading_data(data_path)

    patient_blocks = patient_blocks or {}
    results_dict = {}
    for feature in feature_list:
        for patient_id, subreport in all_reports.groupby(id_col):
            subreport_sorted = subreport[["Session(s)", feature]].dropna().sort_values("Session(s)")

            if len(subreport_sorted) < 2:
                print(f"Skipping patient {patient_id}: not enough sessions with valid '{feature}' data.")
                continue

            block_sizes = patient_blocks.get(patient_id)
            chunks = _split_into_blocks(subreport_sorted, block_sizes) if block_sizes else [subreport_sorted]

            blocks = []
            for i, chunk in enumerate(chunks):
                if len(chunk) < 2:
                    print(f"Skipping block {i + 1} for patient {patient_id} ({feature}): fewer than 2 sessions.")
                    continue
                if nb_sessions is not None:
                    chunk = chunk[0:nb_sessions]
                blocks.append((chunk["Session(s)"], chunk[feature]))

            if not blocks:
                continue

            block_results = fit_stat(blocks)

            _add_to_dict(patient_id, feature, block_results, results_dict)

    save_results_to_excel(results_dict, os.path.join(output_dir, f"fits_over_first_{nb_sessions}_sessions.xlsx"))


#%% ===== MAIN =====

if __name__ == "__main__":
    data_path = "/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/CHUNantes collaboration/donnees/lokomat_reports/all_reports.xlsx"

    feature_list = ["Vitesse_kmh_MOY", "BWS_%_MOY", "Guidage_G_%_MOY", "Guidage_D_%_MOY"]
    output_dir = "results/loko_results/"

    # Patients whose sessions should be split into separate regression blocks,
    # e.g. before/after a change in protocol. Sizes are consecutive and in
    # session order; any leftover sessions form a final block automatically.
    patient_blocks = {
        5750370: [19, 20],
        20047255: [20, 19],
        24190250: [16, 18],
        25801189: [17, 20],
        27522095: [20, 8, 19, 20, 22, 8],
        28373638: [5, 13, 5, 4, 6],
        30312319: [3, 13, 24, 17],
        30528453: [21, 19],
        31022187: [20, 20],
        32548837: [21, 20]
    }
    
    main(data_path, feature_list, output_dir, patient_blocks=patient_blocks, nb_sessions=5)

