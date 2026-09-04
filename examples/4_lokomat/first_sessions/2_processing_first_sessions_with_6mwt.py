import pandas as pd
import numpy as np
from amelio_medullo import Calculus

"""
    This script merges the Lokomat data with the main patient data and calculated MCID values.
    The Lokomat data is a dataframe containing the results of the linear regressions over the first
    n sessions of Lokomat training for each patient.
"""

def load_data(data_path):
    return pd.read_excel(data_path)

def adjust_ipp_multiple_patients(data, loko_data):
    """This function enables to adjust the IPPs from the loko_data files to match
    the ones from the data one in a specific case.
       This functions is only usable when a patient had followed several blocks of Lokomat.
    In that specific case, their IPP has been changed in the data file (four zeros and the
    no of the block have been added to the original IPP), while the IPP in the loko_data
    file is the original one, the column "Block" gives the no of the block followed by the patient.


    Parameters
    ----------
    data : dataframe
        Dataframe containing the main dataset about patients (i.e., IPP (adjusted if needed),
        Neurol cond, 6MWT pre and post, and MCID classes).
    loko_data : dataframe
        Dataframe containing the Lokomat data for each patient for their first sessions.
    """
    
    for index, row in loko_data.iterrows():
        ipp = row["ID"]
        print(f"ID: {ipp}")
        block = row["Block"]
        print(f"Block: {block}")
        if block > 1:
            new_ipp = int(str(ipp) + "0000" + str(block))
            print(f"Block > 1, updating ID for {ipp} to new ID: {new_ipp}")
            loko_data.loc[index, "ID"] = new_ipp

    return loko_data

def main(loko_path, data_path, other_mcid_threshold, output_path=None):
    # 1 - load data
    loko_data = load_data(loko_path)
    data = load_data(data_path)

    loko_data_adjusted = adjust_ipp_multiple_patients(data, loko_data)

    mcids = Calculus.calculate_MCID_2(data, default_threshold=other_mcid_threshold)

    complete_data = pd.merge(loko_data_adjusted, mcids, left_on="ID", right_on="IPP", how="left")

    if output_path:
        complete_data.to_excel(output_path, index=False)
        print(f"Complete data saved to {output_path}")

    print(complete_data.to_markdown())


#%% ===== MAIN =====

if __name__ == "__main__":
    loko_path = "results/loko_results/fits_over_first_5_sessions.xlsx"
    data_path = "/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/CHUNantes collaboration/donnees/data_from_dpi/id_patients_with_classifications.xlsx"
    output_path = "results/complete_data_with_mcids.xlsx"

    main(loko_path, data_path, other_mcid_threshold=45, output_path=output_path)