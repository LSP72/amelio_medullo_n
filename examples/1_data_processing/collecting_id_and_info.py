import time
import pandas as pd
from amelio_medullo import Calculus

def main(data_path, features_to_keep, output_path=None):
    data = pd.read_excel(data_path)
    mcids = Calculus.calculate_MCID_2(data, default_threshold=45)

    all_data = data.merge(mcids, on="IPP", how="left")

    if output_path:
        output_name = f"{output_path}/id_patients_with_classifications_{time.strftime('%Y-%m-%d_%H-%M-%S')}.xlsx"
        pd.DataFrame(all_data[features_to_keep]).to_excel(output_name)

    print(all_data.to_markdown())


if __name__ == "__main__":
    data_path = "/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/CHUNantes collaboration/donnees/data_from_dpi/final_data_matrix_sessions_separated.xlsx"
    output_path = "/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/CHUNantes collaboration/donnees/data_from_dpi/"
    features_to_keep = ["IPP", "Age", "Sex", "functional_level", "Neurol_cond", "6MWT_m_pre", "6MWT_m_post", "MCID_classes"]
    main(data_path, features_to_keep, output_path)