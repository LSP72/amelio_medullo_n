from amelio_medullo import ProcessExcel, ProcessPatients, ProcessData


def read_collect_excel(file_path):
    data = ProcessExcel.read_excel(file_path=file_path)
    lists_cols = ProcessExcel.read_excel_sheets(data=data, show=False)

    return data, lists_cols


def collect_data(data, first_sheet, intervention):
    dict_patients = ProcessPatients.read_patients(data=data)
    check_patients = ProcessPatients.check_patients_in_main_list(dict_patients=dict_patients, main_list=first_sheet)

    if check_patients == True:
        list_patients_intervention = ProcessPatients.collect_patients_for_chosen_intervention(
            data=data, main_info=first_sheet, intervention=intervention
        )
        print(
            f"{intervention} patients ({len(list_patients_intervention.to_list())}): {sorted(list_patients_intervention.to_list())}\n"
        )
        selected_data = ProcessPatients.select_patients(data=data, patients=list_patients_intervention)
        return selected_data
    else:
        raise ValueError(
            "Some patients in the sub-sheets are not in the main sheet.\nThus no demographic data are available."
        )


def create_csv(data, lists_cols, output_path):
    data_for_csv = ProcessData.collect_desired_col_all_sheets(data=data, lists_col=lists_cols)
    print(f"Data looks like: {data_for_csv.head()}")
    file_output_path = output_path + first_sheet[: len(first_sheet) - 7] + ".csv"
    data_for_csv.to_csv(file_output_path)
    print(f"Data saved: {file_output_path}")
    return data_for_csv


def main(file_path, first_sheet, intervention, desired_lists_cols, output_path):
    data, lists_cols = read_collect_excel(file_path)
    selected_data = collect_data(data, first_sheet, intervention)
    create_csv(selected_data, desired_lists_cols, output_path)


if __name__ == "__main__":
    
    output_path = input("Enter the path to the output folder: ")
    intervention = "LOKOMAT"  # /!\ CAPITAL LETTERS
    status = input("Which dataset? init or final: ")
    if status == "init":
        file_path = input("Enter the path to the file: ")
        print(f"Processing file: {file_path}")
        first_sheet = "bilan_init_birdlm"  # bilan init
        desired_lists_cols = {
            "bilan_init_birdlm": [
                "IEP",
                "IPP",
                "Date_Formulaire",
                "Poids_dosage",
                "Taille_dosage",
                "Cerebrolese",
                "BlesseMedullaire",
            ],
            "bilan_init_6m": [
                "IPP",
                "Age",
                "Distance_parcourue",
            ],
            "bilan_init_10m": [
                "IPP",
                "Nombre_de_pas",
                "Temps",
            ],
            "bilan_init_berg": [
                "IPP",
                "Score_total",
            ],
            "bilan_init_tuagt": ["IPP", "Temps_1"],
                "IPP",
                "Temps_1"
            ]
        }
    elif status == "final":
        file_path = input("Enter the path to the file: ")
        print(f"Processing file: {file_path}")
        first_sheet = "bilan_final_bfrdlm"  # bilan final
        desired_lists_cols = {
            "bilan_final_birdlm": [
                "IEP",
                "IPP",
                "Date_Formulaire",
                "Poids_dosage",
                "Taille_dosage",
                "Cerebrolese",
                "BlesseMedullaire",
            ],
            "bilan_final_6m": [
                "IPP",
                "Age",
                "Distance_parcourue",
            ],
            "bilan_final_10m": [
                "IPP",
                "Nombre_de_pas",
                "Temps",
            ],
            "bilan_final_berg": [
                "IPP",
                "score_total",
            ],
            "bilan_final_tuagt": ["IPP", "Temps_1"],
                "IPP",
                "Temps_1"
            ]
        }
    else:
        print("Choose 'init' or 'final'.")
    main(file_path, first_sheet, intervention, desired_lists_cols, output_path)
