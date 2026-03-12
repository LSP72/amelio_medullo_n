from amelio_medullo import ProcessExcel, ProcessPatients, ProcessData


def main(file_path, lists_cols):
    data = ProcessExcel.read_excel(file_path)
    # ProcessExcel.read_excel_sheets(data)
    data_revised = ProcessData.collect_desired_col_all_sheets(data, lists_cols)
    print(data_revised.head())
    print(data_revised.columns)


if __name__ == "__main__":
    file_path = input ("Enter the path to the file: ")
    
    print(f"Processing file: {file_path}")
    lists_cols = {
    "bilan_init_birdlm": ['IEP', 'Date_Formulaire', 'Poids_dosage',
        'Taille_dosage', 'Contre_Indications', 'Cerebrolese',
        'BlesseMedullaire',],
    "bilan_init_6m": ['IEP', 'Age', 'Distance_parcourue',
        'Distance_theorique', 'Distance_theorique_lim_inf',
        'Distance_parcourue_tp']
    }
    main(file_path, lists_cols)
