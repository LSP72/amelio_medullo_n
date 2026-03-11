from amelio_medullo import ProcessExcel


def main(file_path):
    data = ProcessExcel.read_excel(file_path)
    ProcessExcel.read_excel_sheets(data)


if __name__ == "__main__":
    file_path = "/Volumes/SP UFD U2/PhD/Stage Nantes/bilan_init_20260121 PEC opposés.xlsx"
    print(f"Processing file: {file_path}")
    main(file_path)
