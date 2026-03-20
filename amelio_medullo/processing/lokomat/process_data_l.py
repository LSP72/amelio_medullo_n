import pandas as pd
import numpy as np
from openpyxl import load_workbook
import os

class ProcessDataLokomat:
    def __init__(self):
        pass

    def read_data_info(folder_path, name_output_file):

        results = []

        for file in os.listdir(folder_path):
            if file.endswith(".xlsx"):
                file_path = os.path.join(folder_path, file)
                
                try:
                    wb = load_workbook(file_path, read_only=True)
                    sheet = wb[wb.sheetnames[0]]
                    
                    first_sheet_name = sheet.title
                    cell_value = sheet.cell(row=3, column=3).value
                    
                    results.append({
                        "file_name": file,
                        "sheet_name": first_sheet_name,
                        "start_date": cell_value
                    })
                
                except Exception as e:
                    results.append({
                        "file_name": file,
                        "sheet_name": "ERROR",
                        "start_date": str(e)
                    })

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Save to CSV
        output_path = os.path.join(folder_path, name_output_file, ".xlsx")
        df.to_excel(output_path, index=False)

        print(f"CSV saved at: {output_path}")

        return df