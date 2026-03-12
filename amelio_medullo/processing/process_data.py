import pandas as pd
import numpy as np

class ProcessData:
    def __init__(self):
        pass

    @staticmethod
    def _collect_desired_col(df: pd.DataFrame, list_col: list) -> pd.DataFrame:
        """Function to collect the desired columns of the data from one sheet.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing the data of ONE sheet.
        list_col : list
            List of the desired columns.

        Returns
        -------
        pd.DataFrame
            Dataframe containing only the desired columns.
        """
        try:
            df_col = df[list_col]
            return df_col

        except Exception as e:
            print(f"Error collecting desired columns: {e}")
            return None
        
    @staticmethod
    def collect_desired_col_all_sheets(data: dict, lists_col: dict) -> dict:
        """Function to collect the desired columns of the data from all sheets.

        Parameters
        ----------
        data : dict
            Dictionary containing the data, separated by sheets of the excel file.
        lists_col : dict
            Dictionary containing the desired columns for each sheet.

        Returns
        -------
        dict
            Dictionary containing only the desired columns for all sheets.
        """
        data_revised = None
        for name, lists in lists_col.items():
            print("Processing sheet:", name)
            df_revised = ProcessData._collect_desired_col(data[name], lists)
            if data_revised is None:
                data_revised = df_revised.copy()
            else:
                data_revised = data_revised.merge(df_revised, on="IEP", how="outer")

        return data_revised