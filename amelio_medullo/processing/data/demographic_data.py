import pandas as pd
import numpy as np

class DemographicData:
    def __init__(self, value):
        pass

    @staticmethod
    def clean_lesion_type(df, column_name):
        """Cleans the 'Trouble neuro' column by categorizing lesion types.

        Parameters
        ----------
        df : pd.DataFrame
            The original dataframe containing the 'Trouble neuro' column.
        column_name : str
            The name of the column to clean (e.g., 'Trouble neuro').

        Returns
        -------
        pd.DataFrame
            The dataframe with the cleaned lesion types.
        """

        condition = ~df[column_name].isin(['BM', 'AVC'])
        df.loc[condition, column_name] = 'Autre'
        return df
    
    def calculate_day_btwn_2_cols(df, col1, col2, new_col_name=None):
        """Calculates the number of days between two date columns.

        Parameters
        ----------
        df : pd.DataFrame
            The original dataframe.

        Returns
        -------
        pd.DataFrame
            The dataframe with a new column 'Days_btwn_injury_and_rehab' containing the calculated days.
        """

        # Make sure the date columns are in datetime format:
        df[col1] = pd.to_datetime(df[col1], errors='coerce')
        df[col2] = pd.to_datetime(df[col2], errors='coerce')

        if new_col_name is None:
            new_col_name = 'Days_btwn_' + col1 + '_&_' + col2
        # Calculate the number of days between the two dates:
        df[new_col_name] = np.abs((df[col2] - df[col1]).dt.days)

        print(df[[col1, col2, new_col_name]].head())

        return df