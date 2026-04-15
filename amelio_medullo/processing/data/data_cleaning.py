import pandas as pd

class DataCleaning:
    def __init__(self):
        pass
    
    @staticmethod
    def select_patients_with_complete_data(df, keys):
        """Selects patients with complete data in the specified columns.
        
        Parameters
        ----------
        df : pd.DataFrame
            The original dataframe containing patient data.
        keys : list of str
            The list of column names to check for complete data.
        
        Returns
        -------
        pd.DataFrame
            A dataframe containing only patients with complete data in the specified columns.
        """

        return df[df['tests'].isin(keys)]