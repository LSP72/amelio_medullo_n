
class Stats:
    def __init__(self):
        pass
    
    @staticmethod
    def count_missing_values(df, column_name):
            """Counts the number of missing values in a specified column.

            Parameters
            ----------
            df : pd.DataFrame
                The original dataframe.
            column_name : str
                The name of the column to check for missing values.

            Returns
            -------
            int
                The number of missing values in the specified column.
            """
            missing_count = df[column_name].isna().sum()
            index_missings = df[column_name].index[df[column_name].isna()]
            print(f"Number of missing values in '{column_name}': {missing_count}")
            
            return index_missings