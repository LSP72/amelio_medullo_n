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
            The list of tests that have been done (i.e., 'TOUT' for patients with all tests were assessed,
                                                         '6MWT' for patients with only the 6MWT was assessed,
                                                          etc.).

        Returns
        -------
        pd.DataFrame
            A dataframe containing only patients with complete data in the specified columns.
        """
        df = df[df["tests"].isin(keys)]
        df = df[df["PRE_POST_BOTH"] == "BOTH"]
        return df

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

        condition = ~df[column_name].isin(["BM", "AVC"])
        df.loc[condition, column_name] = "Autre"
        df.drop(["Commentaire sur TN"], axis=1, inplace=True)

        return df

    @staticmethod
    def fix_value_of_test(df, tests, condition_col, new_value):
        """Fixes the value of a test based on a condition in another column.

        Parameters
        ----------
        df : pd.DataFrame
            The original dataframe containing patient data.
        tests : list of str
            The list of tests to which values need to be fixed (e.g., ['6MWT_m_post_post', '10MWT_pas_post_post', '10MWT_sec_post_post']).
        condition_col : str
            The name of the column to check for the condition (e.g., 'Perim_marche_m_post_post').
        new_value : int or str
            The new value to assign to the test if the condition is met (e.g., 0).

        Returns
        -------
        pd.DataFrame
            The dataframe with the fixed values for the specified tests.
        """

        condition_perim = (df[condition_col] == 0) | (df[condition_col] == "0?")

        for test in tests:
            row_conditionned = df[test].isnull() & condition_perim
            df.loc[row_conditionned, test] = new_value

        print(df[[condition_col] + tests].head())

        return df

    @staticmethod
    def clean_string(col):
        """Cleans a string by stripping whitespace and converting to lowercase.

        Parameters
        ----------
        col : pd.Series
            The col to clean.

        Returns
        -------
        pd.Series
            The cleaned col.
        """

        return col.replace("0?", 0)

