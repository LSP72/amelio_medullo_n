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
    def lesion_level_to_num(row):
        dict_lesions = {
            "Brain": 0,
            "C1": 1,
            "C2": 2,
            "C3": 3,
            "C4": 4,
            "C5": 5,
            "C6": 6,
            "C7": 7,
            "C8": 8,
            "T1": 9,
            "T2": 10,
            "T3": 11,
            "T4": 12,
            "T5": 13,
            "T6": 14,
            "T7": 15,
            "T8": 16,
            "T9": 17,
            "T10": 18,
            "T11": 19,
            "T12": 20,
            "L1": 21,
            "L2": 22,
            "L3": 23,
            "L4": 24,
            "L5": 25,
            "S1": 26,
            "S2": 27,
            "S3": 28,
            "S4": 29,
            "S5": 30,
        }
        if row["Lesion"] in dict_lesions:
            row["Lesion_num"] = dict_lesions[row["Lesion"]]
        else:
            row["Lesion_num"] = None

        return row

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
