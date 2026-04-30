import pandas as pd
import numpy as np


class DemographicData:
    def __init__(self, value):
        pass

    @staticmethod
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
        df[col1] = pd.to_datetime(df[col1], errors="coerce")
        df[col2] = pd.to_datetime(df[col2], errors="coerce")

        if new_col_name is None:
            new_col_name = "Days_btwn_" + col1 + "_&_" + col2
        # Calculate the number of days between the two dates:
        df[new_col_name] = np.abs((df[col2] - df[col1]).dt.days)

        print(df[[col1, col2, new_col_name]].head())

        return df
    

class FunctionalLevel:
    def __init__(self):
        pass

    @staticmethod
    def _BM_cat(score: int):
        if pd.isna(score):
            return None

        if score <= 2:
            return 0
        elif score == 3:
            return 2
        elif score > 3 and score <= 7:
            return 3
        elif score == 8:
            return 4
        else:
            raise ValueError(f"Classification could not be done.")

    @staticmethod
    def _AVC_cat(score):
        if pd.isna(score):
            return None
        if score <= 2:
            return 0
        elif score == 3:
            return 2
        elif score == 4:
            return 3
        elif score == 5:
            return 4
        else:
            raise ValueError(f"Classification could not be done.")

    @staticmethod
    def _categorise(row):
        if row["Neurol_cond"] == "BM":
            funct = row["SCIM-12"]
            return FunctionalLevel._BM_cat(funct)

        elif row["Neurol_cond"] in ["AVC", "Autre"]:
            funct = row["FAC"]
            return FunctionalLevel._AVC_cat(funct)

        else:
            return None

    @staticmethod
    def functional_categories(data):
        """ "
        This function takes the data excel and from the demographic sheet it derives the functional classification
        Class 0: FAC 0-2 / SCIM 0-2
        Class 2: FAC 3 / SCIM 3
        Class 3: FAC 4 / SCIM 4-7
        Class 4: FAC 5 / SCIM 8
        """
        return data.apply(FunctionalLevel.functional_categories, axis=1)
