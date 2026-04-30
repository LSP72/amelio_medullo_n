import numpy as np
import pandas as pd


class MuscleScore:
    def __init__(self, value):
        pass

    @staticmethod
    def _clean_muscle_score(val):
        """Converts mixed scores (ints, '3+', '2-', 'NaN') into comparable floats.

        Parameters
        ----------
        val : int, float, or str
            The original muscle score value.

        Returns
        -------
        float
            A cleaned float value (e.g., 3.25 for '3+', 2.75 for '3-', np.nan for 'NaN').
        """
        # 1. Handle actual Pandas/Numpy NaNs
        if pd.isna(val):
            return np.nan

        # 2. If already a clean number, returns it as a float
        if isinstance(val, (int, float)):
            return float(val)

        # 3. Handle strings
        val = str(val).strip()

        # 4. Catch string 'NaN'
        if val.lower() == "nan" or val == "":
            return np.nan

        # 5. Handle values with '+' or '-' signs
        if val.endswith("+"):
            return float(val[:-1]) + 0.25
        elif val.endswith("-"):
            return float(val[:-1]) - 0.25
        else:
            # Catch if val formatted as strings (e.g., '2')
            try:
                return float(val)
            except ValueError:
                return np.nan

    @staticmethod
    def convert_muscles_to_movements(df, mapping_dict):
        """Calculates the max score for movements based on collaborative muscles.

        Parameters
        ----------
        df : pd.DataFrame
            The original dataframe containing muscle scores.
        mapping_dict : dict
            A dictionary mapping movement names to lists of muscle column names.

        Returns
        -------
        pd.DataFrame
            A new dataframe with movement scores calculated as the max of the relevant muscle scores.
        """

        df_clean = df.copy()

        ## STEP 1: Convert SCI's scores to match the AVC's ones.

        # New dataframe w/ only the final movement scores
        df_movements = pd.DataFrame(index=df.index)

        # Unique list of all muscles
        all_muscles = set(muscle for muscles in mapping_dict.values() for muscle in muscles)

        # Apply the cleaning function only to the relevant muscle columns
        for col in all_muscles:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(MuscleScore._clean_muscle_score)

        # Find the max for each movement
        for movement, muscles in mapping_dict.items():
            # Check if muscles are indeed in df_clean
            valid_muscles = [m for m in muscles if m in df_clean.columns]

            if valid_muscles:
                df_movements[movement] = df_clean[valid_muscles].max(axis=1)
            else:
                # If none of the muscles for this movement exist in the df, return NaN
                df_movements[movement] = np.nan

        ## STEP 2: Add the scores of the AVC patients.
        # Checking the cols that are in the original df and in the new movement df.
        existing_movements = [col for col in df_movements.columns if col in df.columns]

        final_movement_df = df[existing_movements].combine_first(df_movements)

        final_movement_df.insert(0, "IPP", df["IPP"])  # Add the 'Patient' column back to the final dataframe

        return final_movement_df

    # TODO: fix the suffix changing the name of the columns in the original df, if needed
    @staticmethod
    def transform_dict_to_side(mapping_dict, side, keys=True):
        """
        Transforms the mapping dictionary to match the name of the dataframe columns which includes muscleside.

        Parameters
        ----------
        mapping_dict : dict
            Dictionary mapping the movement and their associated muscles.
        side : str
            The side of the body ('left' (G) or 'right' (D)) to include.

        Returns
        -------
        dict
            A new dictionary with only the muscles on the specified side.
        """
        if side == "right":
            suffix = "_D_pre"
        elif side == "left":
            suffix = "_G_pre"
        else:
            raise ValueError("Side must be 'left' or 'right'")

        transformed_dict = {}
        for movement, muscles in mapping_dict.items():
            if keys == True:
                transformed_dict[movement + suffix] = [muscle + suffix for muscle in muscles]
            else:
                transformed_dict[movement] = [muscle + suffix for muscle in muscles]
        return transformed_dict

