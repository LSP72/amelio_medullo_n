import pandas as pd
import numpy as np


class ProcessExcel:
    def __init__(self):
        pass

    @staticmethod
    def read_excel(file_path: str) -> dict:
        """Function to read and return the data of the excel.

        Parameters
        ----------
        file_path : str
            Path to the excel file.

        Returns
        -------
        dict
            Dictionary containing the data, separated by sheets of the excel file.
        """
        try:
            data = pd.read_excel(file_path, sheet_name=None)  # Read all sheets into a dictionary
            return data

        except Exception as e:
            print(f"Error reading Excel file: {e}")
            return None

    @staticmethod
    def read_excel_sheets(data: dict) -> dict:
        """fonction to read the title of all excel sheets.

        Parameters
        ----------
        data : dict
            Dictionary containing the data, separated by sheets of the excel file.

        Returns
        -------
        dict
            Dictionary containing the titles of the sheets.
        """
        try:
            for name, df in data.items():
                print("Sheet:", name)
                print(df.columns)
            return data.items()

        except Exception as e:
            print(f"Error reading Excel sheet: {e}")
            return None


class ProcessPatients:
    def __init__(self):
        pass

    @staticmethod
    def read_patients(data: dict) -> dict:
        """Function to collect all patients in all sheets.

        Parameters
        ----------
        data : dict
            _description_

        Returns
        -------
        dict
            Dictionary containing the patients in each sheet.
        """
        dict_patients = {}
        for name, df in data.items():
            dict_patients[name] = df["IEP"].tolist()
        return dict_patients

    @staticmethod
    def check_patients(dict_patients: dict, eval1: str , eval2: str ) -> list:
        """Function to check if patients of eval1 are in the eval2.

        Parameters
        ----------
        dict_patients : dict
            Dictionary containing the patients in each sheet.
        eval1 : str
            Name of the sheet of the first evaluation. 
        eval2 : str
            Name of the sheet of the second evaluation.

        Returns
        -------
        list
            List of patients that are in both evaluations.

        Raises
        ------
        ValueError
            If neither eval1 nor eval2 are provided.
        """
        if eval1 is None or eval2 is None:
            raise ValueError("eval1 and eval2 must be provided.")

        patients1 = dict_patients.get(eval1, [])
        patients2 = dict_patients.get(eval2, [])
        common = set(patients1) & set(patients2)
        if common:
            print(f"Patients in common between {eval1} and {eval2}: {common}")
            return list(common)
        else:
            print(f"No common patients between {eval1} and {eval2}.")
            return []

    @staticmethod
    def check_patients_all_sheets(dict_patients: dict) -> list:
        """Function that returns the list of patients that are in all sheets of the excel file.

        Parameters
        ----------
        dict_patients : dict
            Dictionary containing the patients per sheet.

        Returns
        -------
        list
            List of patients that are in all sheets.
        """
        lists = list(dict_patients.values())
        common_patients = set(lists[0])  # remove doubles in the first list

        for patients in lists[1:]:
            common_patients &= set(patients)

        return list(common_patients)

    @staticmethod
    def check_patients_in_main_list(dict_patients: dict, main_list: str) -> bool:
        """_summary_

        Parameters
        ----------
        dict_patients : dict
            Dictionary containing the patients per sheet.
        main_list : str
            Name of the main list.

        Returns
        -------
        bool
            True if all patients in the sub-lists are also in the main list, False otherwise.
        """
        main_set = set(dict_patients[main_list])  # remove doubles in the main list
        all_evaluations_valid = True  # will be set to False if any patient is missing in the main list

        for name, patients in dict_patients.items():
            missing = [p for p in patients if p not in main_set]

            if missing:
                all_evaluations_valid = False
                print(f"{name} - Patients not in the main list: {missing}")

        if all_evaluations_valid:
            print("All patients in the sub-lists are also in the main list.")
            return all_evaluations_valid
        else:
            return all_evaluations_valid

    @staticmethod
    def check_patients_in_pre_and_post_evaluations(pre_eval: pd.DataFrame, post_eval: pd.DataFrame) -> list:
        """Function that checks if patients in pre-evaluation are in post-evaluation.

        Parameters
        ----------
        pre_eval : pd.DataFrame
            DataFrame containing the pre-evaluation patients.
        post_eval : pd.DataFrame
            DataFrame containing the post-evaluation patients.

        Returns
        -------
        list
            List of patients that are in the pre-evaluation but not in the post-evaluation.
        """
        pre_eval_np = pre_eval.to_numpy()
        post_eval_np = post_eval.to_numpy()
        if len(pre_eval_np) != len(post_eval_np):
            print(
                f"Number of patients in pre_evaluation ({len(pre_eval_np)}) and post_evaluation ({len(post_eval_np)}) do not match."
            )

        # TODO: check if need to remove doubles in post_eval
        # post_eval_set = set(post_eval_np)
        missings = [a for a in pre_eval_np if a not in post_eval_np]
        if missings:
            print(f"Patients {missings} \nnot in post_evaluation.")
        return missings
