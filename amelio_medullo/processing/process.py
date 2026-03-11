import pandas as pd
import numpy as np


class ProcessExcel:
    def __init__(self):
        pass

    @staticmethod
    def read_excel(file_path):
        try:
            data = pd.read_excel(file_path, sheet_name=None)  # Read all sheets into a dictionary
            return data

        except Exception as e:
            print(f"Error reading Excel file: {e}")
            return None

    @staticmethod
    def read_excel_sheets(data):
        try:
            for name, df in data.items():
                print("Sheet:", name)
                print(df.columns)

        except Exception as e:
            print(f"Error reading Excel sheet: {e}")
            return None


class ProcessPatients:
    @staticmethod
    def read_patients(data):
        dict_patients = {}
        for name, df in data.items():
            dict_patients[name] = df["IEP"].tolist()
        return dict_patients

    @staticmethod
    def check_patients(dict_patients, eval1=None, eval2=None):
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
    def check_patients_all_sheets(dict_patients):
        lists = list(dict_patients.values())
        common_patients = set(lists[0])  # remove doubles in the first list

        for patients in lists[1:]:
            common_patients &= set(patients)

        return list(common_patients)

    @staticmethod
    def check_patients_in_main_list(dict_patients, main_list):
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
    def check_patients_in_pre_and_post_evaluations(pre_eval, post_eval):
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
