from openpyxl import load_workbook
import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from amelio_medullo import ProcessDataLokomat


def main():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    folder_path = filedialog.askdirectory()

    print("Selected folder:", folder_path)

    name_output_file = input("\nEnter the name of the output file: ")

    ProcessDataLokomat.read_data_info(folder_path, name_output_file)


if __name__ == "__main__":
    main()
