import csv
import os
from datetime import datetime

def create_empty_report():

    path = "report_not_completed.csv"

    with open(path, "w", newline="") as file:
        print("Empty report file created")

def write_to_report(data_to_append):

    path = "report_not_completed.csv"

    with open(path, mode="a", newline="") as file:
        writer = csv.writer(file)

        for string in data_to_append:
            writer.writerow([string])

def rename_report_file():

    current_filename = "report_not_completed.csv"
    new_filename = f"report ({datetime.now().strftime('%d-%m-%Y %H-%M-%S')}).csv"
    os.rename(current_filename, new_filename)