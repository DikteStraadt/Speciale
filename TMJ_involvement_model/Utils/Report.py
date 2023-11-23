import csv
import json
import os
from datetime import datetime

def create_empty_report():

    path = "report_not_completed.json"

    with open(path, "w", newline="") as file:
        json.dump({}, file)

    file.close()

def write_to_report(key, value):

    path = "report_not_completed.json"
    data_to_append = {key: value}

    try:
        with open(path, 'r') as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = {}

    existing_data.update(data_to_append)

    with open(path, 'w') as file:
        json.dump(existing_data, file, indent=4)

    file.close()


def rename_report_file():

    path = "report_not_completed.json"

    with open(path, "r") as file:
        string_data = f"""{file.read()}"""

    file.close()

    json_data = json.loads(string_data)
    new_filename = f"Results/{json_data['id']} report (categories={json_data['n_categories']}, slicing={str(json_data['time slice'][0])} from {str(json_data['time slice'][1])} to {str(json_data['time slice'][2])} years, encoding={json_data['encoding']}, features={json_data['feature selection']} {json_data['timestamp end']}.json"
    os.rename(path, new_filename)
    os.rename(f"Temp/{json_data['id']} data.xlsx", f"Results/{json_data['id']} data.xlsx")

def read_report():

    path = "report_not_completed.json"

    try:
        with open(path, 'r') as file:
            existing_data = json.load(file)
            file.close()
            return existing_data
    except FileNotFoundError:
        existing_data = {}
        file.close()
        return existing_data