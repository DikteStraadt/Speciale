import csv
import json
import os
from datetime import datetime

def create_empty_report():

    path = "../report_not_completed.json"

    with open(path, "w", newline="") as file:
        json.dump({}, file)

    file.close()

def write_to_report(key, value):

    path = "../report_not_completed.json"
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

    path = "../report_not_completed.json"

    with open(path, "r") as file:
        string_data = f"""{file.read()}"""

    file.close()

    json_data = json.loads(string_data)
    current_filename = "../report_not_completed.json"
    new_filename = f"report (timeliness={json_data['timeliness']}, encoding={json_data['encoding']}, fetures={json_data['feature selection']} {json_data['timestamp']}.json"
    os.rename(current_filename, new_filename)