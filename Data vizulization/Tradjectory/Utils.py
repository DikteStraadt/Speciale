def convert_visitation_status(patients):
    i = 0
    for patient in patients:
        j = 0
        for visitation in patient:
            new_involvement_status = transform_status_for_tradjectory(visitation['involvementstatus'])
            patients[i][j]['involvementstatus'] = new_involvement_status
            j = j + 1
        i = i + 1

    return patients

def transform_status_for_tradjectory(value):
    if value == 0:  # No
        return 0
    elif value == 7:  # Obs
        return 1
    elif value == 1 or value == 4 or value == 2 or value == 5:  # Right side
        return 2
    elif value == 2 or value == 5:  # Left side
        return 3
    elif value == 3 or value == 6:  # Both sides
        return 4
