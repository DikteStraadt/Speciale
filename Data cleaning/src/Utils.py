
def transform_status(value):

    if value == 0:  # No
        return 0
    elif value == 1 or value == 2 or value == 3 or value == 4 or value == 5 or value == 6:  # Yes
        return 1
    elif value == 7:  # Obs
        return 2