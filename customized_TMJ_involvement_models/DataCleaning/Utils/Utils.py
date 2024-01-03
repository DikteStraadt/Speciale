
def transform_visitation_status(value):

    if value == 0:  # No
        return 0
    elif value == 1 or value == 2 or value == 3 or value == 4 or value == 5 or value == 6 or value == 7:  # Yes or obs
        return 1
