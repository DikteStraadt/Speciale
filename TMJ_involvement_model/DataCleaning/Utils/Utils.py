
def transform_visitation_status(value, number_of_categories):

    if number_of_categories == 2:
        if value == 0:  # No
            return 0
        elif value == 1 or value == 2 or value == 3 or value == 4 or value == 5 or value == 6 or value == 7:  # Yes or obs
            return 1
    if number_of_categories == 3:
        if value == 0:  # No
            return 0
        elif value == 1 or value == 2 or value == 3 or value == 4 or value == 5 or value == 6:  # Yes
            return 1
        elif value == 7:  # Obs
            return 2
    elif number_of_categories == 4:
        if value == 0:  # No
            return 0
        elif value == 7:  # Obs
            return 1
        elif value == 1 or value == 2 or value == 3:  # TMJ arthritis (left, right and both)
            return 2
        elif value == 4 or value == 5 or value == 6:  # TMJ chronic (left, right and both)
            return 3
    elif number_of_categories == 5:
        if value == 0:  # No
            return 0
        elif value == 7:  # Obs
            return 1
        elif value == 1 or value == 4:  # Right side
            return 2
        elif value == 2 or value == 5:  # Left side
            return 3
        elif value == 3 or value == 6:  # Both sides
            return 4
    elif number_of_categories == 8:
        return value
    else:
        print("Error converting involvement status!")
