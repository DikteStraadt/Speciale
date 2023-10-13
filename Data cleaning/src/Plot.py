from matplotlib import pyplot as plt
import Utils as u

def trajectory_plot(patients):

    patients = u.convert_visitation_status(patients, "tradjectory")
    interval = 50
    i = 0
    n = 1

    plt.figure(figsize=(10, 6))

    while i <= len(patients):

        legend_labels = []

        for patient in patients[i:i+interval]:

            not_all_zeroes = any(visitation["involvementstatus"] != 0 for visitation in patient)

            if not_all_zeroes:

                X = []  # visitation date
                Y = []  # involvement status

                for visitation in patient:
                    involvement_status = visitation['involvementstatus']
                    visitation_date = visitation['visitationdate']
                    Y.append(involvement_status)
                    X.append(visitation_date)

                plt.xticks(rotation=45, ha='right')
                custom_y_ticks = range(0, len(["No", "Obs", "Right", "Left", "Both"]))
                plt.yticks(custom_y_ticks, ["No", "Obs", "Right", "Left", "Both"])
                plt.yticks(rotation=45, ha='right')
                plt.yticks(range(0, 5))
                plt.xlabel("Visitation date")
                plt.ylabel("Involvement status")

                plt.scatter(X, Y)
                line = plt.plot(X, Y, label=f"Study-id: {int(patient[0]['studyid'])}")
                legend_labels.append(line[0])

        plt.legend(handles=legend_labels, bbox_to_anchor=(1, 1), loc="upper left", fontsize="8")
        plt.title(f'TMJ involvement status tradjectory for patients ({n})')
        plt.tight_layout()
        plt.savefig(f"Tradjectory ({n})")
        plt.show()
        plt.figure(figsize=(10, 6)) # (8, 6)
        i = i + interval
        n = n + 1