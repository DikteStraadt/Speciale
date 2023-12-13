import matplotlib.pyplot as plt
import seaborn as sns

def confusion_matrix_2_classes(values, file_name, title):
    plt.figure(figsize=(8, 8))
    sns.heatmap(values, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 28},
                xticklabels=["Class 0", "Class 1"],
                yticklabels=["Class 0", "Class 1"])
    plt.xlabel("Predicted", fontsize=20)
    plt.ylabel("Actual", fontsize=20)
    plt.title(f"Confusion Matrix: {title}", fontsize=23)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(file_name)
    plt.show()

def confusion_matrix_3_classes(values, file_name, title):
    plt.figure(figsize=(8, 8))
    sns.heatmap(values, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 28},
                xticklabels=["Class 0", "Class 1", "Class 2"],
                yticklabels=["Class 0", "Class 1", "Class 2"])
    plt.xlabel("Predicted", fontsize=20)
    plt.ylabel("Actual", fontsize=20)
    plt.title(f"Confusion Matrix: {title}", fontsize=23)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(file_name)
    plt.show()

if __name__ == '__main__':

    experiment_A1 = [[339, 25],
                    [52, 185]]

    experiment_A2 = [[333, 31],
                     [41, 196]]

    experiment_A3 = [[336, 28],
                   [41, 196]]

    experiment_B1 = [[164, 9],
                   [23, 34]]

    experiment_B2 = [[116, 9],
                   [19, 38]]

    experiment_B3 = [[75, 17],
                   [9, 83]]

    experiment_C1 = [[295, 8],
                   [9, 186]]

    experiment_C2 = [[223, 9],
                   [7, 158]]

    experiment_D1 = [[0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]]

    experiment_D2 = [[324, 33, 7],
                   [39, 168, 5],
                   [4, 17, 4]]

    experiment_D3 = [[337, 26, 1],
                   [40, 169, 3],
                   [6, 18, 1]]

    experiment_E1 = [[164, 7, 2],
                   [16, 21, 3],
                   [10, 4, 3]]

    experiment_E2 = [[116, 7, 2],
                   [15, 36, 2],
                   [3, 0, 1]]

    experiment_E3 = [[73, 19, 0],
                   [11, 76, 1],
                   [0, 3, 1]]

    experiment_F1 = [[294, 3, 6],
                   [9, 167, 4],
                   [2, 4, 9]]

    experiment_F2 = [[224, 6, 2],
                   [3, 145, 5],
                   [4, 2, 6]]

    confusion_matrix_2_classes(experiment_A1, "experiment_A1", "Experiment A.1")
    confusion_matrix_2_classes(experiment_A2, "experiment_A2", "Experiment A.2")
    confusion_matrix_2_classes(experiment_A3, "experiment_A3", "Experiment A.3")
    confusion_matrix_2_classes(experiment_B1, "experiment_B1", "Experiment B.1")
    confusion_matrix_2_classes(experiment_B2, "experiment_B2", "Experiment B.2")
    confusion_matrix_2_classes(experiment_B3, "experiment_B3", "Experiment B.3")
    confusion_matrix_2_classes(experiment_C1, "experiment_C1", "Experiment C.1")
    confusion_matrix_2_classes(experiment_C2, "experiment_C2", "Experiment C.2")

    #confusion_matrix_3_classes(experiment_D1, "experiment_D1")
    confusion_matrix_3_classes(experiment_D2, "experiment_D2", "Experiment D.2")
    confusion_matrix_3_classes(experiment_D3, "experiment_D3", "Experiment D.3")
    confusion_matrix_3_classes(experiment_E1, "experiment_E1", "Experiment E.1")
    confusion_matrix_3_classes(experiment_E2, "experiment_E2", "Experiment E.2")
    confusion_matrix_3_classes(experiment_E3, "experiment_E3", "Experiment E.3")
    confusion_matrix_3_classes(experiment_F1, "experiment_F1", "Experiment F.1")
    confusion_matrix_3_classes(experiment_F2, "experiment_F2", "Experiment F.2")