import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def confusion_matrix_2_classes(values, file_name):
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(values, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Class 0", "Class 1"],
                yticklabels=["Class 0", "Class 1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(file_name)
    plt.show()

def confusion_matrix_3_classes(values, file_name):
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(values, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Class 0", "Class 1", "Class 2"],
                yticklabels=["Class 0", "Class 1", "Class 2"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(file_name)
    plt.show()

if __name__ == '__main__':
    conf_matrix_two_classes = [[5, 0],
                                [40, 5]]

    conf_matrix_three_classes = [[50, 5, 0],
                                 [10, 40, 5],
                                 [2, 8, 45]]

    confusion_matrix_2_classes(conf_matrix_two_classes, "file_name")
    confusion_matrix_3_classes(conf_matrix_three_classes, "file_name_2")