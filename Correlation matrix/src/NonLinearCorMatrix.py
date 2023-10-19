import pandas as pd
import matplotlib.pyplot as plt
def non_linear_cor_matrix(data):

    data.head()

    # Pivot the DataFrame
    pivot_df = data.pivot(columns="non_linear_correlation_type", values="r2score")

    # Plot the bar chart
    pivot_df.plot(kind="bar", stacked=True, figsize=(12, 6))
    plt.title("Bar Chart of Types")
    plt.xlabel("Category")
    plt.ylabel("r2score")
    plt.legend(title="Type")

    # Show the plot
    plt.show()