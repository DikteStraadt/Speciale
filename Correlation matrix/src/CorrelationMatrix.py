# https://stackoverflow.com/questions/50997662/how-to-plot-heatmap-for-high-dimensional-dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def correlation_matrix(df, plot_name):

    plt.close("all")
    corr = df.iloc[:, 1:].corr()
    fig, ax = plt.subplots(figsize=(24, 18))

    heatmap = sns.heatmap(corr, cbar=True, vmin=-0.5, vmax=0.5, fmt='.2f', annot_kws={'size': 3}, annot=True, square=True, cmap="YlGnBu")

    ticks = np.arange(corr.shape[0]) + 0.5
    ax.set_xticks(ticks)
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(ticks)
    ax.set_yticklabels(corr.index, rotation=360, fontsize=8)

    ax.set_title('correlation matrix')
    plt.tight_layout()
    plt.savefig(plot_name, dpi=300)