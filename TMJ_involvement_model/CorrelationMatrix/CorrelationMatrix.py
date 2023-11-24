import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def make_previous_status_correlation_matrix(df_previous_status):

    df_previous_status[['Base', 'Suffix']] = df_previous_status['ID'].str.split('_', expand=True)
    df_previous_status['Suffix'] = pd.to_numeric(df_previous_status['Suffix'])
    result_df = df_previous_status.drop_duplicates('Base', keep='last')
    result_df = result_df[result_df['Suffix'] != 0]
    result_df = result_df.drop(columns=['Suffix', 'Base', 'ID', 'previousstatus'], axis=1).sort_values(by='index').drop(columns=['index'], axis=1)

    plt.close("all")
    corr = result_df.iloc[:, :10].corr()
    fig, ax = plt.subplots(figsize=(32, 24))

    sns.heatmap(corr, cbar=True, vmin=0, vmax=1, fmt='.2f', annot=True, annot_kws={'size': 35}, square=True, cmap="rocket_r")

    tick_labels = ['v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v5', 'v7', 'v8', 'v9']
    ticks = np.arange(corr.shape[0]) + 0.5
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=50)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=50)
    ax.collections[0].colorbar.ax.tick_params(labelsize=40)
    ax.set_title('Correlation matrix')

    plt.tight_layout()
    plt.savefig("CorrelationMatrix/correlation_matrix.png", dpi=300)
    print("Correlation matrix saved as file")

    print()




