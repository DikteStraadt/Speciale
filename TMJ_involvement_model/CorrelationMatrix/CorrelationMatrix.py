import ast

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def generate_seaborn_heatmap(result_df, n):

    if n == "all":
        number_of_columns = 11
        tick_labels = ['v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10']
    elif n == 0:
        number_of_columns = 7
        tick_labels = ['v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
    elif n == 1:
        number_of_columns = 11
        tick_labels = ['v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10']
    elif n == 2:
        number_of_columns = 4
        tick_labels = ['v0', 'v1', 'v2', 'v3']

    plt.close("all")
    corr = result_df.iloc[:, :number_of_columns].corr()
    fig, ax = plt.subplots(figsize=(32, 24))

    sns.heatmap(corr, cbar=True, vmin=0, vmax=1, fmt='.2f', annot=True, annot_kws={'size': 35}, square=True, cmap="rocket_r")

    ticks = np.arange(corr.shape[0]) + 0.5
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=50)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=50)
    ax.collections[0].colorbar.ax.tick_params(labelsize=40)
    ax.set_title(f'Correlation matrix (status={n})', fontsize=60)

    plt.tight_layout()
    plt.savefig(f"CorrelationMatrix/correlation_matrix_status={n}.png", dpi=300)
    print(f"Correlation matrix (status={n}) saved as file")

def make_previous_status_correlation_matrix(df_previous_status, config):

    if config["previous_two_involvement_status"]:
        print("Cannot create correlation matrix when only two previous involvement status is included in visitation")
    else:
        df_previous_status[['Base', 'Suffix']] = df_previous_status['ID'].str.split('_', expand=True)
        df_previous_status['Suffix'] = pd.to_numeric(df_previous_status['Suffix'])
        result_df = df_previous_status.drop_duplicates('Base', keep='last')
        result_df = result_df.dropna(subset=['previousinvolvementstatusvisitation0'])
        last_values = pd.to_numeric(result_df['previousstatus'].apply(lambda x: x.split(', ')[-1].replace('[', '').replace(']', '')))
        result_df['last status'] = last_values

        result_df_0 = result_df[result_df['last status'] == 0]
        result_df_1 = result_df[result_df['last status'] == 1]
        result_df_2 = result_df[result_df['last status'] == 2]

        result_df = result_df.drop(columns=['Suffix', 'Base', 'ID', 'previousstatus', 'last status'],
                                   axis=1).sort_values(by='index').drop(columns=['index'], axis=1)
        result_df_0 = result_df_0.drop(columns=['Suffix', 'Base', 'ID', 'previousstatus', 'last status'],
                                       axis=1).sort_values(by='index').drop(columns=['index'], axis=1)
        result_df_1 = result_df_1.drop(columns=['Suffix', 'Base', 'ID', 'previousstatus', 'last status'],
                                       axis=1).sort_values(by='index').drop(columns=['index'], axis=1)
        result_df_2 = result_df_2.drop(columns=['Suffix', 'Base', 'ID', 'previousstatus', 'last status'],
                                       axis=1).sort_values(by='index').drop(columns=['index'], axis=1)

        generate_seaborn_heatmap(result_df, "all")
        generate_seaborn_heatmap(result_df_0, 0)
        generate_seaborn_heatmap(result_df_1, 1)
        generate_seaborn_heatmap(result_df_2, 2)


