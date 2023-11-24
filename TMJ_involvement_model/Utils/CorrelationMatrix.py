import pandas as pd


def make_status_correlation_matrix(df_previous_status):

    df_previous_status[['Base', 'Suffix']] = df_previous_status['ID'].str.split('_', expand=True)
    df_previous_status['Suffix'] = pd.to_numeric(df_previous_status['Suffix'])
    result_df = df_previous_status.drop_duplicates('Base', keep='last')
    result_df = result_df[result_df['Suffix'] != 0]
    result_df = result_df.drop(columns=['Suffix', 'ID', 'previousstatus'], axis=1).sort_values(by='index')
    result_df.rename(columns={'Base': 'ID'}, inplace=True)

    print()




