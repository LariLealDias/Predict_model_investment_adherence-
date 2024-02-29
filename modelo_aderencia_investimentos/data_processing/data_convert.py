import pandas as pd
def convert_all_columns_in_list_to_numeric(df, collumn_list):
    for column in collumn_list:
        df.loc[:, column] = pd.to_numeric(df[column])
    return df