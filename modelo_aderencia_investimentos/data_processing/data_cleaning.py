
def clean_row_with_null_values(df, collumn_name):
    return  df.dropna(axis=0, subset=[collumn_name]).copy()



