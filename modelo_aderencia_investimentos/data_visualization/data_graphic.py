import plotly.express as px
import pandas as pd


def generated_graphic_only_numeric_column(df, column_name):
    # print(df[column_name].dtype)

    if column_name not in df.columns:
        raise ValueError(f"There's no column colled '{column_name}' in dataset")
    
    
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        raise TypeError(f"'{column_name}' is not a numeric column")
    
    graphic = px.histogram(df, x = column_name,text_auto=True)
    
    return graphic