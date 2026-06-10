import pandas as pd
from src.utils import config

def clean_mall_data(df):
    """
    Cleans the Mall Customers dataset.
    """
    df_cleaned = df.dropna(subset=config.NUMERICAL_COLS + ["CustomerID", "Gender"]).copy()
    return df_cleaned
