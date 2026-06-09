import pandas as pd
from src.utils import config

def clean_uber_data(df):
    """
    Ensures no null coordinates exist.
    """
    return df.dropna(subset=config.COORDS_COLS).copy()
