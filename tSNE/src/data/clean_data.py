import pandas as pd
from src.utils import config

def clean_anime_data(df):
    """
    Cleans anime face features.
    """
    return df.dropna(subset=config.GEOM_COLS + config.CAT_COLS).copy()
