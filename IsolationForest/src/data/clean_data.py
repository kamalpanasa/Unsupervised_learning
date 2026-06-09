import pandas as pd
from src.utils import config

def clean_turbofan_data(df):
    """
    Cleans turbofan data.
    """
    return df.dropna(subset=config.META_COLS + config.SETTING_COLS + config.SENSOR_COLS).copy()
