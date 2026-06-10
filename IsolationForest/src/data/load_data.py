import pandas as pd
from src.utils import config

def load_turbofan_data():
    """
    Loads raw Turbofan dataset from local data folder.
    """
    return pd.read_csv(config.TURBOFAN_DATA_PATH)
